from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gc

from PIL import Image

from .settings import settings


def _tokenize_to_chunks(tokenizer, text: str, *, chunk_length: int) -> tuple[list[list[int]], int]:
    """Tokenize without truncation and split into multiple CLIP-sized chunks.

    Each produced chunk is a list of token IDs (no special tokens included).
    Returns (chunks, max_length) where max_length is tokenizer.model_max_length.
    """

    # CLIP-style tokenizers expect BOS/EOS; we chunk the *inner* tokens.
    max_length = int(getattr(tokenizer, "model_max_length", 77) or 77)
    if max_length < 3:
        max_length = 77

    if not text:
        return [[]], max_length

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_tensors=None,
    )
    ids = encoded["input_ids"]
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        # Batch format
        ids = ids[0]
    if not isinstance(ids, list):
        ids = list(ids)

    inner = ids
    if not inner:
        return [[]], max_length

    chunks: list[list[int]] = []
    for i in range(0, len(inner), chunk_length):
        chunks.append(inner[i : i + chunk_length])
    if not chunks:
        chunks = [[]]
    return chunks, max_length


def _chunks_to_padded_batch(tokenizer, chunks: list[list[int]], *, max_length: int, device):
    import torch

    bos = getattr(tokenizer, "bos_token_id", None)
    eos = getattr(tokenizer, "eos_token_id", None)
    pad = getattr(tokenizer, "pad_token_id", None)

    if bos is None:
        bos = eos if eos is not None else 0
    if eos is None:
        eos = bos
    if pad is None:
        pad = eos

    input_ids: list[list[int]] = []
    attention: list[list[int]] = []

    for inner in chunks:
        ids = [int(bos)] + [int(x) for x in inner] + [int(eos)]
        ids = ids[:max_length]
        attn = [1] * len(ids)
        if len(ids) < max_length:
            pad_len = max_length - len(ids)
            ids = ids + [int(pad)] * pad_len
            attn = attn + [0] * pad_len
        input_ids.append(ids)
        attention.append(attn)

    return (
        torch.tensor(input_ids, dtype=torch.long, device=device),
        torch.tensor(attention, dtype=torch.long, device=device),
    )


def _encode_text_encoder_chunks(text_encoder, tokenizer, text: str, *, n_chunks: int, device):
    import torch

    max_length = int(getattr(tokenizer, "model_max_length", 77) or 77)
    chunk_length = max(1, max_length - 2)
    chunks, _ = _tokenize_to_chunks(tokenizer, text or "", chunk_length=chunk_length)

    # Pad with empty chunks to match a requested global chunk count.
    if len(chunks) < n_chunks:
        chunks = list(chunks) + ([[]] * (n_chunks - len(chunks)))
    elif len(chunks) > n_chunks:
        chunks = chunks[:n_chunks]

    input_ids, attention = _chunks_to_padded_batch(tokenizer, chunks, max_length=max_length, device=device)

    out = text_encoder(
        input_ids,
        attention_mask=attention,
        output_hidden_states=True,
        return_dict=True,
    )

    # Diffusers commonly uses the penultimate hidden state + final_layer_norm for CLIP text encoders.
    hidden_states = getattr(out, "hidden_states", None)
    if hidden_states is not None and len(hidden_states) >= 2:
        prompt_embeds = hidden_states[-2]
        text_model = getattr(text_encoder, "text_model", None)
        final_ln = getattr(text_model, "final_layer_norm", None) if text_model is not None else None
        if callable(final_ln):
            prompt_embeds = final_ln(prompt_embeds)
    else:
        prompt_embeds = getattr(out, "last_hidden_state", None)
        if prompt_embeds is None:
            # Last resort: assume tuple-like output.
            prompt_embeds = out[0]

    pooled = None
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        pooled = out.pooler_output
    elif hasattr(out, "text_embeds") and out.text_embeds is not None:
        pooled = out.text_embeds
    elif isinstance(out, (tuple, list)) and len(out) > 1:
        pooled = out[1]
    else:
        # Fallback: use CLS token.
        pooled = prompt_embeds[:, 0]

    # Concatenate chunks along the sequence dimension; average pooled across chunks.
    prompt_embeds = prompt_embeds.reshape(n_chunks, max_length, -1)
    pooled = pooled.reshape(n_chunks, -1)
    prompt_embeds = torch.cat([prompt_embeds[i : i + 1] for i in range(n_chunks)], dim=1)
    pooled = pooled.mean(dim=0, keepdim=True)

    return prompt_embeds, pooled


def _maybe_build_long_prompt_kwargs(pipe: Any, prompt: str, negative_prompt: str, *, device: str):
    """If needed, build SDXL prompt embedding kwargs to avoid 77-token truncation."""

    mode = (settings.long_prompt_mode or "chunk").lower().strip()
    if mode not in {"chunk", "truncate", "error"}:
        mode = "chunk"

    if mode == "truncate":
        return None

    # Only SDXL pipelines have tokenizer_2/text_encoder_2.
    tokenizer_1 = getattr(pipe, "tokenizer", None)
    tokenizer_2 = getattr(pipe, "tokenizer_2", None)
    text_encoder_1 = getattr(pipe, "text_encoder", None)
    text_encoder_2 = getattr(pipe, "text_encoder_2", None)
    if tokenizer_1 is None or tokenizer_2 is None or text_encoder_1 is None or text_encoder_2 is None:
        return None

    # Figure out how many chunks we need (max across both tokenizers + both prompts).
    max_len_1 = int(getattr(tokenizer_1, "model_max_length", 77) or 77)
    max_len_2 = int(getattr(tokenizer_2, "model_max_length", 77) or 77)
    chunk_len_1 = max(1, max_len_1 - 2)
    chunk_len_2 = max(1, max_len_2 - 2)

    p_chunks_1, _ = _tokenize_to_chunks(tokenizer_1, prompt or "", chunk_length=chunk_len_1)
    p_chunks_2, _ = _tokenize_to_chunks(tokenizer_2, prompt or "", chunk_length=chunk_len_2)
    n_chunks_1, _ = _tokenize_to_chunks(tokenizer_1, negative_prompt or "", chunk_length=chunk_len_1)
    n_chunks_2, _ = _tokenize_to_chunks(tokenizer_2, negative_prompt or "", chunk_length=chunk_len_2)

    n_chunks = max(len(p_chunks_1), len(p_chunks_2), len(n_chunks_1), len(n_chunks_2), 1)

    # If everything fits in one chunk, there's nothing to do.
    if n_chunks <= 1:
        return None

    if mode == "error":
        raise ValueError(
            f"Prompt is too long for CLIP (needs {n_chunks} chunks). "
            "Shorten the prompt or set SDUI_LONG_PROMPT_MODE=chunk."
        )

    # When using cpu offload, diffusers usually wants tensors on the pipeline execution device.
    execution_device = getattr(pipe, "_execution_device", None) or getattr(pipe, "device", None) or device

    prompt_embeds_1, pooled_1 = _encode_text_encoder_chunks(
        text_encoder_1, tokenizer_1, prompt or "", n_chunks=n_chunks, device=execution_device
    )
    prompt_embeds_2, pooled_2 = _encode_text_encoder_chunks(
        text_encoder_2, tokenizer_2, prompt or "", n_chunks=n_chunks, device=execution_device
    )

    neg_embeds_1, neg_pooled_1 = _encode_text_encoder_chunks(
        text_encoder_1, tokenizer_1, negative_prompt or "", n_chunks=n_chunks, device=execution_device
    )
    neg_embeds_2, neg_pooled_2 = _encode_text_encoder_chunks(
        text_encoder_2, tokenizer_2, negative_prompt or "", n_chunks=n_chunks, device=execution_device
    )

    import torch

    prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
    negative_prompt_embeds = torch.cat([neg_embeds_1, neg_embeds_2], dim=-1)

    # SDXL uses pooled embeddings from the second encoder for additional conditioning.
    pooled_prompt_embeds = pooled_2
    negative_pooled_prompt_embeds = neg_pooled_2

    # Match pipeline dtype for numerical stability.
    try:
        target_dtype = getattr(getattr(pipe, "unet", None), "dtype", None) or prompt_embeds.dtype
        prompt_embeds = prompt_embeds.to(dtype=target_dtype)
        negative_prompt_embeds = negative_prompt_embeds.to(dtype=target_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=target_dtype)
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(dtype=target_dtype)
    except Exception:
        pass

    return {
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
    }


@dataclass(frozen=True)
class LoadedPipelines:
    base_id: str
    refiner_id: str | None
    device: str
    dtype: str
    base: Any
    refiner: Any | None


def _resolve_device(device: str) -> str:
    device = (device or "auto").lower()
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    if device in {"cuda", "cpu"}:
        return device
    raise ValueError("device must be auto|cuda|cpu")


def _resolve_dtype(dtype: str):
    dtype = (dtype or settings.default_dtype).lower()
    try:
        import torch

        if dtype == "float16":
            return torch.float16
        if dtype == "bfloat16":
            return torch.bfloat16
        if dtype == "float32":
            return torch.float32
    except Exception:
        pass
    raise ValueError("dtype must be float16|bfloat16|float32")


def _scheduler_from_name(name: str):
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
    from diffusers.schedulers.scheduling_euler_ancestral_discrete import (
        EulerAncestralDiscreteScheduler,
    )
    from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
    from diffusers.schedulers.scheduling_heun_discrete import HeunDiscreteScheduler
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    n = (name or "").strip().lower()
    if n in {"dpm++ 2m karras", "dpmpp 2m karras", "dpm 2m karras"}:
        return DPMSolverMultistepScheduler
    if n in {"unipc", "unipc multistep"}:
        return UniPCMultistepScheduler
    if n in {"euler"}:
        return EulerDiscreteScheduler
    if n in {"euler a", "euler ancestral"}:
        return EulerAncestralDiscreteScheduler
    if n in {"heun"}:
        return HeunDiscreteScheduler
    if n in {"ddim"}:
        return DDIMScheduler

    return DPMSolverMultistepScheduler


def _infer_prediction_type_from_name(model_name: str) -> str:
    n = (model_name or "").lower()
    # Heuristics for NoobAI/NovelAI-style naming.
    if "v-pred" in n or "v_pred" in n or "v prediction" in n or "v_prediction" in n:
        return "v_prediction"
    if "eps" in n or "epsilon" in n or "eps-pred" in n:
        return "epsilon"
    return "epsilon"


def _apply_scheduler_config(scheduler_cls, scheduler, *, prediction_type: str, rescale_betas_zero_snr: bool):
    # Try the most direct approach first.
    kwargs: dict[str, object] = {}
    if prediction_type in {"epsilon", "v_prediction"}:
        kwargs["prediction_type"] = prediction_type
    if rescale_betas_zero_snr:
        kwargs["rescale_betas_zero_snr"] = True

    if kwargs:
        try:
            return scheduler_cls.from_config(scheduler.config, **kwargs)
        except TypeError:
            pass

    # Fallback: rebuild scheduler then register overrides if supported.
    rebuilt = scheduler_cls.from_config(scheduler.config)
    if kwargs and hasattr(rebuilt, "register_to_config"):
        try:
            rebuilt.register_to_config(**kwargs)
        except Exception:
            pass
    return rebuilt


def _abs_in_models(rel: str) -> Path:
    if not rel:
        raise ValueError("missing path")
    models_dir = settings.models_dir.resolve()
    candidate = (models_dir / rel).resolve()
    if models_dir not in candidate.parents and candidate != models_dir:
        raise ValueError("path must be inside models_dir")
    return candidate


def _is_single_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {".safetensors", ".ckpt"}


class PipelineManager:
    def __init__(self) -> None:
        self._loaded: LoadedPipelines | None = None
        self._original_base_vae: Any | None = None
        self._original_refiner_vae: Any | None = None
        self._applied_vae: str | None = None
        self._applied_loras: tuple[str, ...] = ()
        self._applied_lora_scale: float = 0.0
        self._cpu_offload_enabled: bool = False

    def _reset_dynamic_state(self, *, base_pipe: Any, refiner_pipe: Any | None) -> None:
        self._original_base_vae = getattr(base_pipe, "vae", None)
        self._original_refiner_vae = getattr(refiner_pipe, "vae", None) if refiner_pipe is not None else None
        self._applied_vae = None
        self._applied_loras = ()
        self._applied_lora_scale = 0.0
        self._cpu_offload_enabled = False

    def _try_unload_loras(self, pipe: Any) -> bool:
        if pipe is None:
            return False
        if hasattr(pipe, "unload_lora_weights"):
            try:
                pipe.unload_lora_weights()
                return True
            except Exception:
                return False
        return False

    def _build_base(self, base_path: Path, device: str, dtype):
        from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
            StableDiffusionXLPipeline,
        )

        if _is_single_file(base_path):
            if not hasattr(StableDiffusionXLPipeline, "from_single_file"):
                raise RuntimeError(
                    "Your diffusers version does not support from_single_file(). "
                    "Use a Diffusers-format SDXL directory instead."
                )
            pipe = StableDiffusionXLPipeline.from_single_file(
                str(base_path),
                torch_dtype=dtype,
                use_safetensors=base_path.suffix.lower() == ".safetensors",
            )
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                str(base_path),
                torch_dtype=dtype,
                use_safetensors=True,
                local_files_only=True,
            )

        pipe.to(device)
        pipe.enable_vae_slicing()
        pipe.enable_attention_slicing()
        return pipe

    def _build_refiner(self, refiner_path: Path, device: str, dtype):
        from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import (
            StableDiffusionXLImg2ImgPipeline,
        )

        if _is_single_file(refiner_path):
            if not hasattr(StableDiffusionXLImg2ImgPipeline, "from_single_file"):
                raise RuntimeError(
                    "Your diffusers version does not support from_single_file(). "
                    "Use a Diffusers-format SDXL refiner directory instead."
                )
            pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
                str(refiner_path),
                torch_dtype=dtype,
                use_safetensors=refiner_path.suffix.lower() == ".safetensors",
            )
        else:
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                str(refiner_path),
                torch_dtype=dtype,
                use_safetensors=True,
                local_files_only=True,
            )

        pipe.to(device)
        pipe.enable_vae_slicing()
        pipe.enable_attention_slicing()
        return pipe

    def get(
        self,
        *,
        base_model: str,
        refiner_model: str | None,
        device: str,
        dtype: str,
        scheduler_name: str,
        prediction_type: str,
        rescale_betas_zero_snr: bool,
        vae: str | None,
        loras: list[str],
        lora_scale: float,
    ):
        import torch

        resolved_device = _resolve_device(device)
        resolved_dtype = _resolve_dtype(dtype)

        base_path = _abs_in_models(base_model)
        refiner_path = _abs_in_models(refiner_model) if refiner_model else None

        cache_key = (base_model, refiner_model, resolved_device, str(resolved_dtype).replace("torch.", ""))
        if (
            self._loaded is None
            or (self._loaded.base_id, self._loaded.refiner_id, self._loaded.device, self._loaded.dtype)
            != cache_key
        ):
            base_pipe = self._build_base(base_path, resolved_device, resolved_dtype)
            refiner_pipe = (
                self._build_refiner(refiner_path, resolved_device, resolved_dtype) if refiner_path else None
            )
            self._loaded = LoadedPipelines(
                base_id=base_model,
                refiner_id=refiner_model,
                device=resolved_device,
                dtype=cache_key[3],
                base=base_pipe,
                refiner=refiner_pipe,
            )
            self._reset_dynamic_state(base_pipe=base_pipe, refiner_pipe=refiner_pipe)

        base_pipe = self._loaded.base
        refiner_pipe = self._loaded.refiner

        scheduler_cls = _scheduler_from_name(scheduler_name)

        if prediction_type == "auto":
            prediction_type = _infer_prediction_type_from_name(base_model)
        if prediction_type not in {"epsilon", "v_prediction"}:
            raise ValueError("prediction_type must be auto|epsilon|v_prediction")

        base_pipe.scheduler = _apply_scheduler_config(
            scheduler_cls,
            base_pipe.scheduler,
            prediction_type=prediction_type,
            rescale_betas_zero_snr=rescale_betas_zero_snr,
        )
        if refiner_pipe is not None:
            refiner_pipe.scheduler = _apply_scheduler_config(
                scheduler_cls,
                refiner_pipe.scheduler,
                prediction_type=prediction_type,
                rescale_betas_zero_snr=rescale_betas_zero_snr,
            )

        if vae != self._applied_vae:
            from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
            if not vae:
                # Revert to original model VAE.
                if self._original_base_vae is not None:
                    base_pipe.vae = self._original_base_vae
                if refiner_pipe is not None and self._original_refiner_vae is not None:
                    refiner_pipe.vae = self._original_refiner_vae
                self._applied_vae = None
            else:
                vae_path = _abs_in_models(vae)
                if _is_single_file(vae_path):
                    if not hasattr(AutoencoderKL, "from_single_file"):
                        raise RuntimeError(
                            "Your diffusers version does not support VAE from_single_file(). "
                            "Use a diffusers-format VAE folder instead."
                        )
                    vae_model = AutoencoderKL.from_single_file(
                        str(vae_path),
                        torch_dtype=resolved_dtype,
                    )
                else:
                    vae_model = AutoencoderKL.from_pretrained(
                        str(vae_path),
                        torch_dtype=resolved_dtype,
                        local_files_only=True,
                    )

                vae_model: Any = vae_model
                vae_model.to(resolved_device)
                base_pipe.vae = vae_model
                if refiner_pipe is not None:
                    refiner_pipe.vae = vae_model
                self._applied_vae = vae

        requested_loras = tuple(loras or [])
        requested_lora_scale = float(lora_scale)

        if requested_loras != self._applied_loras or requested_lora_scale != float(self._applied_lora_scale):
            # Best-effort: unload previous LoRAs to avoid adapter accumulation over many generations.
            unloaded_base = self._try_unload_loras(base_pipe)
            unloaded_refiner = self._try_unload_loras(refiner_pipe) if refiner_pipe is not None else False

            if requested_loras:
                try:
                    import peft  # noqa: F401
                except Exception as e:
                    raise RuntimeError(
                        "LoRA loading requires the PEFT backend. Install it with `pip install -r requirements.txt` "
                        "(or `pip install peft`) and restart the app."
                    ) from e

                adapter_names: list[str] = []
                for idx, lora_rel in enumerate(requested_loras):
                    lora_path = _abs_in_models(lora_rel)
                    adapter_name = f"lora_{idx}"
                    try:
                        base_pipe.load_lora_weights(str(lora_path), adapter_name=adapter_name)
                        adapter_names.append(adapter_name)
                        if refiner_pipe is not None:
                            refiner_pipe.load_lora_weights(str(lora_path), adapter_name=adapter_name)
                    except ValueError as e:
                        # Newer diffusers requires PEFT for LoRA loading.
                        if "PEFT backend is required" in str(e):
                            raise RuntimeError(
                                "LoRA loading requires the PEFT backend. Install it with `pip install -r requirements.txt` "
                                "(or `pip install peft`) and restart the app."
                            ) from e
                        raise
                    except TypeError:
                        base_pipe.load_lora_weights(str(lora_path))
                        if refiner_pipe is not None:
                            refiner_pipe.load_lora_weights(str(lora_path))

                if adapter_names and hasattr(base_pipe, "set_adapters"):
                    weights = [float(requested_lora_scale)] * len(adapter_names)
                    base_pipe.set_adapters(adapter_names, adapter_weights=weights)
                    if refiner_pipe is not None:
                        refiner_pipe.set_adapters(adapter_names, adapter_weights=weights)
                elif hasattr(base_pipe, "fuse_lora"):
                    # If we can't unload adapters reliably, reloading/fusing repeatedly can stack.
                    # In that case, force a pipeline rebuild next time LoRAs change.
                    if not unloaded_base and self._applied_loras and requested_loras != self._applied_loras:
                        self._loaded = None
                        return self.get(
                            base_model=base_model,
                            refiner_model=refiner_model,
                            device=device,
                            dtype=dtype,
                            scheduler_name=scheduler_name,
                            prediction_type=prediction_type,
                            rescale_betas_zero_snr=rescale_betas_zero_snr,
                            vae=vae,
                            loras=list(requested_loras),
                            lora_scale=requested_lora_scale,
                        )
                    base_pipe.fuse_lora(lora_scale=float(requested_lora_scale))
                    if refiner_pipe is not None:
                        if not unloaded_refiner and self._applied_loras and requested_loras != self._applied_loras:
                            self._loaded = None
                            return self.get(
                                base_model=base_model,
                                refiner_model=refiner_model,
                                device=device,
                                dtype=dtype,
                                scheduler_name=scheduler_name,
                                prediction_type=prediction_type,
                                rescale_betas_zero_snr=rescale_betas_zero_snr,
                                vae=vae,
                                loras=list(requested_loras),
                                lora_scale=requested_lora_scale,
                            )
                        refiner_pipe.fuse_lora(lora_scale=float(requested_lora_scale))

            # If we're disabling LoRAs but can't unload them, rebuild to avoid keeping fused weights.
            if not requested_loras and self._applied_loras:
                can_unload = hasattr(base_pipe, "unload_lora_weights")
                if not can_unload:
                    self._loaded = None
                    return self.get(
                        base_model=base_model,
                        refiner_model=refiner_model,
                        device=device,
                        dtype=dtype,
                        scheduler_name=scheduler_name,
                        prediction_type=prediction_type,
                        rescale_betas_zero_snr=rescale_betas_zero_snr,
                        vae=vae,
                        loras=[],
                        lora_scale=requested_lora_scale,
                    )

            self._applied_loras = requested_loras
            self._applied_lora_scale = float(requested_lora_scale)

        if resolved_device == "cuda" and not self._cpu_offload_enabled:
            try:
                base_pipe.enable_model_cpu_offload()
                if refiner_pipe is not None:
                    refiner_pipe.enable_model_cpu_offload()
                self._cpu_offload_enabled = True
            except Exception:
                self._cpu_offload_enabled = False

        torch.backends.cuda.matmul.allow_tf32 = True

        return base_pipe, refiner_pipe, resolved_device, resolved_dtype


PIPELINES = PipelineManager()


def _random_seed() -> int:
    return secrets.randbelow(2**31 - 1)


def _save_images(images: list[Image.Image], seed: int) -> list[str]:
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    paths: list[str] = []
    for i, img in enumerate(images, start=1):
        name = f"{stamp}_seed{seed}_{i}.png"
        out_path = (settings.outputs_dir / name).resolve()
        img.save(out_path, format="PNG", optimize=True)
        paths.append(out_path.name)
    return paths


def generate(
    *,
    base_model: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    scheduler: str,
    prediction_type: str,
    rescale_betas_zero_snr: bool,
    guidance_rescale: float,
    seed: int,
    num_images: int,
    enable_refiner: bool,
    refiner_model: str | None,
    refiner_strength: float,
    refiner_steps: int,
    vae: str | None,
    loras: list[str],
    lora_scale: float,
    device: str,
    dtype: str,
    return_images: bool = True,
) -> tuple[int, list[Image.Image], list[str], int | None]:
    import torch

    if not prompt.strip():
        raise ValueError("prompt is required")

    used_seed = _random_seed() if seed == -1 else int(seed)

    base_pipe, refiner_pipe, resolved_device, resolved_dtype = PIPELINES.get(
        base_model=base_model,
        refiner_model=refiner_model if enable_refiner else None,
        device=device,
        dtype=dtype,
        scheduler_name=scheduler,
        prediction_type=prediction_type,
        rescale_betas_zero_snr=rescale_betas_zero_snr,
        vae=vae,
        loras=loras,
        lora_scale=lora_scale,
    )

    gen_device = "cuda" if resolved_device == "cuda" else "cpu"
    generator = torch.Generator(device=gen_device).manual_seed(used_seed)

    base_kwargs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images,
        generator=generator,
        output_type="pil",
    )
    if guidance_rescale and float(guidance_rescale) > 0:
        base_kwargs["guidance_rescale"] = float(guidance_rescale)

    # Handle prompts longer than CLIP's 77-token limit.
    long_prompt_kwargs = _maybe_build_long_prompt_kwargs(
        base_pipe,
        prompt,
        negative_prompt,
        device=resolved_device,
    )
    if long_prompt_kwargs:
        # Ensure diffusers does not try to tokenize text when embeddings are supplied.
        base_kwargs.pop("prompt", None)
        base_kwargs.pop("negative_prompt", None)
        base_kwargs.update(long_prompt_kwargs)

    images: list[Image.Image] = []
    paths: list[str] = []
    out = None
    try:
        with torch.inference_mode():
            try:
                out = base_pipe(**base_kwargs)
            except TypeError:
                # Older diffusers versions may not support guidance_rescale.
                base_kwargs.pop("guidance_rescale", None)
                out = base_pipe(**base_kwargs)

        images = list(out.images)

        if enable_refiner:
            if refiner_pipe is None:
                raise ValueError("refiner enabled but refiner_model not provided")

            refiner_long_prompt_kwargs = None
            try:
                refiner_long_prompt_kwargs = _maybe_build_long_prompt_kwargs(
                    refiner_pipe,
                    prompt,
                    negative_prompt,
                    device=resolved_device,
                )
            except Exception:
                refiner_long_prompt_kwargs = None

            refined: list[Image.Image] = []
            with torch.inference_mode():
                for idx, img in enumerate(images):
                    rgen = torch.Generator(device=gen_device).manual_seed(used_seed + idx + 1)
                    ref_kwargs = dict(
                        prompt=prompt,
                        negative_prompt=negative_prompt or None,
                        image=img,
                        strength=float(refiner_strength),
                        num_inference_steps=int(refiner_steps),
                        guidance_scale=guidance_scale,
                        generator=rgen,
                    )
                    if refiner_long_prompt_kwargs:
                        ref_kwargs.pop("prompt", None)
                        ref_kwargs.pop("negative_prompt", None)
                        ref_kwargs.update(refiner_long_prompt_kwargs)
                    r = refiner_pipe(**ref_kwargs)
                    refined.append(r.images[0])
                    del r
            images = refined

        paths = _save_images(images, used_seed)

        if not return_images:
            for img in images:
                try:
                    img.close()
                except Exception:
                    pass
            images = []

    finally:
        # Ensure we don't keep large intermediates alive.
        del out
        if settings.cleanup_after_generate:
            gc.collect()
            if resolved_device == "cuda" and settings.empty_cuda_cache:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    history_id: int | None = None
    try:
        from .history import add_generation

        params = {
            "width": int(width),
            "height": int(height),
            "steps": int(steps),
            "guidance_scale": float(guidance_scale),
            "scheduler": str(scheduler),
            "prediction_type": str(prediction_type),
            "rescale_betas_zero_snr": bool(rescale_betas_zero_snr),
            "guidance_rescale": float(guidance_rescale),
            "num_images": int(num_images),
            "vae": vae,
            "loras": list(loras),
            "lora_scale": float(lora_scale),
            "enable_refiner": bool(enable_refiner),
            "refiner_model": refiner_model,
            "refiner_strength": float(refiner_strength),
            "refiner_steps": int(refiner_steps),
            "device": str(device),
            "dtype": str(dtype),
        }
        history_id = add_generation(
            seed=used_seed,
            base_model=base_model,
            prompt=prompt,
            negative_prompt=negative_prompt,
            params=params,
            image_files=paths,
        )
    except Exception:
        history_id = None

    return used_seed, images, paths, history_id
