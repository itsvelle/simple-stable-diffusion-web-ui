from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .settings import settings


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

        if vae:
            from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

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

        if loras:
            adapter_names: list[str] = []
            for idx, lora_rel in enumerate(loras):
                lora_path = _abs_in_models(lora_rel)
                adapter_name = f"lora_{idx}"
                try:
                    base_pipe.load_lora_weights(str(lora_path), adapter_name=adapter_name)
                    adapter_names.append(adapter_name)
                    if refiner_pipe is not None:
                        refiner_pipe.load_lora_weights(str(lora_path), adapter_name=adapter_name)
                except TypeError:
                    base_pipe.load_lora_weights(str(lora_path))
                    if refiner_pipe is not None:
                        refiner_pipe.load_lora_weights(str(lora_path))

            if adapter_names and hasattr(base_pipe, "set_adapters"):
                weights = [float(lora_scale)] * len(adapter_names)
                base_pipe.set_adapters(adapter_names, adapter_weights=weights)
                if refiner_pipe is not None:
                    refiner_pipe.set_adapters(adapter_names, adapter_weights=weights)
            elif hasattr(base_pipe, "fuse_lora"):
                base_pipe.fuse_lora(lora_scale=float(lora_scale))
                if refiner_pipe is not None:
                    refiner_pipe.fuse_lora(lora_scale=float(lora_scale))

        if resolved_device == "cuda":
            try:
                base_pipe.enable_model_cpu_offload()
                if refiner_pipe is not None:
                    refiner_pipe.enable_model_cpu_offload()
            except Exception:
                pass

        torch.backends.cuda.matmul.allow_tf32 = True

        return base_pipe, refiner_pipe, resolved_device


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
) -> tuple[int, list[Image.Image], list[str], int | None]:
    import torch

    if not prompt.strip():
        raise ValueError("prompt is required")

    used_seed = _random_seed() if seed == -1 else int(seed)

    base_pipe, refiner_pipe, resolved_device = PIPELINES.get(
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

    try:
        out = base_pipe(**base_kwargs)
    except TypeError:
        # Older diffusers versions may not support guidance_rescale.
        base_kwargs.pop("guidance_rescale", None)
        out = base_pipe(**base_kwargs)

    images: list[Image.Image] = list(out.images)

    if enable_refiner:
        if refiner_pipe is None:
            raise ValueError("refiner enabled but refiner_model not provided")

        refined: list[Image.Image] = []
        for idx, img in enumerate(images):
            rgen = torch.Generator(device=gen_device).manual_seed(used_seed + idx + 1)
            r = refiner_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                image=img,
                strength=float(refiner_strength),
                num_inference_steps=int(refiner_steps),
                guidance_scale=guidance_scale,
                generator=rgen,
            )
            refined.append(r.images[0])
        images = refined

    paths = _save_images(images, used_seed)

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
