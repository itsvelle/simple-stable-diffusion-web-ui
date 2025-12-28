from __future__ import annotations

from pathlib import Path

import gradio as gr

from .history import format_choice, get_generation, list_generations
from .model_scanner import list_model_entries
from .pipeline import generate
from .settings import settings


SCHEDULERS = [
    "DPM++ 2M Karras",
    "UniPC",
    "Euler",
    "Euler a",
    "Heun",
    "DDIM",
]


def _choices(subdir: str) -> list[str]:
    base = settings.models_dir / subdir
    return list_model_entries(base)


def build_ui() -> gr.Blocks:
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)

    base_choices = _choices("base")
    vae_choices = ["(none)"] + _choices("vae")
    lora_choices = _choices("lora")

    with gr.Blocks(title="Simple Stable Diffusion UI") as demo:
        gr.Markdown(
            "# Simple Stable Diffusion UI\n"
            "Local-only loading from `models/` and outputs saved to `outputs/`."
        )

        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(label="Prompt", lines=4, placeholder="A cinematic photo of …")
                negative = gr.Textbox(label="Negative Prompt", lines=3)

                with gr.Accordion("Models", open=True):
                    refresh_models = gr.Button("Refresh model lists")
                    base_model = gr.Dropdown(
                        label="Base model (models/base)",
                        choices=base_choices,
                        value=base_choices[0] if base_choices else None,
                        interactive=True,
                    )
                    vae = gr.Dropdown(label="VAE (models/vae)", choices=vae_choices, value="(none)")
                    loras = gr.Dropdown(
                        label="LoRAs (models/lora)",
                        choices=lora_choices,
                        multiselect=True,
                        value=[],
                    )
                    lora_scale = gr.Slider(
                        label="LoRA scale",
                        minimum=0.0,
                        maximum=2.0,
                        value=0.8,
                        step=0.05,
                    )

                with gr.Accordion("Sampling", open=True):
                    with gr.Row():
                        width = gr.Slider(256, 2048, value=1024, step=64, label="Width")
                        height = gr.Slider(256, 2048, value=1024, step=64, label="Height")
                    with gr.Row():
                        steps = gr.Slider(1, 150, value=30, step=1, label="Steps")
                        cfg = gr.Slider(0, 30, value=5.0, step=0.5, label="CFG")
                    scheduler = gr.Dropdown(label="Scheduler", choices=SCHEDULERS, value=SCHEDULERS[0])

                with gr.Accordion("NoobAI / V-pred options", open=False):
                    prediction_type = gr.Dropdown(
                        label="Prediction type",
                        choices=["auto", "epsilon", "v_prediction"],
                        value="auto",
                        info="NoobAI-XL v-pred models require v_prediction.",
                    )
                    rescale_betas_zero_snr = gr.Checkbox(
                        label="rescale_betas_zero_snr",
                        value=True,
                        info="Recommended for many v-pred models.",
                    )
                    guidance_rescale = gr.Slider(
                        label="CFG rescale (guidance_rescale)",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.0,
                        step=0.05,
                        info="For v-pred models, ~0.7 can help reduce overexposure/oversaturation.",
                    )

                with gr.Accordion("Refiner (optional)", open=False):
                    enable_refiner = gr.Checkbox(label="Enable SDXL refiner", value=False)
                    refiner_model = gr.Dropdown(
                        label="Refiner model (models/refiner)",
                        choices=_choices("refiner"),
                        value=None,
                    )
                    refiner_strength = gr.Slider(0, 1, value=0.2, step=0.05, label="Strength")
                    refiner_steps = gr.Slider(1, 100, value=20, step=1, label="Refiner steps")

                with gr.Accordion("Runtime", open=False):
                    device = gr.Dropdown(label="Device", choices=["auto", "cuda", "cpu"], value="auto")
                    dtype = gr.Dropdown(
                        label="DType",
                        choices=["float16", "bfloat16", "float32"],
                        value=settings.default_dtype,
                    )

                with gr.Accordion("Seed / Batch", open=True):
                    seed = gr.Number(label="Seed (-1 random)", value=-1, precision=0)
                    num_images = gr.Slider(1, 8, value=1, step=1, label="Images")

                run_btn = gr.Button("Generate", variant="primary")

                status = gr.Markdown("")

            with gr.Column(scale=3):
                gallery = gr.Gallery(label="Results", show_label=True, columns=2, height=720)
                seed_out = gr.Number(label="Used seed", precision=0)
                history_id_out = gr.Number(label="History ID", precision=0)
                paths_out = gr.Textbox(
                    label="Saved to outputs/ (filenames)", lines=4, interactive=False
                )

                with gr.Accordion("History", open=False):
                    history_refresh = gr.Button("Refresh history")
                    history_limit = gr.Slider(5, 200, value=50, step=5, label="Show last N")
                    history_pick = gr.Dropdown(label="Pick a past generation", choices=[], value=None)
                    history_preview = gr.Gallery(
                        label="Selected generation images", columns=2, height=360
                    )
                    load_btn = gr.Button("Load settings into form")

        def _run(
            base_model: str,
            vae: str,
            loras: list[str],
            lora_scale: float,
            prompt: str,
            negative: str,
            width: int,
            height: int,
            steps: int,
            cfg: float,
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
            device: str,
            dtype: str,
        ):
            if not base_model:
                raise gr.Error("No base model found. Put an SDXL model in models/base")

            used_seed, images, saved, history_id = generate(
                base_model=str(Path("base") / base_model),
                prompt=prompt,
                negative_prompt=negative,
                width=int(width),
                height=int(height),
                steps=int(steps),
                guidance_scale=float(cfg),
                scheduler=scheduler,
                prediction_type=prediction_type,
                rescale_betas_zero_snr=bool(rescale_betas_zero_snr),
                guidance_rescale=float(guidance_rescale),
                seed=int(seed),
                num_images=int(num_images),
                enable_refiner=bool(enable_refiner),
                refiner_model=(str(Path("refiner") / refiner_model) if (enable_refiner and refiner_model) else None),
                refiner_strength=float(refiner_strength),
                refiner_steps=int(refiner_steps),
                vae=None if vae == "(none)" else str(Path("vae") / vae),
                loras=[str(Path("lora") / x) for x in (loras or [])],
                lora_scale=float(lora_scale),
                device=device,
                dtype=dtype,
            )
            files_list = "\n".join(saved)
            links = "\n".join([f"/outputs/{p}" for p in saved])
            msg = f"Saved {len(saved)} image(s).\\n{links}"
            return images, used_seed, history_id or 0, files_list, msg

        def _history_choices(limit: int):
            items = list_generations(limit=int(limit))
            return [format_choice(i) for i in items]

        def _history_preview(choice: str):
            if not choice:
                return []
            gen_id = int(choice.split("|", 1)[0].strip())
            item = get_generation(gen_id)
            if item is None:
                return []
            # Show images via the outputs directory.
            return [str((settings.outputs_dir / f).resolve()) for f in item.image_files]

        def _load_from_history(choice: str):
            if not choice:
                raise gr.Error("Pick a history item first")
            gen_id = int(choice.split("|", 1)[0].strip())
            item = get_generation(gen_id)
            if item is None:
                raise gr.Error("History item not found")

            p = item.params

            # Stored paths are the internal backend paths (e.g. base/xxx.safetensors).
            base_rel = str(p.get("base_model", item.base_model))
            if base_rel.startswith("base/"):
                base_rel = base_rel[len("base/") :]

            vae_rel = p.get("vae")
            if isinstance(vae_rel, str) and vae_rel.startswith("vae/"):
                vae_rel = vae_rel[len("vae/") :]

            loras_rel = p.get("loras") or []
            if isinstance(loras_rel, list):
                loras_rel = [x[len("lora/") :] if isinstance(x, str) and x.startswith("lora/") else x for x in loras_rel]

            refiner_rel = p.get("refiner_model")
            if isinstance(refiner_rel, str) and refiner_rel.startswith("refiner/"):
                refiner_rel = refiner_rel[len("refiner/") :]

            return (
                item.prompt,
                item.negative_prompt,
                base_rel,
                "(none)" if not vae_rel else vae_rel,
                loras_rel,
                float(p.get("lora_scale", 0.8)),
                int(p.get("width", 1024)),
                int(p.get("height", 1024)),
                int(p.get("steps", 30)),
                float(p.get("guidance_scale", 5.0)),
                str(p.get("scheduler", SCHEDULERS[0])),
                str(p.get("prediction_type", "auto")),
                bool(p.get("rescale_betas_zero_snr", True)),
                float(p.get("guidance_rescale", 0.0)),
                int(item.seed),
                int(p.get("num_images", 1)),
                bool(p.get("enable_refiner", False)),
                refiner_rel,
                float(p.get("refiner_strength", 0.2)),
                int(p.get("refiner_steps", 20)),
                str(p.get("device", "auto")),
                str(p.get("dtype", settings.default_dtype)),
            )

        def _refresh_models():
            new_base = _choices("base")
            new_vae = ["(none)"] + _choices("vae")
            new_lora = _choices("lora")
            new_refiner = _choices("refiner")

            return (
                gr.update(choices=new_base, value=(new_base[0] if new_base else None)),
                gr.update(choices=new_vae, value="(none)"),
                gr.update(choices=new_lora, value=[]),
                gr.update(choices=new_refiner, value=(new_refiner[0] if new_refiner else None)),
            )

        refresh_models.click(
            _refresh_models,
            inputs=[],
            outputs=[base_model, vae, loras, refiner_model],
        )

        run_btn.click(
            _run,
            inputs=[
                base_model,
                vae,
                loras,
                lora_scale,
                prompt,
                negative,
                width,
                height,
                steps,
                cfg,
                scheduler,
                prediction_type,
                rescale_betas_zero_snr,
                guidance_rescale,
                seed,
                num_images,
                enable_refiner,
                refiner_model,
                refiner_strength,
                refiner_steps,
                device,
                dtype,
            ],
            outputs=[gallery, seed_out, history_id_out, paths_out, status],
        )

        # History wiring
        def _refresh_history(limit: int):
            choices = _history_choices(limit)
            return gr.update(choices=choices, value=(choices[0] if choices else None))

        history_refresh.click(_refresh_history, inputs=[history_limit], outputs=[history_pick])
        history_limit.change(_refresh_history, inputs=[history_limit], outputs=[history_pick])
        history_pick.change(_history_preview, inputs=[history_pick], outputs=[history_preview])
        load_btn.click(
            _load_from_history,
            inputs=[history_pick],
            outputs=[
                prompt,
                negative,
                base_model,
                vae,
                loras,
                lora_scale,
                width,
                height,
                steps,
                cfg,
                scheduler,
                prediction_type,
                rescale_betas_zero_snr,
                guidance_rescale,
                seed,
                num_images,
                enable_refiner,
                refiner_model,
                refiner_strength,
                refiner_steps,
                device,
                dtype,
            ],
        )

        # Populate history at startup.
        demo.load(_refresh_history, inputs=[history_limit], outputs=[history_pick])

    return demo
