from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .schemas import GenerateRequest, GenerateResponse
from .settings import settings
from .ui import build_ui


def create_app() -> FastAPI:
    app = FastAPI(title="SDXL UI", version="0.1.0")

    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/outputs", StaticFiles(directory=str(settings.outputs_dir)), name="outputs")

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/history")
    def api_history(limit: int = 50) -> list[dict]:
        from .history import format_choice, list_generations

        items = list_generations(limit=limit)
        return [
            {
                "id": i.id,
                "created_at": i.created_at,
                "seed": i.seed,
                "base_model": i.base_model,
                "prompt": i.prompt,
                "negative_prompt": i.negative_prompt,
                "params": i.params,
                "image_files": i.image_files,
                "label": format_choice(i),
            }
            for i in items
        ]

    @app.get("/api/history/{gen_id}")
    def api_history_item(gen_id: int) -> dict:
        from fastapi import HTTPException

        from .history import format_choice, get_generation

        item = get_generation(gen_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Not found")
        return {
            "id": item.id,
            "created_at": item.created_at,
            "seed": item.seed,
            "base_model": item.base_model,
            "prompt": item.prompt,
            "negative_prompt": item.negative_prompt,
            "params": item.params,
            "image_files": item.image_files,
            "label": format_choice(item),
        }

    @app.post("/api/generate", response_model=GenerateResponse)
    def api_generate(req: GenerateRequest) -> GenerateResponse:
        from .pipeline import generate

        seed, _images, paths, history_id = generate(
            base_model=req.base_model,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=req.width,
            height=req.height,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            scheduler=req.scheduler,
            prediction_type=req.prediction_type,
            rescale_betas_zero_snr=req.rescale_betas_zero_snr,
            guidance_rescale=req.guidance_rescale,
            seed=req.seed,
            num_images=req.num_images,
            enable_refiner=req.enable_refiner,
            refiner_model=req.refiner_model,
            refiner_strength=req.refiner_strength,
            refiner_steps=req.refiner_steps,
            vae=req.vae,
            loras=req.loras,
            lora_scale=req.lora_scale,
            device=req.device or settings.default_device,
            dtype=req.dtype or settings.default_dtype,
            return_images=False,
        )
        return GenerateResponse(
            history_id=history_id,
            seed=seed,
            image_paths=[f"/outputs/{p}" for p in paths],
        )

    import gradio as gr

    demo = build_ui()
    app = gr.mount_gradio_app(app, demo, path="/")

    return app


app = create_app()
