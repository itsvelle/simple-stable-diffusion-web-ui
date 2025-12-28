from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    base_model: str = Field(..., description="Path under models/ to the SDXL base model")
    vae: str | None = Field(default=None, description="Optional VAE path under models/")
    loras: list[str] = Field(default_factory=list, description="LoRA paths under models/")
    lora_scale: float = Field(default=0.8, ge=0.0, le=2.0)

    prompt: str
    negative_prompt: str = ""

    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)

    steps: int = Field(default=30, ge=1, le=150)
    guidance_scale: float = Field(default=5.0, ge=0.0, le=30.0)

    scheduler: str = Field(default="DPM++ 2M Karras")

    prediction_type: str = Field(
        default="auto",
        description="auto|epsilon|v_prediction. NoobAI-XL v-pred models require v_prediction.",
    )
    rescale_betas_zero_snr: bool = Field(
        default=True,
        description="Recommended for many v-pred models to reduce overexposure/oversaturation.",
    )
    guidance_rescale: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="CFG rescale. For v-pred models often ~0.7 helps; 0 disables.",
    )

    seed: int = Field(default=-1, description="-1 means random")
    num_images: int = Field(default=1, ge=1, le=8)

    enable_refiner: bool = False
    refiner_model: str | None = None
    refiner_strength: float = Field(default=0.2, ge=0.0, le=1.0)
    refiner_steps: int = Field(default=20, ge=1, le=100)

    dtype: str | None = Field(default=None, description="Override dtype: float16|bfloat16|float32")
    device: str | None = Field(default=None, description="Override device: auto|cuda|cpu")


class GenerateResponse(BaseModel):
    history_id: int | None = None
    seed: int
    image_paths: list[str]
