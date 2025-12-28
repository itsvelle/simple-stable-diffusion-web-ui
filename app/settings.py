from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SDUI_", env_file=".env", extra="ignore")

    models_dir: Path = Field(default=Path("models"), description="Directory containing model files")
    outputs_dir: Path = Field(default=Path("outputs"), description="Where generated images are saved")
    history_db: Path = Field(
        default=Path("outputs") / "history.sqlite3",
        description="SQLite DB path for generation history",
    )

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=7860)

    default_device: str = Field(
        default="auto",
        description="auto|cuda|cpu. For AMD ROCm, 'cuda' is still correct.",
    )
    default_dtype: str = Field(default="float16", description="float16|bfloat16|float32")


settings = Settings()
