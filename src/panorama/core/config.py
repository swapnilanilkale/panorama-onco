from __future__ import annotations
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PANORAMA_",   # env vars are read as PANORAMA_DEVICE, etc.
        env_file=".env",          # ...or from a local .env file
        extra="ignore",
    )

    data_root: Path = Field(default=Path("./data"))
    output_dir: Path = Field(default=Path("./outputs"))
    device: str = Field(default="cuda")   # "cuda" | "cpu" | "mps"
    seed: int = Field(default=1337)

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_root / "processed"


@lru_cache
def get_settings() -> Settings:
    return Settings()