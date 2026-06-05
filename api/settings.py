"""API configuration (env vars + defaults)."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "CricGiri Analytics API"
    app_version: str = "1.0.0"
    model_version: str = "ball_best_v1"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    public_base_url: str = ""

    # Storage (production layout)
    upload_dir: Path = ROOT / "uploads"
    output_dir: Path = ROOT / "outputs" / "api"
    log_dir: Path = ROOT / "logs"
    max_upload_mb: int = 200
    retention_hours: int = 48

    # Models (relative to project root)
    ball_model_path: str = "models/ball_best.pt"
    stump_model_path: str = "models/stump_best.pt"
    device: str | None = None
    use_half_precision: bool = False
    inference_imgsz: int = 640

    # Pipeline defaults for API requests
    bowler_arm: str = "right"
    ball_confidence: float = 0.10
    save_annotated_video: bool = True
    blur_recovery: bool = False

    # Job queue (1 worker — pipeline singleton)
    max_concurrent_jobs: int = 1

    # Optional API key (set API_KEY env to enable)
    api_key: str | None = None


settings = ApiSettings()
