"""Environment-backed application settings with safe local defaults."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    database_path: Path = PROJECT_ROOT / "data" / "predictions.sqlite3"
    max_upload_bytes: int = 10 * 1024 * 1024
    max_image_side: int = 6000
    max_image_pixels: int = 30_000_000
    allowed_origins: tuple[str, ...] = ("http://localhost:5173",)

    @classmethod
    def from_env(cls):
        origins = os.getenv("CORS_ORIGINS", "http://localhost:5173")
        artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
        database_path = Path(
            os.getenv("DATABASE_PATH", PROJECT_ROOT / "data" / "predictions.sqlite3")
        )
        if not artifacts_dir.is_absolute():
            artifacts_dir = PROJECT_ROOT / artifacts_dir
        if not database_path.is_absolute():
            database_path = PROJECT_ROOT / database_path
        return cls(
            artifacts_dir=artifacts_dir,
            database_path=database_path,
            max_upload_bytes=int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024,
            max_image_side=int(os.getenv("MAX_IMAGE_SIDE", "6000")),
            max_image_pixels=int(os.getenv("MAX_IMAGE_PIXELS", "30000000")),
            allowed_origins=tuple(
                origin.strip() for origin in origins.split(",") if origin.strip()
            ),
        )
