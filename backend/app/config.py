"""
ThreatShield AI | 2026
config.py
"""

from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    app_name: str = "ThreatShield AI"
    env: str = "development"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str = ""
    log_level: str = "INFO"

    database_url: str = f"sqlite+aiosqlite:///{(BACKEND_DIR / 'angelusvigil.db').as_posix()}"

    redis_url: str = "redis://localhost:6379"

    geoip_db_path: str = "/usr/share/GeoIP/GeoLite2-City.mmdb"

    nginx_log_path: str = "data/sample-logs/access.log"

    raw_queue_size: int = 1000
    parsed_queue_size: int = 500
    feature_queue_size: int = 200
    alert_queue_size: int = 100

    batch_size: int = 32
    batch_timeout_ms: int = 50

    model_dir: str = "data/models"
    detection_mode: str = "deep_learning"
    ensemble_weight_ae: float = 0.40
    ensemble_weight_dnn: float = 0.55
    ae_threshold_percentile: float = 99.5
    mlflow_tracking_uri: str = "file:./mlruns"

    @model_validator(mode="after")
    def resolve_sqlite_paths(self) -> "Settings":
        """
        Keep SQLite files under backend/ even when uvicorn is started
        from the project root or another working directory.
        """
        prefix = "sqlite+aiosqlite:///"
        if not self.database_url.startswith(prefix):
            return self

        db_path = self.database_url.removeprefix(prefix)
        if db_path.startswith(":memory:"):
            return self

        path = Path(db_path)
        if not path.is_absolute():
            path = BACKEND_DIR / path
            self.database_url = f"{prefix}{path.as_posix()}"
        return self


settings = Settings()
