from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://arkshield:arkshield@localhost:5432/arkshield"
    alembic_database_url: str = "postgresql://arkshield:arkshield@localhost:5432/arkshield"

    jwt_secret: str = "change-me"
    jwt_algorithm: str = "HS256"
    jwt_expires_hours: int = 12

    port: int = 4000
    web_origin: str = "http://localhost:3000"

    auto_start_simulation: bool = True
    simulation_tick_seconds: float = 2.0

    # ARK Predict — streaming anomaly detection (see ANOMALY.md)
    anomaly_auto_start: bool = True
    anomaly_public_demo: bool = True   # open /api/anomaly/* without a JWT (demo UI has no real token)
    anomaly_speed: int = 2             # simulated seconds per 0.5 s tick
    anomaly_fault_rate_per_min: float = 0.8
    anomaly_seed: int = 7


settings = Settings()
