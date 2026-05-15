"""
Application configuration.

API key resolution order (first match wins):
  1. GCP Secret Manager  — when GCP_PROJECT_ID + GCP_SECRET_NAME are set
  2. ANTHROPIC_API_KEY   — env var / .env file (local dev fallback)

All other settings are read from environment / .env as usual.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings
from pydantic import Field, model_validator


class Settings(BaseSettings):
    # ── GCP Secret Manager (production) ──────────────────────────────────────
    gcp_project_id: str = Field("", env="GCP_PROJECT_ID")
    gcp_secret_name: str = Field("anthropic-api-key", env="GCP_SECRET_NAME")
    gcp_secret_version: str = Field("latest", env="GCP_SECRET_VERSION")

    # ── API key (resolved below; set directly only for local dev) ─────────────
    anthropic_api_key: str = Field("", env="ANTHROPIC_API_KEY")

    # ── App settings ─────────────────────────────────────────────────────────
    chroma_persist_dir: str = Field("./chroma_db", env="CHROMA_PERSIST_DIR")
    alias_registry_path: str = Field("./data/alias_registry.json", env="ALIAS_REGISTRY_PATH")
    etl_repo_path: str = Field("./data/sample_etl", env="ETL_REPO_PATH")
    embedding_model: str = Field("all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    llm_model: str = Field("claude-opus-4-7", env="LLM_MODEL")
    collection_name: str = "etl_lineage"

    @model_validator(mode="after")
    def resolve_api_key(self) -> "Settings":
        """
        If GCP_PROJECT_ID is configured, fetch the API key from Secret Manager.
        Falls back to ANTHROPIC_API_KEY env var for local development.
        Raises ValueError at startup if no key can be resolved.
        """
        if self.gcp_project_id:
            from src.secrets.gcp_secret_manager import fetch_secret
            self.anthropic_api_key = fetch_secret(
                project_id=self.gcp_project_id,
                secret_name=self.gcp_secret_name,
                version=self.gcp_secret_version,
            )

        if not self.anthropic_api_key:
            raise ValueError(
                "No Anthropic API key found. Either:\n"
                "  Production: set GCP_PROJECT_ID and GCP_SECRET_NAME in .env\n"
                "  Local dev:  set ANTHROPIC_API_KEY in .env"
            )
        return self

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
