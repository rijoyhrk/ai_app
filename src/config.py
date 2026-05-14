from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    anthropic_api_key: str = Field(..., env="ANTHROPIC_API_KEY")
    chroma_persist_dir: str = Field("./chroma_db", env="CHROMA_PERSIST_DIR")
    alias_registry_path: str = Field("./data/alias_registry.json", env="ALIAS_REGISTRY_PATH")
    etl_repo_path: str = Field("./data/sample_etl", env="ETL_REPO_PATH")
    embedding_model: str = Field("claude-sonnet-4-6", env="EMBEDDING_MODEL")
    llm_model: str = Field("claude-sonnet-4-6", env="LLM_MODEL")
    collection_name: str = "etl_lineage"
    chunk_collection_name: str = "etl_chunks"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
