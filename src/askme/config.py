from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env", override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"

    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "askme-rag"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_eval_dataset: str = "askme-rag-eval"

    hf_token: str = ""
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hf_reasoning_model: str = "google/flan-t5-base"
    hf_device: str = "cpu"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "askme_documents"

    retriever_top_k: int = 5
    chunk_size: int = 900
    chunk_overlap: int = 150

    data_dir: Path = Field(default=BASE_DIR / "data")


@lru_cache
def get_settings() -> Settings:
    return Settings()
