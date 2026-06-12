import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env", override=True)


class Settings(BaseModel):
    app_env: str = "development"

    # LangSmith/Hugging Face values may live in .env because they are either
    # secrets or tied to the local tracing workspace.
    langsmith_api_key: str = Field(default_factory=lambda: os.getenv("LANGSMITH_API_KEY", ""))
    langsmith_tracing: bool = Field(
        default_factory=lambda: _env_bool("LANGSMITH_TRACING", default=False)
    )
    langsmith_project: str = Field(
        default_factory=lambda: os.getenv("LANGSMITH_PROJECT", "askme-rag").strip('"')
    )
    langsmith_endpoint: str = Field(
        default_factory=lambda: os.getenv(
            "LANGSMITH_ENDPOINT",
            "https://api.smith.langchain.com",
        )
    )
    hf_token: str = Field(default_factory=lambda: os.getenv("HF_TOKEN", ""))
    gemini_api_key: str = Field(
        default_factory=lambda: (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GOOGLE_GENAI_API_KEY")
            or os.getenv("GOOGLE_GEMINI_API_KEY")
            or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
            or ""
        )
    )

    # LangSmith
    langsmith_eval_dataset: str = "askme-rag-eval"

    # Hugging Face embedding/reranking
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hf_reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    hf_device: str = "cpu"

    # Gemini text generation
    llm_backend: Literal["gemini"] = "gemini"
    gemini_model: str = "gemini-2.5-flash"
    gemini_max_input_tokens: int = 28672
    gemini_context_token_budget: int = 24000
    gemini_max_output_tokens: int = 1024
    gemini_temperature: float = 0.1

    debug_llm_config: bool = True
    debug_input_classification: bool = True

    # Reranker
    reranker_max_length: int = 512
    enable_reranker: bool = True
    debug_reranker: bool = True

    # Qdrant local
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "askme_documents"

    # Retrieval
    retriever_top_k: int = 20
    chunk_size: int = 600
    chunk_overlap: int = 100
    reranker_top_k: int = 5

    data_dir: Path = Field(default=BASE_DIR / "data")


@lru_cache
def get_settings() -> Settings:
    return Settings()


def configure_langsmith_environment() -> None:
    settings = get_settings()
    tracing = "true" if settings.langsmith_tracing else "false"

    os.environ["LANGSMITH_TRACING"] = tracing
    os.environ["LANGCHAIN_TRACING_V2"] = tracing
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint

    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key

    if settings.langsmith_tracing:
        print(
            "[debug] langsmith_tracing:",
            {
                "enabled": True,
                "endpoint": settings.langsmith_endpoint,
                "project": settings.langsmith_project,
            },
        )


def _env_bool(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}
