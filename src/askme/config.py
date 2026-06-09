from functools import lru_cache
from pathlib import Path
from typing import Literal

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
    hf_reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    hf_device: str = "cpu"

    llm_backend: Literal["llama_cpp"] = "llama_cpp"
    llm_model_path: Path = Path("E:/LLMs/Vi-Qwen2-7B-RAG.Q2_K.gguf")
    llm_n_ctx: int = 4096
    llm_max_input_tokens: int = 3072
    llm_context_token_budget: int = 1800
    llm_max_output_tokens: int = 512
    llm_temperature: float = 0.1
    llm_n_gpu_layers: int = 0
    llm_n_threads: int = 0

    reranker_max_length: int = 512

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "askme_documents"

    retriever_top_k: int = 5
    chunk_size: int = 600
    chunk_overlap: int = 100
    reranker_top_k: int = 1

    data_dir: Path = Field(default=BASE_DIR / "data")


@lru_cache
def get_settings() -> Settings:
    return Settings()
