from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env", override=True)


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
    llm_n_ctx: int = 32768
    llm_context_fallbacks: str = "32768,24576,16384,8192,4096"
    llm_max_input_tokens: int = 28672
    llm_context_token_budget: int = 24000
    llm_max_output_tokens: int = 512
    llm_temperature: float = 0.1
    llm_n_gpu_layers: int = 0
    llm_n_threads: int = 0
    llm_n_batch: int = 512
    debug_llm_config: bool = True
    debug_input_classification: bool = True

    reranker_max_length: int = 512
    enable_reranker: bool = True
    debug_reranker: bool = True

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "askme_documents"

    retriever_top_k: int = 20
    chunk_size: int = 600
    chunk_overlap: int = 100
    reranker_top_k: int = 5

    data_dir: Path = Field(default=BASE_DIR / "data")


@lru_cache
def get_settings() -> Settings:
    return Settings()
