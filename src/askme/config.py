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

    # Secrets are intentionally the only values read from .env.
    langsmith_api_key: str = Field(default_factory=lambda: os.getenv("LANGSMITH_API_KEY", ""))
    hf_token: str = Field(default_factory=lambda: os.getenv("HF_TOKEN", ""))

    # LangSmith
    langsmith_tracing: bool = False
    langsmith_project: str = "askme-rag"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_eval_dataset: str = "askme-rag-eval"

    # Hugging Face embedding/reranking
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hf_reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    hf_device: str = "cpu"

    # Local GGUF text generation. Low-memory profile is active by default.
    llm_backend: Literal["llama_cpp"] = "llama_cpp"
    llm_model_path: Path = Path("E:/LLMs/Vi-Qwen2-7B-RAG.Q2_K.gguf")

    # High-context profile, kept here for quick switching when RAM allows:
    # llm_n_ctx: int = 32768
    # llm_context_fallbacks: str = "32768,24576,16384,8192,4096"
    # llm_max_input_tokens: int = 28672
    # llm_context_token_budget: int = 24000
    # llm_n_batch: int = 512

    llm_n_ctx: int = 4096
    llm_context_fallbacks: str = "4096"
    llm_max_input_tokens: int = 3072
    llm_context_token_budget: int = 2200
    llm_max_output_tokens: int = 512
    llm_temperature: float = 0.1
    llm_n_gpu_layers: int = 0
    llm_n_threads: int = 0
    llm_n_batch: int = 128

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
