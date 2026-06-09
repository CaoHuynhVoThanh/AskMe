from functools import lru_cache

from langchain_community.llms import LlamaCpp

from askme.config import get_settings


@lru_cache
def build_reasoning_llm() -> LlamaCpp:
    settings = get_settings()
    kwargs = {
        "model_path": str(settings.llm_model_path),
        "n_ctx": settings.llm_n_ctx,
        "max_tokens": settings.llm_max_output_tokens,
        "temperature": settings.llm_temperature,
        "n_gpu_layers": settings.llm_n_gpu_layers,
        "verbose": False,
    }
    if settings.llm_n_threads > 0:
        kwargs["n_threads"] = settings.llm_n_threads

    return LlamaCpp(**kwargs)


def count_reasoning_tokens(text: str) -> int:
    llm = build_reasoning_llm()
    try:
        return llm.get_num_tokens(text)
    except Exception:
        return max(1, len(text) // 4)


def trim_text_for_reasoning(text: str, max_tokens: int) -> str:
    llm = build_reasoning_llm()
    try:
        token_ids = llm.client.tokenize(text.encode("utf-8"), add_bos=False)
        if len(token_ids) <= max_tokens:
            return text
        return llm.client.detokenize(token_ids[:max_tokens]).decode(
            "utf-8",
            errors="ignore",
        )
    except Exception:
        max_chars = max_tokens * 4
        return text if len(text) <= max_chars else text[:max_chars]
