from functools import lru_cache

from langchain_community.llms import LlamaCpp

from askme.config import get_settings


_EFFECTIVE_N_CTX: int | None = None


@lru_cache
def build_reasoning_llm() -> LlamaCpp:
    global _EFFECTIVE_N_CTX

    settings = get_settings()
    last_error: Exception | None = None

    for n_ctx in _context_candidates():
        kwargs = _llama_kwargs(n_ctx)
        if settings.debug_llm_config:
            print("[debug] llama_cpp_config_attempt:", _debug_config(kwargs))

        try:
            llm = LlamaCpp(**kwargs)
            _EFFECTIVE_N_CTX = n_ctx
            if settings.debug_llm_config:
                print("[debug] llama_cpp_config_loaded:", _debug_config(kwargs))
            return llm
        except Exception as exc:
            last_error = exc
            if settings.debug_llm_config:
                print(f"[debug] llama_cpp_config_failed: n_ctx={n_ctx}; error={exc}")

    raise RuntimeError(
        "Could not load the GGUF model with any configured context size. "
        "Lower llm_context_fallbacks/llm_n_ctx in src/askme/config.py or use a llama-cpp-python build with more memory support."
    ) from last_error


def count_reasoning_tokens(text: str) -> int:
    llm = build_reasoning_llm()
    try:
        return llm.get_num_tokens(text)
    except Exception:
        return max(1, len(text) // 4)


def trim_text_for_reasoning(text: str, max_tokens: int) -> str:
    llm = build_reasoning_llm()
    max_tokens = min(max_tokens, _effective_input_budget())
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


def _llama_kwargs(n_ctx: int) -> dict:
    settings = get_settings()
    kwargs = {
        "model_path": str(settings.llm_model_path),
        "n_ctx": n_ctx,
        "max_tokens": settings.llm_max_output_tokens,
        "temperature": settings.llm_temperature,
        "n_gpu_layers": settings.llm_n_gpu_layers,
        "n_batch": min(settings.llm_n_batch, n_ctx),
        "verbose": False,
    }
    if settings.llm_n_threads > 0:
        kwargs["n_threads"] = settings.llm_n_threads
    return kwargs


def _context_candidates() -> list[int]:
    settings = get_settings()
    values = [settings.llm_n_ctx]
    values.extend(_parse_context_fallbacks(settings.llm_context_fallbacks))

    candidates: list[int] = []
    for value in values:
        if value > 0 and value not in candidates:
            candidates.append(value)
    return candidates


def _parse_context_fallbacks(raw_value: str) -> list[int]:
    values: list[int] = []
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(int(item))
        except ValueError:
            continue
    return values


def _effective_input_budget() -> int:
    settings = get_settings()
    n_ctx = _EFFECTIVE_N_CTX or settings.llm_n_ctx
    reserved_output = settings.llm_max_output_tokens + 128
    return max(512, min(settings.llm_max_input_tokens, n_ctx - reserved_output))


def _debug_config(kwargs: dict) -> dict:
    settings = get_settings()
    n_ctx = kwargs["n_ctx"]
    reserved_output = settings.llm_max_output_tokens + 128
    effective_input_budget = max(512, min(settings.llm_max_input_tokens, n_ctx - reserved_output))
    return {
        "model_path": kwargs["model_path"],
        "n_ctx": n_ctx,
        "n_batch": kwargs["n_batch"],
        "effective_input_budget": effective_input_budget,
        "configured_max_input_tokens": settings.llm_max_input_tokens,
        "configured_context_token_budget": settings.llm_context_token_budget,
        "max_output_tokens": settings.llm_max_output_tokens,
        "n_gpu_layers": kwargs["n_gpu_layers"],
        "n_threads": kwargs.get("n_threads", 0),
    }
