from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI

from askme.config import get_settings


@lru_cache
def build_reasoning_llm() -> ChatGoogleGenerativeAI:
    settings = get_settings()
    if settings.debug_llm_config:
        print(
            "[debug] gemini_config:",
            {
                "model": settings.gemini_model,
                "temperature": settings.gemini_temperature,
                "max_output_tokens": settings.gemini_max_output_tokens,
                "max_input_tokens": settings.gemini_max_input_tokens,
                "context_token_budget": settings.gemini_context_token_budget,
            },
        )

    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key or None,
        temperature=settings.gemini_temperature,
        max_output_tokens=settings.gemini_max_output_tokens,
    )


def invoke_reasoning_text(prompt: str) -> str:
    response = build_reasoning_llm().invoke(prompt)
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "\n".join(_content_part_to_text(part) for part in content)
    return str(content)


def count_reasoning_tokens(text: str) -> int:
    llm = build_reasoning_llm()
    try:
        return llm.get_num_tokens(text)
    except Exception:
        return max(1, len(text) // 4)


def trim_text_for_reasoning(text: str, max_tokens: int) -> str:
    token_count = count_reasoning_tokens(text)
    if token_count <= max_tokens:
        return text

    max_chars = max_tokens * 4
    return text[:max_chars]


def _content_part_to_text(part) -> str:
    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        return str(part.get("text", ""))
    return str(part)
