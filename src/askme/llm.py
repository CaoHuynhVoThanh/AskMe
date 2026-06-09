from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from askme.config import get_settings


def build_reasoning_llm() -> HuggingFacePipeline:
    settings = get_settings()
    tokenizer = AutoTokenizer.from_pretrained(
        settings.hf_reasoning_model,
        token=settings.hf_token or None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        settings.hf_reasoning_model,
        token=settings.hf_token or None,
    )
    text2text = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        device=-1 if settings.hf_device == "cpu" else 0,
    )
    return HuggingFacePipeline(pipeline=text2text)
