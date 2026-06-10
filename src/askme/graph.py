import json
import unicodedata
from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import END, StateGraph

from askme.config import get_settings
from askme.llm import build_reasoning_llm, trim_text_for_reasoning
from askme.prompts import LLM_STOP_TOKENS, build_input_classification_prompt, build_qa_prompt
from askme.reranker import rerank_documents
from askme.schemas import AnswerResponse, InputClassification
from askme.vectorstore import build_retriever


class QAState(TypedDict, total=False):
    question: str
    documents: list[Document]
    context: str
    answer: dict[str, Any]
    input_classification: dict[str, Any]
    raw_input_classification: str
    raw_answer: str


def build_qa_graph():
    retriever = build_retriever()
    llm = build_reasoning_llm()
    parser = PydanticOutputParser(pydantic_object=AnswerResponse)

    def classify_input(state: QAState) -> QAState:
        settings = get_settings()
        prompt = build_input_classification_prompt(state["question"])
        prompt_text = trim_text_for_reasoning(prompt, settings.llm_max_input_tokens)
        raw_classification = str(llm.invoke(prompt_text, stop=LLM_STOP_TOKENS))
        classification = _parse_input_classification(
            raw_classification=raw_classification,
            question=state["question"],
        )
        payload = classification.model_dump()
        if settings.debug_input_classification:
            print("[debug] input_classification:", json.dumps(payload, ensure_ascii=False))
        return {
            "input_classification": payload,
            "raw_input_classification": raw_classification,
        }

    def retrieve(state: QAState) -> QAState:
        documents = retriever.invoke(state["question"])
        return {"documents": documents}

    def rerank(state: QAState) -> QAState:
        settings = get_settings()
        documents = state.get("documents", [])
        if not documents:
            return {"documents": []}
        reranked = rerank_documents(documents, state["question"], settings.reranker_top_k)
        return {"documents": reranked}

    def prepare_context(state: QAState) -> QAState:
        settings = get_settings()
        context = "\n\n".join(
            _format_document(index, doc)
            for index, doc in enumerate(state.get("documents", []))
        )
        context = trim_text_for_reasoning(context, settings.llm_context_token_budget)
        return {"context": context}

    def generate(state: QAState) -> QAState:
        settings = get_settings()
        if _is_simple_greeting(state["question"]):
            answer = AnswerResponse(
                answer="Xin chao! Minh co the ho tro ban tra cuu va giai dap thong tin trong tai lieu.",
                has_enough_context=True,
                confidence=1.0,
                citations=[],
                missing_info=[],
            )
            return {"answer": answer.model_dump(), "raw_answer": answer.model_dump_json()}

        prompt = build_qa_prompt(
            question=state["question"],
            context=state.get("context", ""),
        )
        prompt_text = trim_text_for_reasoning(prompt, settings.llm_max_input_tokens)
        raw_answer = str(llm.invoke(prompt_text, stop=LLM_STOP_TOKENS))
        answer = _parse_structured_answer(parser, raw_answer)
        return {"answer": answer.model_dump(), "raw_answer": raw_answer}

    graph = StateGraph(QAState)
    graph.add_node("classify_input", classify_input)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("prepare_context", prepare_context)
    graph.add_node("generate", generate)
    graph.set_entry_point("classify_input")
    graph.add_conditional_edges(
        "classify_input",
        _route_after_input_classification,
        {
            "rag": "retrieve",
            "normal": "prepare_context",
        },
    )
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "prepare_context")
    graph.add_edge("prepare_context", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def _route_after_input_classification(state: QAState) -> str:
    classification = state.get("input_classification", {})
    route = classification.get("route", "rag")
    return "normal" if route == "normal" else "rag"


def _format_document(index: int, doc: Document) -> str:
    source = doc.metadata.get("source", "unknown")
    return f"[{index + 1}] source={source}\n{doc.page_content}"


def _parse_structured_answer(
    parser: PydanticOutputParser,
    raw_answer: str,
) -> AnswerResponse:
    try:
        return parser.parse(raw_answer)
    except Exception:
        json_text = _extract_first_json_object(raw_answer)
        if json_text:
            try:
                return AnswerResponse.model_validate(json.loads(json_text))
            except Exception:
                pass

    return AnswerResponse(
        answer=_clean_prompt_echo(raw_answer),
        has_enough_context=False,
        confidence=0.0,
        citations=[],
        missing_info=["Model did not return valid structured JSON."],
    )


def _parse_input_classification(
    raw_classification: str,
    question: str,
) -> InputClassification:
    json_text = _extract_first_json_object(raw_classification)
    if json_text:
        try:
            payload = json.loads(json_text)
            payload["route"] = _normalize_classification_route(payload.get("route", "rag"))
            return InputClassification.model_validate(payload)
        except Exception:
            pass

    route = "normal" if _is_simple_greeting(question) else "rag"
    reason = "Fallback heuristic after classifier returned invalid JSON."
    return InputClassification(route=route, reason=reason, confidence=0.0)


def _normalize_classification_route(route: Any) -> str:
    normalized = str(route).strip().lower()
    if "rag" in normalized or "retriev" in normalized or "document" in normalized:
        return "rag"
    if "normal" in normalized or "chat" in normalized or "general" in normalized:
        return "normal"
    return "rag"


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    decoder = json.JSONDecoder()
    try:
        _, end = decoder.raw_decode(text[start:])
        return text[start : start + end]
    except json.JSONDecodeError:
        return None


def _clean_prompt_echo(text: str) -> str:
    markers = ["<|im_start|>assistant", "Assistant:", "Answer:"]
    cleaned = text.strip()
    for marker in markers:
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[-1].strip()
    return cleaned


def _is_simple_greeting(question: str) -> bool:
    normalized = _strip_accents(question).strip().lower()
    return normalized in {
        "hi",
        "hello",
        "hey",
        "xin chao",
        "chao",
        "alo",
    }


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
