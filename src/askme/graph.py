from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import END, StateGraph

from askme.config import get_settings
from askme.llm import build_reasoning_llm, trim_text_for_reasoning
from askme.prompts import QA_PROMPT, STRUCTURED_OUTPUT_INSTRUCTIONS
from askme.reranker import rerank_documents
from askme.schemas import AnswerResponse
from askme.vectorstore import build_retriever


class QAState(TypedDict, total=False):
    question: str
    documents: list[Document]
    context: str
    answer: dict[str, Any]
    raw_answer: str


def build_qa_graph():
    retriever = build_retriever()
    llm = build_reasoning_llm()
    parser = PydanticOutputParser(pydantic_object=AnswerResponse)

    def retrieve(state: QAState) -> QAState:
        documents = retriever.invoke(state["question"])
        return {"documents": documents}

    def rerank(state: QAState) -> QAState:
        settings = get_settings()
        documents = state.get("documents", [])
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
        prompt = QA_PROMPT.invoke(
            {
                "question": state["question"],
                "context": state.get("context", ""),
                "format_instructions": STRUCTURED_OUTPUT_INSTRUCTIONS,
            }
        )
        prompt_text = trim_text_for_reasoning(prompt.to_string(), settings.llm_max_input_tokens)
        raw_answer = str(llm.invoke(prompt_text))
        answer = _parse_structured_answer(parser, raw_answer)
        return {"answer": answer.model_dump(), "raw_answer": raw_answer}

    graph = StateGraph(QAState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("prepare_context", prepare_context)
    graph.add_node("generate", generate)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "prepare_context")
    graph.add_edge("prepare_context", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


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
        return AnswerResponse(
            answer=raw_answer,
            has_enough_context=False,
            confidence=0.0,
            citations=[],
            missing_info=["Model did not return valid structured JSON."],
        )
