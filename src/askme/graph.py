from typing import TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from askme.llm import build_reasoning_llm
from askme.prompts import QA_PROMPT
from askme.vectorstore import build_retriever


class QAState(TypedDict, total=False):
    question: str
    documents: list[Document]
    context: str
    answer: str


def build_qa_graph():
    retriever = build_retriever()
    llm = build_reasoning_llm()

    def retrieve(state: QAState) -> QAState:
        documents = retriever.invoke(state["question"])
        return {"documents": documents}

    def prepare_context(state: QAState) -> QAState:
        context = "\n\n".join(
            f"[{index + 1}] {doc.page_content}"
            for index, doc in enumerate(state.get("documents", []))
        )
        return {"context": context}

    def generate(state: QAState) -> QAState:
        prompt = QA_PROMPT.invoke(
            {
                "question": state["question"],
                "context": state.get("context", ""),
            }
        )
        answer = llm.invoke(prompt.to_string())
        return {"answer": str(answer)}

    graph = StateGraph(QAState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("prepare_context", prepare_context)
    graph.add_node("generate", generate)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "prepare_context")
    graph.add_edge("prepare_context", "generate")
    graph.add_edge("generate", END)
    return graph.compile()
