import json
from pathlib import Path
from typing import Any

from askme.config import BASE_DIR, configure_langsmith_environment, get_settings
from askme.graph import build_qa_graph
from langsmith import Client, traceable


EVAL_FILE = BASE_DIR / "data" / "evals" / "qa_examples.json"


def main() -> None:
    configure_langsmith_environment()
    settings = get_settings()
    examples = _load_eval_examples(EVAL_FILE)
    if not examples:
        print(f"No eval examples found in {EVAL_FILE}. Add evaluation questions first.")
        return

    client = Client(
        api_key=settings.langsmith_api_key or None,
        api_url=settings.langsmith_endpoint,
    )
    dataset_name = settings.langsmith_eval_dataset
    dataset = _get_or_create_dataset(client, dataset_name, examples)

    results = client.evaluate(
        answer_question,
        data=dataset.name,
        evaluators=[answer_contains_required_terms],
        experiment_prefix="askme-rag",
        description="RAG evaluation for AskMe over local Qdrant data.",
        max_concurrency=1,
        metadata={
            "models": [settings.gemini_model],
            "embedding_model": settings.hf_embedding_model,
            "vectorstore": "qdrant-local",
        },
    )
    print("Evaluation submitted to LangSmith.")
    print(f"Dataset: {dataset.name}")
    print(f"Experiment: {results}")


@traceable(name="askme-rag-answer")
def answer_question(inputs: dict[str, Any]) -> dict[str, str]:
    graph = _get_graph()
    result = graph.invoke({"question": inputs["question"]})
    structured_answer = result.get("answer", {})
    return {"answer": structured_answer.get("answer", "")}


def answer_contains_required_terms(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> bool:
    del inputs
    answer = str(outputs.get("answer", "")).lower()
    required_terms = reference_outputs.get("must_contain", [])
    return all(str(term).lower() in answer for term in required_terms)


def _get_graph():
    if not hasattr(_get_graph, "_graph"):
        _get_graph._graph = build_qa_graph()
    return _get_graph._graph


def _load_eval_examples(file_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    examples: list[dict[str, Any]] = []

    for item in payload:
        examples.append(
            {
                "inputs": {"question": item["question"]},
                "outputs": {
                    "answer": item.get("answer", ""),
                    "must_contain": item.get("must_contain", []),
                },
            }
        )
    return examples


def _get_or_create_dataset(
    client: Client,
    dataset_name: str,
    examples: list[dict[str, Any]],
):
    try:
        return client.read_dataset(dataset_name=dataset_name)
    except Exception:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="AskMe RAG evaluation dataset.",
        )
        for example in examples:
            client.create_example(
                dataset_id=dataset.id,
                inputs=example["inputs"],
                outputs=example["outputs"],
            )
        return dataset


if __name__ == "__main__":
    main()
