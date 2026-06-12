import argparse
from textwrap import shorten

from langchain_core.documents import Document
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from askme.config import get_settings
from askme.reranker import rerank_documents_with_scores
from askme.vectorstore import build_retriever


console = Console()


def main() -> None:
    args = _parse_args()
    settings = get_settings()
    question = args.question or console.input("[bold cyan]Question:[/bold cyan] ").strip()

    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"Retriever top_k: {settings.retriever_top_k}",
                    f"Reranker enabled: {settings.enable_reranker}",
                    f"Reranker top_k: {settings.reranker_top_k}",
                    f"Qdrant collection: {settings.qdrant_collection}",
                ]
            ),
            title="Retrieval Debug Config",
        )
    )

    retriever = build_retriever()
    retrieved_docs = retriever.invoke(question)
    console.print(f"[bold green]Retrieved {len(retrieved_docs)} candidate documents.[/bold green]")
    _print_documents_table(
        title="Retriever Output",
        documents=[(None, doc) for doc in retrieved_docs],
        max_chars=args.max_chars,
    )

    reranked_docs = rerank_documents_with_scores(
        documents=retrieved_docs,
        question=question,
        top_k=settings.reranker_top_k,
    )
    console.print(f"[bold green]Reranked to {len(reranked_docs)} documents.[/bold green]")
    _print_documents_table(
        title="Reranker Output",
        documents=reranked_docs,
        max_chars=args.max_chars,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run only retriever and reranker for debugging RAG document selection."
    )
    parser.add_argument("question", nargs="?", help="Question to retrieve/rerank against.")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=320,
        help="Maximum characters to show from each document chunk.",
    )
    return parser.parse_args()


def _print_documents_table(
    title: str,
    documents: list[tuple[float | None, Document]],
    max_chars: int,
) -> None:
    table = Table(title=title, show_lines=True)
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Score", justify="right", style="magenta", no_wrap=True)
    table.add_column("Source", style="green")
    table.add_column("Record", justify="right", no_wrap=True)
    table.add_column("Excerpt")

    for index, (score, doc) in enumerate(documents, start=1):
        source = str(doc.metadata.get("source", ""))
        record_index = str(doc.metadata.get("record_index", ""))
        excerpt = shorten(
            " ".join(doc.page_content.split()),
            width=max_chars,
            placeholder="...",
        )
        table.add_row(
            str(index),
            "" if score is None else f"{score:.4f}",
            source,
            record_index,
            excerpt,
        )

    console.print(table)


if __name__ == "__main__":
    main()
