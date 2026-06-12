from rich.console import Console
from rich.table import Table

from askme.config import configure_langsmith_environment
from askme.graph import build_qa_graph


console = Console()


def main() -> None:
    configure_langsmith_environment()
    graph = build_qa_graph()
    console.print("[bold green]AskMe is ready. Type 'exit' to quit.[/bold green]")

    while True:
        question = console.input("[bold cyan]Question:[/bold cyan] ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            break

        result = graph.invoke(
            {"question": question},
            config={
                "run_name": "askme_chat",
                "tags": ["chat", "rag"],
                "metadata": {"entrypoint": "scripts/chat.py"},
            },
        )
        answer = result["answer"]
        console.print(f"[bold yellow]Answer:[/bold yellow] {answer['answer']}")
        console.print(
            f"[dim]Has enough context: {answer['has_enough_context']} | "
            f"Confidence: {answer['confidence']}[/dim]"
        )
        if answer.get("citations"):
            table = Table("Source", "Excerpt")
            for citation in answer["citations"]:
                table.add_row(str(citation.get("source", "")), citation.get("excerpt", ""))
            console.print(table)


if __name__ == "__main__":
    main()
