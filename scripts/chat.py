from rich.console import Console

from askme.graph import build_qa_graph


console = Console()


def main() -> None:
    graph = build_qa_graph()
    console.print("[bold green]AskMe da san sang. Go 'exit' de thoat.[/bold green]")

    while True:
        question = console.input("[bold cyan]Ban hoi:[/bold cyan] ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            break

        result = graph.invoke({"question": question})
        console.print(f"[bold yellow]Tra loi:[/bold yellow] {result['answer']}")


if __name__ == "__main__":
    main()
