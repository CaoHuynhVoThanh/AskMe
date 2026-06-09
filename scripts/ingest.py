from askme.document_loaders import load_source_documents
from askme.vectorstore import build_vectorstore


def main() -> None:
    documents = load_source_documents()
    if not documents:
        print("Khong tim thay tai lieu nao trong data/docx hoac data/json.")
        return

    vectorstore = build_vectorstore()
    vectorstore.add_documents(documents)
    print(f"Da nap {len(documents)} chunks vao Qdrant.")


if __name__ == "__main__":
    main()
