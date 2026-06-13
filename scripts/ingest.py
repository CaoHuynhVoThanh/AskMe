import argparse

from askme.document_loaders import load_source_documents
from askme.vectorstore import build_vectorstore, reset_vectorstore_collection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing Qdrant collection before ingesting documents.",
    )
    args = parser.parse_args()

    if args.reset:
        reset_vectorstore_collection()

    documents = load_source_documents()
    if not documents:
        print("No documents found in data/docx, data/pdf, or data/json.")
        return

    vectorstore = build_vectorstore()
    vectorstore.add_documents(documents)
    print(f"Indexed {len(documents)} chunks into Qdrant.")


if __name__ == "__main__":
    main()
