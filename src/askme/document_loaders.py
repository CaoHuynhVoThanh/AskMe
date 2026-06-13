import json
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from askme.config import get_settings


def load_source_documents() -> list[Document]:
    settings = get_settings()
    docs = _load_docx(settings.data_dir / "docx")
    docs.extend(_load_pdf(settings.data_dir / "pdf"))
    docs.extend(_load_json(settings.data_dir / "json"))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(docs)


def _load_docx(folder: Path) -> list[Document]:
    documents: list[Document] = []
    if not folder.exists():
        return documents
    for file_path in sorted(folder.glob("*.docx")):
        documents.extend(Docx2txtLoader(str(file_path)).load())
    return documents


def _load_pdf(folder: Path) -> list[Document]:
    documents: list[Document] = []
    if not folder.exists():
        return documents
    for file_path in sorted(folder.glob("*.[pP][dD][fF]")):
        documents.extend(PyPDFLoader(str(file_path)).load())
    return documents


def _load_json(folder: Path) -> list[Document]:
    documents: list[Document] = []
    for file_path in sorted(folder.glob("*.json")):
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        documents.extend(_json_to_documents(payload, file_path))
    return documents


def _json_to_documents(payload: Any, source: Path) -> list[Document]:
    if isinstance(payload, list):
        return [_json_item_to_document(item, source, index) for index, item in enumerate(payload)]
    return [_json_item_to_document(payload, source, 0)]


def _json_item_to_document(item: Any, source: Path, index: int) -> Document:
    if isinstance(item, dict):
        text = item.get("content") or item.get("text") or item.get("answer")
        if not text:
            text = json.dumps(item, ensure_ascii=False, indent=2)
        metadata = {key: value for key, value in item.items() if isinstance(value, (str, int, float, bool))}
    else:
        text = str(item)
        metadata = {}

    metadata.update({"source": str(source), "record_index": index})
    return Document(page_content=text, metadata=metadata)
