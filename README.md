# AskMe RAG

Du an hoi dap bat ky thac mac nao cua nguoi dung dua tren du lieu trong `data/`.
Nguon du lieu ban dau ho tro `.docx` va `.json`, sau do index vao Qdrant local de truy van bang LangChain, dieu phoi reasoning bang LangGraph, va co san LangSmith tracing.
Ket qua tra loi duoc parse bang Pydantic schema de co cau truc gom `answer`, `has_enough_context`, `confidence`, `citations`, va `missing_info`.
Model sinh cau tra loi mac dinh la GGUF local qua `llama-cpp-python`: `E:\LLMs\Vi-Qwen2-7B-RAG.Q2_K.gguf`.

## Cau truc

```text
.
+-- data/
|   +-- docx/              # Dat file .docx tai day
|   +-- json/              # Dat file .json tai day
|   +-- evals/             # Bo cau hoi va dap an mau de LangSmith eval
+-- scripts/
|   +-- ingest.py          # Nap du lieu vao Qdrant
|   +-- chat.py            # Chat CLI voi tri thuc da index
|   +-- evaluate.py        # Chay eval va xem experiment tren LangSmith UI
+-- src/
|   +-- askme/
|       +-- config.py
|       +-- document_loaders.py
|       +-- embeddings.py
|       +-- graph.py
|       +-- llm.py
|       +-- vectorstore.py
|       +-- prompts.py
+-- docker-compose.yml     # Qdrant local
+-- .env
+-- requirements.txt
+-- pyproject.toml
```

## Cai dat

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Chay Qdrant local

```powershell
docker compose up -d qdrant
```

Neu khong dung Docker, co the chay Qdrant rieng va cap nhat `QDRANT_URL` trong `.env`.

## Cau hinh GGUF

Text generation duoc cau hinh trong `.env`:

```env
LLM_BACKEND=llama_cpp
LLM_MODEL_PATH=E:\LLMs\Vi-Qwen2-7B-RAG.Q2_K.gguf
LLM_N_CTX=4096
LLM_MAX_INPUT_TOKENS=3072
LLM_CONTEXT_TOKEN_BUDGET=1800
LLM_MAX_OUTPUT_TOKENS=512
```

Neu `pip install -r requirements.txt` gap loi build `llama-cpp-python` tren Windows, cai ban wheel phu hop voi CPU/GPU cua may truoc roi chay lai requirements.

## Nap du lieu

Dat file vao:

- `data/docx/` cho `.docx`
- `data/json/` cho `.json`

Sau do chay:

```powershell
python scripts/ingest.py
```

## Hoi dap

```powershell
python scripts/chat.py
```

Bat LangSmith tracing bang cach dien `LANGSMITH_API_KEY` va dat `LANGSMITH_TRACING=true` trong `.env`.

## LangSmith eval

LangSmith UI chay tren web tai `https://smith.langchain.com`. Project nay chay eval bang SDK local, sau do ket qua experiment se hien trong UI.

1. Dien `LANGSMITH_API_KEY` trong `.env`.
2. Dat `LANGSMITH_TRACING=true`.
3. Sua `data/evals/qa_examples.json` theo bo cau hoi cua ban.
4. Dam bao Qdrant da co du lieu bang `python scripts/ingest.py`.
5. Chay:

```powershell
python scripts/evaluate.py
```

Evaluator mac dinh kiem tra cau tra loi co chua cac tu/cum tu trong `must_contain`.
