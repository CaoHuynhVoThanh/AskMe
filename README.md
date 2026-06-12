# AskMe RAG

AskMe là dự án hỏi đáp dựa trên dữ liệu nội bộ. Dữ liệu nguồn hiện hỗ trợ `.docx` và `.json`, được tách chunk, nhúng vector, lưu vào Qdrant local, sau đó truy vấn bằng LangChain và điều phối bằng LangGraph.

Hệ thống đang dùng:

- Qdrant local làm vector database.
- Hugging Face/Sentence Transformers cho embedding và reranking.
- GGUF local qua `llama-cpp-python` để sinh câu trả lời.
- Pydantic để ép output có cấu trúc.
- LangSmith để tracing/evaluation khi cần.

Output trả lời có dạng:

```json
{
  "answer": "string",
  "has_enough_context": true,
  "confidence": 0.0,
  "citations": [],
  "missing_info": []
}
```

## Cấu Trúc

```text
.
+-- data/
|   +-- docx/              # Đặt file .docx tại đây
|   +-- json/              # Đặt file .json tại đây
|   +-- evals/             # Bộ câu hỏi/đáp án mẫu để eval bằng LangSmith
+-- scripts/
|   +-- ingest.py          # Nạp dữ liệu vào Qdrant
|   +-- chat.py            # Chat CLI
|   +-- debug_retrieval.py # Chạy riêng retrieve/rerank để debug tài liệu được chọn
|   +-- evaluate.py        # Chạy eval và gửi experiment lên LangSmith
+-- src/
|   +-- askme/
|       +-- config.py      # Cấu hình runtime chính của dự án
|       +-- document_loaders.py
|       +-- embeddings.py
|       +-- graph.py       # LangGraph pipeline
|       +-- llm.py         # GGUF/LlamaCpp loader
|       +-- prompts.py
|       +-- reranker.py
|       +-- schemas.py
|       +-- vectorstore.py
+-- docker-compose.yml     # Qdrant local
+-- .env.example           # Mẫu secrets
+-- requirements.txt
+-- pyproject.toml
```

## Luồng Xử Lý

```mermaid
graph TD
    START([START]) --> classify_input[classify_input]

    classify_input -->|rag| retrieve[retrieve]
    classify_input -->|normal| prepare_context[prepare_context]

    retrieve --> rerank[rerank]
    rerank --> prepare_context
    prepare_context --> generate[generate]
    generate --> END([END])
```

Ý nghĩa các node:

- `classify_input`: dùng Qwen GGUF để phân loại câu hỏi là `normal` hay `rag`.
- `retrieve`: lấy 20 tài liệu/chunk có khả năng liên quan từ Qdrant.
- `rerank`: dùng CrossEncoder để chọn top 5 tài liệu liên quan nhất.
- `prepare_context`: ghép context và cắt theo token budget.
- `generate`: gọi GGUF local để sinh JSON answer theo schema Pydantic.

## Cài Đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Nếu cài `llama-cpp-python` trên Windows bị lỗi build, hãy cài wheel phù hợp CPU/GPU trước rồi chạy lại `pip install -r requirements.txt`.

## Secrets Và Cấu Hình

Toàn bộ cấu hình runtime nằm trong:

```text
src/askme/config.py
```

File `.env` chỉ dùng cho secrets và đã được ignore bởi Git. Tạo `.env` từ file mẫu:

```powershell
Copy-Item .env.example .env
```

Sau đó điền nếu cần:

```env
LANGSMITH_API_KEY=
HF_TOKEN=
```

Lưu ý: không đặt các cấu hình như `LLM_N_CTX`, `QDRANT_URL`, `RETRIEVER_TOP_K` trong `.env` nữa. Muốn đổi runtime settings thì sửa trực tiếp trong `src/askme/config.py`.

## Chạy Qdrant Local

```powershell
docker compose up -d qdrant
```

Nếu không dùng Docker, hãy chạy Qdrant riêng và sửa `qdrant_url` trong `src/askme/config.py`.

## Cấu Hình GGUF

Model mặc định:

```python
llm_model_path = Path("E:/LLMs/Vi-Qwen2-7B-RAG.Q2_K.gguf")
```

Profile mặc định đang ưu tiên chạy được trên máy ít RAM:

```python
llm_n_ctx = 4096
llm_context_fallbacks = "4096"
llm_max_input_tokens = 3072
llm_context_token_budget = 2200
llm_n_batch = 128
```

Nếu máy đủ RAM/VRAM, có thể chuyển sang profile context lớn đã được comment sẵn trong `config.py`:

```python
# llm_n_ctx: int = 32768
# llm_context_fallbacks: str = "32768,24576,16384,8192,4096"
# llm_max_input_tokens: int = 28672
# llm_context_token_budget: int = 24000
# llm_n_batch: int = 512
```

Lưu ý quan trọng:

- `llm_n_ctx` càng lớn thì càng tốn RAM/VRAM do KV cache.
- Nếu gặp lỗi `Failed to create llama_context`, hãy giảm `llm_n_ctx`.
- Nếu `debug_llm_config=True`, khi khởi động model sẽ in các lần thử context bằng `llama_cpp_config_attempt`.
- Khi dùng context lớn, nên cân nhắc tắt reranker để giảm áp lực RAM.

## Retrieval Và Rerank

Cấu hình mặc định:

```python
retriever_top_k = 20
reranker_top_k = 5
enable_reranker = True
debug_reranker = True
```

Nghĩa là Qdrant lấy 20 ứng viên, sau đó CrossEncoder reranker chọn top 5 chunk liên quan nhất để đưa vào prompt.

Nếu thiếu RAM khi load reranker, hệ thống sẽ tự fallback về thứ tự retriever và lấy 5 tài liệu đầu tiên. Có thể tắt reranker trong `config.py`:

```python
enable_reranker = False
```

## Nạp Dữ Liệu

Đặt dữ liệu vào:

- `data/docx/` cho file `.docx`
- `data/json/` cho file `.json`

Nạp dữ liệu:

```powershell
python scripts/ingest.py
```

Nếu muốn xóa collection cũ rồi index lại từ đầu:

```powershell
python scripts/ingest.py --reset
```

Lưu ý:

- Sau khi đổi `CHUNK_SIZE`/`chunk_size` hoặc dữ liệu nguồn, nên chạy lại `ingest.py --reset`.
- Nếu Qdrant chưa chạy, ingest và chat sẽ lỗi kết nối.

## Chat

```powershell
python scripts/chat.py
```

Khi `debug_input_classification=True`, mỗi câu hỏi sẽ in JSON phân loại đầu vào:

```text
[debug] input_classification: {"route":"rag","reason":"...","confidence":0.96}
```

Khi `debug_reranker=True`, hệ thống cũng in trạng thái reranker đã chạy, bị tắt, hoặc fallback vì lỗi RAM.

## LangSmith Tracing Và Eval

LangSmith UI chạy trên web tại:

```text
https://smith.langchain.com
```

Để bật tracing/eval:

1. Điền `LANGSMITH_API_KEY` trong `.env`.
2. Đặt `langsmith_tracing = True` trong `src/askme/config.py`.
3. Sửa `data/evals/qa_examples.json` theo bộ câu hỏi của bạn.
4. Đảm bảo Qdrant đã có dữ liệu bằng `python scripts/ingest.py`.
5. Chạy:

```powershell
python scripts/evaluate.py
```

Evaluator mặc định kiểm tra câu trả lời có chứa các từ/cụm từ trong `must_contain`.

## Lưu Ý Vận Hành

- GGUF context lớn không miễn phí: `n_ctx=32768` có thể làm thiếu RAM ngay cả với model quant nhỏ.
- CrossEncoder reranker cũng tốn RAM vì dùng PyTorch/Transformers.
- Nếu máy yếu, dùng `llm_n_ctx=4096`, `llm_n_batch=128`, và cân nhắc `enable_reranker=False`.
- Nếu model trả lời lặp prompt, kiểm tra `prompts.py` và stop tokens trong `LLM_STOP_TOKENS`.
- Nếu output không parse được JSON, hệ thống sẽ fallback về raw answer với `has_enough_context=False`.
