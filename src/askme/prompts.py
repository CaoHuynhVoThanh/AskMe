from langchain_core.prompts import ChatPromptTemplate


STRUCTURED_OUTPUT_INSTRUCTIONS = """Return only valid JSON with this shape:
{"answer":"string","has_enough_context":true,"confidence":0.0,"citations":[{"source_index":1,"source":"string","excerpt":"string"}],"missing_info":[]}
Use citations only when a retrieved context chunk supports the answer.
"""


QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ban la tro ly hoi dap. Chi tra loi dua tren ngu canh duoc cung cap. "
            "Neu thieu thong tin, hay noi ro rang rang ban chua tim thay du lieu phu hop. "
            "Tra ve dung dinh dang JSON theo schema duoc yeu cau.",
        ),
        (
            "human",
            "Huong dan dinh dang:\n{format_instructions}\n\n"
            "Cau hoi: {question}\n\n"
            "Ngu canh:\n{context}\n\n"
            "Hay tra loi ngan gon, co dan chung neu co.",
        ),
    ]
)
