from langchain_core.prompts import ChatPromptTemplate


QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ban la tro ly hoi dap. Chi tra loi dua tren ngu canh duoc cung cap. "
            "Neu thieu thong tin, hay noi ro rang rang ban chua tim thay du lieu phu hop.",
        ),
        (
            "human",
            "Cau hoi: {question}\n\nNgu canh:\n{context}\n\nHay tra loi ngan gon, co dan chung neu co.",
        ),
    ]
)
