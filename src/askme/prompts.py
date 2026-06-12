STRUCTURED_OUTPUT_INSTRUCTIONS = """Return exactly one valid JSON object with this shape:
{"answer":"string","has_enough_context":true,"confidence":0.0,"citations":[{"source_index":1,"source":"string","excerpt":"string"}],"missing_info":[]}

Rules:
- Do not wrap the JSON in markdown.
- Do not repeat the prompt.
- Use citations only when a retrieved context chunk directly supports the answer.
- Set has_enough_context=false when the retrieved context does not contain enough evidence.
- Keep confidence between 0 and 1.
"""

def build_input_classification_prompt(question: str) -> str:
    system = (
        "You are the input router for a Vietnamese RAG question-answering system. "
        "Your job is to decide whether the user message should be answered directly "
        "or should use document retrieval from the vector database. "
        "Return exactly one valid JSON object. "
        "Do not add markdown, comments, explanations, or extra text outside JSON."
    )

    user = (
        "Classify the user message into exactly one route:\n\n"

        "Route: normal\n"
        "Use normal ONLY when the message is simple conversation and does not need factual evidence.\n"
        "Examples of normal:\n"
        "- greetings: xin chao, chao ban, hello, hi\n"
        "- thanks: cam on, thank you\n"
        "- goodbye: tam biet, bye\n"
        "- small talk that does not ask for factual information\n"
        "- questions about what the assistant can do\n"
        "- requests for help using the app itself\n\n"

        "Route: rag\n"
        "Use rag when the message asks for information that may be stored in uploaded documents, "
        "the knowledge base, contracts, policies, regulations, clauses, company profiles, reports, files, "
        "or any private/domain-specific source.\n"
        "Use rag for questions asking about:\n"
        "- uploaded files or document contents\n"
        "- contracts, policies, regulations, clauses, rules, terms, obligations, rights\n"
        "- company or organization information\n"
        "- names, dates, years, numbers, addresses, products, services, timelines\n"
        "- summaries, comparisons, explanations, or extracted facts from documents\n"
        "- anything requiring evidence or citations\n\n"

        "Important routing rules:\n"
        "- If the question contains a specific company, organization, person, product, project, document name, or file name, choose rag.\n"
        "- If the question asks for 'thong tin co ban', 'tom tat', 'nam nao', 'bao nhieu', 'quy dinh', 'dieu khoan', choose rag.\n"
        "- If the user asks about Fujifilm or any company information, choose rag.\n"
        "- If the message could reasonably require document evidence, choose rag.\n"
        "- When uncertain, choose rag.\n"
        "- Do not classify a factual question as normal just because it looks general.\n"
        "- normal must not be used for company facts, dates, numbers, summaries, or document-related questions.\n\n"

        "Output requirements:\n"
        "- Return exactly one JSON object.\n"
        "- Do not wrap JSON in markdown.\n"
        "- Do not use ```json.\n"
        "- Do not repeat the prompt.\n"
        "- route must be either \"normal\" or \"rag\".\n"
        "- reason must be short.\n"
        "- confidence must be a number between 0.0 and 1.0.\n\n"

        "Output JSON shape:\n"
        "{\"route\":\"rag\",\"reason\":\"short reason\",\"confidence\":0.0}\n\n"

        "Examples:\n"
        "User: \"Xin chao\"\n"
        "Output: {\"route\":\"normal\",\"reason\":\"Greeting only; no document evidence needed.\",\"confidence\":0.99}\n\n"

        "User: \"Ban co the lam gi?\"\n"
        "Output: {\"route\":\"normal\",\"reason\":\"Asks about assistant capabilities.\",\"confidence\":0.9}\n\n"

        "User: \"mot vai thong tin co ban ve cong ty Fujifilm\"\n"
        "Output: {\"route\":\"rag\",\"reason\":\"Asks for company-specific information that should use document retrieval.\",\"confidence\":0.95}\n\n"

        "User: \"cong ty Fujifilm duoc thanh lap nam nao?\"\n"
        "Output: {\"route\":\"rag\",\"reason\":\"Asks for a specific company fact and date.\",\"confidence\":0.96}\n\n"

        "User: \"Tom tat file noidung.docx\"\n"
        "Output: {\"route\":\"rag\",\"reason\":\"Requests a summary of an uploaded document.\",\"confidence\":0.98}\n\n"

        "User: \"Hop dong quy dinh thoi han thanh toan nhu the nao?\"\n"
        "Output: {\"route\":\"rag\",\"reason\":\"Asks for a contract clause that requires retrieval.\",\"confidence\":0.96}\n\n"

        "User: \"Cam on ban\"\n"
        "Output: {\"route\":\"normal\",\"reason\":\"Thanks only; no document retrieval needed.\",\"confidence\":0.99}\n\n"

        f"User message to classify:\n{question}"
    )

    return f"{system}\n\n{user}"


def build_qa_prompt(question: str, context: str) -> str:
    system = (
        "You are a Vietnamese document QA assistant. "
        "Answer in Vietnamese, but follow the JSON schema exactly. "
        "Use the retrieved context as your only source of document-specific facts. "
        "If the question is normal conversation and no document evidence is needed, answer naturally without citations. "
        "If the question needs document knowledge but the context is empty or insufficient, say that the relevant information was not found. "
        "Never invent citations, numbers, dates, clauses, or document facts. "
        "Return only JSON. Do not write System, Human, Assistant, markdown, or explanations outside JSON."
    )
    user = (
        f"Output format instructions:\n{STRUCTURED_OUTPUT_INSTRUCTIONS}\n\n"
        f"User question:\n{question}\n\n"
        "Retrieved context chunks:\n"
        f"{context or '[No retrieved context]'}\n\n"
        "Answering instructions:\n"
        "- Base document-specific answers only on the retrieved context chunks above.\n"
        "- When citing, use source_index matching the chunk number in square brackets, copy the source value, "
        "and include a short excerpt from that chunk.\n"
        "- If the user asks for a summary, summarize only the retrieved context.\n"
        "- If context is insufficient, set has_enough_context=false, confidence<=0.3, citations=[], "
        "and list missing information in missing_info.\n"
        "- If context is sufficient, set has_enough_context=true and include citations when useful.\n"
        "- Keep the answer concise and helpful."
    )
    return f"{system}\n\n{user}"
