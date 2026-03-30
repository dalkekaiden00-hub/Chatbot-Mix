import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from kb.vector_store import similarity_search
from utils.config import OPENAI_CHAT_MODEL

load_dotenv()

llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0)


def safe_float(v, default=0.0):
    try:
        return float(str(v).replace("$", "").strip())
    except Exception:
        return default


def safe_int(v, default=0):
    try:
        return int(str(v).replace(",", "").strip())
    except Exception:
        return default


def rank_docs(query: str, items: list[dict]) -> list[dict]:
    q = query.lower()

    if any(x in q for x in ["recommend", "good", "best", "top rated", "top-rated", "which product"]):
        return sorted(
            items,
            key=lambda d: (
                safe_float(d.get("rating", 0)),
                safe_int(d.get("review_count", 0))
            ),
            reverse=True
        )

    if any(x in q for x in ["cheap", "cheaper", "lowest price", "budget", "affordable"]):
        return sorted(
            items,
            key=lambda d: safe_float(d.get("price", 999999))
        )

    return items


def clean_page_content_description(text: str) -> str:
    if not text:
        return ""

    text = str(text).strip()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    match = re.search(r"Description:\s*(.*)", text, re.IGNORECASE)
    if match:
        text = match.group(1).strip()

    text = re.split(
        r"\b(Product Name|Name|Category|Price|Rating|Review Count|Reviews|URL|Image URL|Image Url|Ingredients|Size):",
        text,
        flags=re.IGNORECASE
    )[0].strip()

    return text


REWRITE_PROMPT = """
Rewrite the user's latest query into a standalone product-search query for King Arthur Baking products.

Rules:
- Use the chat history to resolve omitted context in follow-up questions.
- Preserve relevant product type or preference from the history when needed.
- Keep it short and search-friendly.
- Output only the rewritten query.

Chat history:
{chat_history}

Latest query:
{question}
"""


def rewrite_query(question: str, chat_history: str) -> str:
    if not chat_history.strip():
        return question

    try:
        rewritten = llm.invoke(
            REWRITE_PROMPT.format(
                question=question,
                chat_history=chat_history
            )
        ).content.strip()
        return rewritten or question
    except Exception:
        return question


def rag_node(state):
    query = state.get("user_query", "")
    chat_history = state.get("chat_history", "")

    retrieval_query = rewrite_query(query, chat_history)

    docs = similarity_search(retrieval_query, k=8)

    if not docs:
        return {
            **state,
            "route": "rag",
            "rag_result": "No relevant product information found.",
            "retrieved_docs": []
        }

    retrieved_docs = []
    for doc in docs:
        metadata = doc.metadata or {}

        raw_description = metadata.get("description", "")
        if not raw_description:
            raw_description = clean_page_content_description(doc.page_content)

        item = {
            "id": metadata.get("id"),
            "name": metadata.get("name", "Unknown"),
            "price": metadata.get("price", ""),
            "rating": metadata.get("rating", ""),
            "review_count": metadata.get("review_count", ""),
            "category": metadata.get("category", ""),
            "url": metadata.get("url", ""),
            "image_url": metadata.get("image_url", ""),
            "description": raw_description,
            "ingredients": metadata.get("ingredients", ""),
            "size": metadata.get("size", "")
        }
        retrieved_docs.append(item)

    ranked_docs = rank_docs(retrieval_query, retrieved_docs)

    doc_summaries = []
    for item in ranked_docs[:4]:
        summary = f"""
Name: {item['name']}
Price: {item['price']}
Rating: {item['rating']}
Review Count: {item['review_count']}
Category: {item['category']}
URL: {item['url']}
Image Url: {item['image_url']}
Description: {item['description']}
"""
        doc_summaries.append(summary.strip())

    rag_result = "\n\n---\n\n".join(doc_summaries)

    return {
        **state,
        "route": "rag",
        "rag_result": rag_result,
        "retrieved_docs": ranked_docs
    }