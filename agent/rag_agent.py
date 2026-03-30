import json
import math
import re
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from kb.vector_store import similarity_search
from utils.config import OPENAI_CHAT_MODEL

load_dotenv()

llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0)


def safe_float(v, default=0.0):
    try:
        return float(str(v).replace("$", "").replace(",", "").strip())
    except Exception:
        return default


def safe_int(v, default=0):
    try:
        return int(str(v).replace(",", "").strip())
    except Exception:
        return default


def extract_requested_count(query: str, default=4, max_count=6) -> int:
    if not query:
        return default

    match = re.search(r"\b([1-9]|10)\b", query)
    if match:
        return min(int(match.group(1)), max_count)

    return default


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


def normalize_doc(doc: Any) -> dict:
    if isinstance(doc, dict):
        return {
            "id": doc.get("id"),
            "name": doc.get("name", "Unknown"),
            "price": doc.get("price", ""),
            "rating": doc.get("rating", ""),
            "review": doc.get("review", doc.get("review_count", "")),
            "category": doc.get("category", ""),
            "url": doc.get("url", ""),
            "image_url": doc.get("image_url", ""),
            "description": doc.get("description", ""),
            "ingredients": doc.get("ingredients", ""),
            "size": doc.get("size", ""),
        }

    metadata = getattr(doc, "metadata", {}) or {}

    raw_description = metadata.get("description", "")
    if not raw_description:
        raw_description = clean_page_content_description(getattr(doc, "page_content", ""))

    return {
        "id": metadata.get("id"),
        "name": metadata.get("name", "Unknown"),
        "price": metadata.get("price", ""),
        "rating": metadata.get("rating", ""),
        "review": metadata.get("review", metadata.get("review_count", "")),
        "category": metadata.get("category", ""),
        "url": metadata.get("url", ""),
        "image_url": metadata.get("image_url", ""),
        "description": raw_description,
        "ingredients": metadata.get("ingredients", ""),
        "size": metadata.get("size", ""),
    }


def rank_docs(query: str, items: list[dict]) -> list[dict]:
    q = (query or "").lower()

    def recommendation_score(d):
        rating = safe_float(d.get("rating", 0))
        review_count = safe_int(d.get("review", 0))
        return rating * 2.0 + math.log1p(max(review_count, 0)) * 1.2

    def cheap_score(d):
        price = safe_float(d.get("price", 999999))
        rating = safe_float(d.get("rating", 0))
        review_count = safe_int(d.get("review", 0))
        return (price, -rating, -review_count)

    def popular_score(d):
        review_count = safe_int(d.get("review", 0))
        rating = safe_float(d.get("rating", 0))
        return (review_count, rating)

    if any(x in q for x in ["recommend", "good", "best", "top rated", "top-rated", "which product"]):
        return sorted(items, key=recommendation_score, reverse=True)

    if any(x in q for x in ["cheap", "cheaper", "lowest price", "budget", "affordable"]):
        return sorted(items, key=cheap_score)

    if any(x in q for x in ["popular", "most reviewed", "most popular"]):
        return sorted(items, key=popular_score, reverse=True)

    return sorted(
        items,
        key=lambda d: (
            safe_float(d.get("rating", 0)),
            safe_int(d.get("review", 0))
        ),
        reverse=True
    )


def filter_previous_docs(query: str, previous_docs: list[dict]) -> list[dict]:
    q = (query or "").lower()
    docs = previous_docs or []

    if not docs:
        return []

    filtered = docs

    if "gluten free" in q or "gluten-free" in q:
        filtered = [
            d for d in filtered
            if "gluten free" in f"{d.get('name', '')} {d.get('category', '')} {d.get('description', '')}".lower()
            or "gluten-free" in f"{d.get('name', '')} {d.get('category', '')} {d.get('description', '')}".lower()
        ]

    under_match = re.search(r"under\s*\$?\s*(\d+(?:\.\d+)?)", q)
    if under_match:
        threshold = float(under_match.group(1))
        filtered = [d for d in filtered if safe_float(d.get("price", 999999)) < threshold]

    over_match = re.search(r"(?:over|more than)\s*\$?\s*(\d+(?:\.\d+)?)", q)
    if over_match:
        threshold = float(over_match.group(1))
        filtered = [d for d in filtered if safe_float(d.get("price", 0)) > threshold]

    if any(x in q for x in ["cheap", "cheaper", "lowest price", "budget", "affordable"]):
        filtered = sorted(filtered, key=lambda d: safe_float(d.get("price", 999999)))

    elif any(x in q for x in ["popular", "most reviewed", "most popular"]):
        filtered = sorted(
            filtered,
            key=lambda d: (safe_int(d.get("review", 0)), safe_float(d.get("rating", 0))),
            reverse=True
        )

    elif any(x in q for x in ["best", "top rated", "highest rated", "good", "recommend"]):
        filtered = sorted(
            filtered,
            key=lambda d: (safe_float(d.get("rating", 0)), safe_int(d.get("review", 0))),
            reverse=True
        )

    return filtered


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


ANALYZE_PROMPT = """
You are analyzing a user's latest message in a multi-turn King Arthur Baking product assistant.

Your job is to classify whether the latest message is:
- a brand new product search
- a refinement of previous search results
- a follow-up asking details about a previously shown product
- a comparison among previously shown products
- ambiguous and needs clarification

Return JSON only with this schema:
{{
  "intent": "new_search" | "refine_previous_search" | "followup_detail" | "followup_attribute" | "compare_previous" | "ambiguous_followup",
  "needs_new_retrieval": true | false,
  "scope": "previous_top_product" | "previous_products" | "new_search",
  "should_ask_clarification": true | false,
  "reason": "short explanation"
}}

Guidelines:
- If the user asks about "it", "this", "that", "this one", "that one", "tell me more", "what ingredients", "is it gluten free", or similar, it is usually about previous product context.
- If the user asks "compare them", "which one is cheaper", "which is better", or similar, it is about previous products.
- If the user asks for cheaper, more popular, gluten-free only, or similar constraints on prior results, it is refine_previous_search and needs new retrieval.
- If the user clearly changes category/product type, it is new_search.
- If the message is ambiguous singular reference and there are multiple previous products, mark should_ask_clarification=true.

Chat history:
{chat_history}

Previous products:
{previous_products}

Latest query:
{question}
"""


def rewrite_query(question: str, chat_history: str) -> str:
    if not (chat_history or "").strip():
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


def is_singular_reference(query: str) -> bool:
    q = (query or "").lower()
    patterns = [
        r"\bit\b",
        r"\bthis\b",
        r"\bthat\b",
        r"\bthis one\b",
        r"\bthat one\b",
        r"\bthe first one\b",
        r"\bthe second one\b",
        r"\btell me more\b",
        r"\bmore details\b",
        r"\bwhat ingredients\b",
        r"\bwhat size\b",
        r"\bhow much\b",
        r"\bis it\b",
        r"\bdoes it\b",
    ]
    return any(re.search(p, q) for p in patterns)


def is_plural_reference(query: str) -> bool:
    q = (query or "").lower()
    patterns = [
        r"\bthem\b",
        r"\bthose\b",
        r"\bthese\b",
        r"\bcompare\b",
        r"\bwhich one\b",
        r"\bwhich is\b",
        r"\bwhat'?s the difference\b",
        r"\bdifference between\b",
        r"\bamong those\b",
        r"\bamong these\b",
    ]
    return any(re.search(p, q) for p in patterns)


def heuristic_analyze_question(query: str, previous_docs: list[dict]) -> dict:
    q = (query or "").lower().strip()
    prev_count = len(previous_docs or [])

    compare_markers = [
        "compare", "difference", "different", "which one", "which is better",
        "which is cheaper", "among those", "among these", "better of the two"
    ]
    refine_markers = [
        "cheaper", "cheapest", "lower price", "budget", "affordable",
        "more popular", "popular", "best rated", "higher rated", "gluten free only",
        "gluten-free only", "under $", "under ", "less than ", "more like this",
        "similar ones", "show more", "another option"
    ]
    detail_markers = [
        "tell me more", "more details", "details", "ingredients", "size",
        "price", "rating", "reviews", "review", "is it", "does it",
        "what is it", "describe it"
    ]
    new_search_markers = [
        "recommend", "show me", "find", "search", "looking for", "i want",
        "do you have", "what products", "best"
    ]

    if prev_count > 0 and any(x in q for x in compare_markers):
        return {
            "intent": "compare_previous",
            "needs_new_retrieval": False,
            "scope": "previous_products",
            "should_ask_clarification": False,
            "reason": "heuristic compare with previous products"
        }

    if prev_count > 0 and any(x in q for x in refine_markers):
        return {
            "intent": "refine_previous_search",
            "needs_new_retrieval": True,
            "scope": "new_search",
            "should_ask_clarification": False,
            "reason": "heuristic refinement of prior results"
        }

    if prev_count == 1 and (is_singular_reference(q) or any(x in q for x in detail_markers)):
        return {
            "intent": "followup_attribute",
            "needs_new_retrieval": False,
            "scope": "previous_top_product",
            "should_ask_clarification": False,
            "reason": "heuristic follow-up on single previous product"
        }

    if prev_count > 1 and is_singular_reference(q):
        return {
            "intent": "ambiguous_followup",
            "needs_new_retrieval": False,
            "scope": "previous_products",
            "should_ask_clarification": True,
            "reason": "heuristic ambiguous singular reference with multiple previous products"
        }

    if any(x in q for x in new_search_markers):
        return {
            "intent": "new_search",
            "needs_new_retrieval": True,
            "scope": "new_search",
            "should_ask_clarification": False,
            "reason": "heuristic new product search"
        }

    return {
        "intent": "new_search" if prev_count == 0 else "refine_previous_search",
        "needs_new_retrieval": True,
        "scope": "new_search",
        "should_ask_clarification": False,
        "reason": "heuristic fallback"
    }


def analyze_question(query: str, chat_history: str, previous_docs: list[dict]) -> dict:
    previous_products_text = "\n".join(
        f"- {d.get('name', 'Unknown')} | price={d.get('price', '')} | rating={d.get('rating', '')} | reviews={d.get('review', '')}"
        for d in (previous_docs or [])[:6]
    ).strip()

    heuristic = heuristic_analyze_question(query, previous_docs)

    try:
        raw = llm.invoke(
            ANALYZE_PROMPT.format(
                chat_history=chat_history or "",
                previous_products=previous_products_text or "None",
                question=query or ""
            )
        ).content.strip()

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return heuristic

        parsed = json.loads(match.group(0))

        intent = parsed.get("intent", heuristic["intent"])
        needs_new_retrieval = bool(parsed.get("needs_new_retrieval", heuristic["needs_new_retrieval"]))
        scope = parsed.get("scope", heuristic["scope"])
        should_ask_clarification = bool(parsed.get("should_ask_clarification", heuristic["should_ask_clarification"]))
        reason = parsed.get("reason", heuristic["reason"])

        prev_count = len(previous_docs or [])

        if prev_count == 0 and not needs_new_retrieval:
            return {
                "intent": "new_search",
                "needs_new_retrieval": True,
                "scope": "new_search",
                "should_ask_clarification": False,
                "reason": "no previous docs available, forcing new retrieval"
            }

        if prev_count <= 1 and should_ask_clarification:
            should_ask_clarification = False

        return {
            "intent": intent,
            "needs_new_retrieval": needs_new_retrieval,
            "scope": scope,
            "should_ask_clarification": should_ask_clarification,
            "reason": reason,
        }
    except Exception:
        return heuristic


def format_docs(items: list[dict]) -> str:
    doc_summaries = []
    for item in items:
        summary = f"""
Name: {item.get('name', '')}
Price: {item.get('price', '')}
Rating: {item.get('rating', '')}
Review: {item.get('review', '')}
Category: {item.get('category', '')}
URL: {item.get('url', '')}
Image Url: {item.get('image_url', '')}
Description: {item.get('description', '')}
Ingredients: {item.get('ingredients', '')}
Size: {item.get('size', '')}
"""
        doc_summaries.append(summary.strip())

    return "\n\n---\n\n".join(doc_summaries)


def build_clarification_response(previous_docs: list[dict]) -> str:
    names = [d.get("name", "Unknown") for d in (previous_docs or [])[:6]]
    if not names:
        return "Which product do you mean?"

    if len(names) == 1:
        return f"Do you mean {names[0]}?"

    return "Which product do you mean: " + ", ".join(names) + "?"


def rag_node(state):
    query = state.get("user_query", "")
    chat_history = state.get("chat_history", "")
    previous_docs_raw = state.get("retrieved_docs", []) or []
    previous_docs = [normalize_doc(doc) for doc in previous_docs_raw]

    analysis = analyze_question(query, chat_history, previous_docs)

    if not analysis.get("needs_new_retrieval", True):
        if analysis.get("should_ask_clarification", False):
            clarification = build_clarification_response(previous_docs)
            return {
                **state,
                "route": "rag",
                "rag_result": clarification,
                "final_answer": clarification,
                "retrieved_docs": previous_docs,
                "rag_retrieved_count": len(previous_docs),
                "rag_selected_count": len(previous_docs),
                "rag_analysis": analysis,
            }

        if analysis.get("scope") == "previous_top_product":
            selected_docs = previous_docs[:1]
        elif analysis.get("scope") == "previous_products":
            requested_count = extract_requested_count(
                query,
                default=min(len(previous_docs), 4) if previous_docs else 4,
                max_count=6
            )
            selected_docs = previous_docs[:requested_count]
        else:
            selected_docs = previous_docs[:1] if previous_docs else []

        if not selected_docs:
            return {
                **state,
                "route": "rag",
                "rag_result": "No relevant product information found.",
                "retrieved_docs": [],
                "rag_retrieved_count": 0,
                "rag_selected_count": 0,
                "rag_analysis": analysis,
            }

        rag_result = format_docs(selected_docs)

        return {
            **state,
            "route": "rag",
            "rag_result": rag_result,
            "retrieved_docs": selected_docs,
            "rag_retrieved_count": len(previous_docs),
            "rag_selected_count": len(selected_docs),
            "rag_analysis": analysis,
        }

    filtered_previous_docs = filter_previous_docs(query, previous_docs)
    if filtered_previous_docs:
        requested_count = extract_requested_count(
            query,
            default=min(len(filtered_previous_docs), 4),
            max_count=6
        )
        selected_docs = filtered_previous_docs[:requested_count]
        rag_result = format_docs(selected_docs)

        return {
            **state,
            "route": "rag",
            "rag_result": rag_result,
            "retrieved_docs": selected_docs,
            "rag_retrieved_count": len(previous_docs),
            "rag_selected_count": len(selected_docs),
            "rag_analysis": {
                **analysis,
                "reason": f"{analysis.get('reason', '')}; reused filtered previous docs"
            },
            "retrieval_query": query,
        }

    retrieval_query = rewrite_query(query, chat_history)
    docs = similarity_search(retrieval_query, k=12)
    rag_retrieved_count = len(docs)

    if not docs:
        return {
            **state,
            "route": "rag",
            "rag_result": "No relevant product information found.",
            "retrieved_docs": previous_docs,
            "rag_retrieved_count": 0,
            "rag_selected_count": 0,
            "rag_analysis": analysis,
            "retrieval_query": retrieval_query,
        }

    retrieved_docs = [normalize_doc(doc) for doc in docs]
    ranked_docs = rank_docs(retrieval_query, retrieved_docs)
    requested_count = extract_requested_count(query, default=4, max_count=6)
    selected_docs = ranked_docs[:requested_count]
    rag_result = format_docs(selected_docs)

    return {
        **state,
        "route": "rag",
        "rag_result": rag_result,
        "retrieved_docs": selected_docs,
        "rag_retrieved_count": rag_retrieved_count,
        "rag_selected_count": len(selected_docs),
        "rag_analysis": analysis,
        "retrieval_query": retrieval_query,
    }