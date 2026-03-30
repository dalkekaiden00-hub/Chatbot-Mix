import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from utils.config import OPENAI_CHAT_MODEL


def _doc_to_metadata(doc: Any) -> Dict[str, Any]:
    if hasattr(doc, "metadata"):
        return doc.metadata or {}
    if isinstance(doc, dict):
        return doc
    return {}


def _format_price(value: Any) -> str:
    if value is None or value == "":
        return "N/A"
    try:
        return f"${float(value):.2f}"
    except Exception:
        return str(value)


def _normalize_review_count(data: Dict[str, Any]) -> Any:
    return (
        data.get("review_count")
        or data.get("reviews")
        or data.get("review_cnt")
        or data.get("num_reviews")
        or data.get("reviewCount")
        or data.get("rating_count")
        or data.get("total_reviews")
        or data.get("review")
        or "N/A"
    )


def _format_docs_for_prompt(retrieved_docs: List[Any]) -> str:
    if not retrieved_docs:
        return "No retrieved products."

    chunks = []
    for i, doc in enumerate(retrieved_docs, start=1):
        data = _doc_to_metadata(doc)

        name = data.get("name", "Unknown")
        category = data.get("category", "N/A")
        price = data.get("price", "N/A")
        rating = (
            data.get("rating")
            or data.get("average_rating")
            or data.get("stars")
            or "N/A"
        )
        review_count = _normalize_review_count(data)
        description = data.get("description") or data.get("content") or ""
        url = data.get("url", "")

        chunk = f"""
Product {i}
Name: {name}
Category: {category}
Price: {price}
Rating: {rating}
Review Count: {review_count}
Description: {description}
URL: {url}
        """.strip()

        chunks.append(chunk)

    return "\n\n".join(chunks)


def _tokenize_search_phrase(text: str) -> list[str]:
    text = (text or "").lower().strip()
    parts = re.split(r"[\s\-_\/]+", text)
    return [p for p in parts if p]


def _extract_product_phrase(question: str) -> str | None:
    q = (question or "").strip().lower()

    patterns = [
        r"^what is (.+?)\??$",
        r"^what's (.+?)\??$",
        r"^tell me about (.+?)\??$",
        r"^show me (.+?)\??$",
        r"^price of (.+?)\??$",
        r"^what is the price of (.+?)\??$",
        r"^do you have (.+?)\??$",
    ]

    for pattern in patterns:
        match = re.match(pattern, q)
        if match:
            phrase = match.group(1).strip()
            if phrase:
                return phrase
    return None


def _is_specific_product_query(question: str) -> bool:
    return _extract_product_phrase(question) is not None


def _is_high_confidence_name_match(question: str, row: dict) -> bool:
    phrase = _extract_product_phrase(question)
    if not phrase or not row.get("name"):
        return False

    q_words = set(_tokenize_search_phrase(phrase))
    q_words = {w for w in q_words if w not in {"the", "a", "an"}}

    name_words = set(_tokenize_search_phrase(str(row["name"])))
    return bool(q_words) and q_words.issubset(name_words)


def _looks_like_aggregate_row(row: Dict[str, Any]) -> bool:
    if not row:
        return False

    aggregate_keys = {
        "count", "avg_price", "average_price", "total_price",
        "max_price", "min_price", "sum", "avg", "min", "max"
    }
    return any(key in row for key in aggregate_keys)


def _is_product_row(row: dict) -> bool:
    if not isinstance(row, dict) or not row:
        return False

    product_indicators = {
        "id", "name", "url", "price", "description",
        "ingredients", "size", "rating", "review",
        "review_count", "image_url", "category"
    }
    matched = product_indicators.intersection(set(row.keys()))
    return len(matched) >= 3


def _is_product_result(rows: list[dict]) -> bool:
    if not rows:
        return False
    return all(_is_product_row(row) for row in rows)


def _build_exact_product_answer(question: str, retrieved_docs: List[Any]) -> str | None:
    if not retrieved_docs:
        return None

    top = _doc_to_metadata(retrieved_docs[0])
    if not top:
        return None

    name = top.get("name", "This product")
    category = top.get("category", "N/A")
    price = _format_price(top.get("price"))
    rating = top.get("rating", "N/A")
    review_count = _normalize_review_count(top)
    description = top.get("description", "")
    url = top.get("url", "")

    if _is_high_confidence_name_match(question, top):
        answer = f"{name} is in the {category} category. It costs {price}"
        if rating != "N/A":
            answer += f" and has a rating of {rating}"
            if review_count != "N/A":
                answer += f" based on {review_count} reviews"
        answer += "."
        if description:
            answer += f" {description}"
        if url:
            answer += f" You can find it here: {url}"
        return answer

    matches = []
    for item in retrieved_docs[:3]:
        data = _doc_to_metadata(item)
        matches.append(
            f"{data.get('name', 'Unknown')} ({_format_price(data.get('price'))}, category: {data.get('category', 'N/A')})"
        )
    return "I couldn’t confirm one exact product match, but I found similar products: " + "; ".join(matches) + "."


def _build_sql_product_list_answer(retrieved_docs: List[Any]) -> str | None:
    if not retrieved_docs:
        return None

    top = [_doc_to_metadata(x) for x in retrieved_docs[:3]]
    if not top:
        return None

    if len(top) == 1:
        item = top[0]
        return (
            f"I found 1 matching product: {item.get('name', 'Unknown')} "
            f"({_format_price(item.get('price'))}, rating {item.get('rating', 'N/A')}, "
            f"category: {item.get('category', 'N/A')})."
        )

    parts = []
    for item in top:
        parts.append(
            f"{item.get('name', 'Unknown')} "
            f"({_format_price(item.get('price'))}, rating {item.get('rating', 'N/A')})"
        )

    return "Here are the top matching products: " + "; ".join(parts) + "."


def _build_sql_aggregate_answer(sql_result: str, retrieved_docs: List[Any]) -> str | None:
    if retrieved_docs:
        first = _doc_to_metadata(retrieved_docs[0])
        if _looks_like_aggregate_row(first):
            parts = [f"{k}: {v}" for k, v in first.items()]
            return "Here’s what I found: " + ", ".join(parts) + "."

    if sql_result and sql_result != "No matching results found.":
        return sql_result

    return None


def _fallback_answer(route: str, retrieved_docs: List[Any], sql_result: str, rag_result: str) -> str:
    if retrieved_docs:
        top = retrieved_docs[:3]
        parts = []

        for item in top:
            data = _doc_to_metadata(item)
            name = data.get("name", "Unknown")
            price = _format_price(data.get("price", "N/A"))
            category = data.get("category", "N/A")
            rating = (
                data.get("rating")
                or data.get("average_rating")
                or data.get("stars")
                or "N/A"
            )
            parts.append(f"{name} ({category}, {price}, rating {rating})")

        return "Here are some relevant options: " + "; ".join(parts) + "."

    if route == "sql" and sql_result:
        return sql_result

    if rag_result:
        return rag_result

    return (
        "I couldn’t find a strong match right now. "
        "Try asking about bread, cookies, brownies, cakes, waffles, scones, or pancake mixes."
    )


def synth_node(state: Dict[str, Any]) -> Dict[str, Any]:
    if state.get("final_answer"):
        return state

    user_query = state.get("user_query", "")
    chat_history = state.get("chat_history", "")
    retrieved_docs = state.get("retrieved_docs", [])
    route = state.get("route", "unknown")
    sql_result = state.get("sql_result", "")
    rag_result = state.get("rag_result", "")
    generated_sql = state.get("generated_sql", "")

    normalized_docs = [_doc_to_metadata(doc) for doc in retrieved_docs]
    docs_text = _format_docs_for_prompt(normalized_docs)
    product_count = len(normalized_docs)

    if route == "sql":
        if _is_specific_product_query(user_query) and normalized_docs and _is_product_result(normalized_docs):
            exact_answer = _build_exact_product_answer(user_query, normalized_docs)
            if exact_answer:
                return {
                    **state,
                    "final_answer": exact_answer
                }

        if normalized_docs and _is_product_result(normalized_docs):
            list_answer = _build_sql_product_list_answer(normalized_docs)
            if list_answer:
                return {
                    **state,
                    "final_answer": list_answer
                }

        aggregate_answer = _build_sql_aggregate_answer(sql_result, normalized_docs)
        if aggregate_answer:
            return {
                **state,
                "final_answer": aggregate_answer
            }

    llm = ChatOpenAI(
        model=OPENAI_CHAT_MODEL,
        temperature=0.2
    )

    system_prompt = """
You are a helpful King Arthur Baking product recommendation assistant.

Your job:
- Answer the user's question using only the provided product data.
- If the route is SQL, prioritize structured facts like price, rating, review count, counts, ranking, and filters.
- If the route is RAG, prioritize semantic relevance, descriptions, and product fit.
- Recommend the most relevant products for the user's stated need.
- Use product metadata such as category, price, rating, review count, and description.
- If multiple products are relevant, briefly compare them.
- Be concise, clear, natural, and user-friendly.
- Do not invent facts that are not present in the data.
- If the data is incomplete, say so briefly and still give the best available answer.

Critical rules:
- Refer only to the products actually provided.
- Do not say "three products" unless exactly 3 products were provided.
- Do not mention products that are not in the retrieved products list.
- If there is 1 product, recommend that single product clearly.
- If there are 2 to 3 products, mention the best option first and briefly compare the others.
- If there are more than 3 products, summarize the best few without listing everything.
- If route is SQL and the question is about a specific product, stay tightly grounded in the top matching product and do not broaden to unrelated alternatives unless explicitly stated as similar items.

Response style:
- Start with a direct answer or recommendation.
- Then add a short reason.
- Keep the answer to about 2 to 5 sentences.
- Do not output markdown bullets unless necessary.
- Do not include raw URLs unless specifically useful.
""".strip()

    human_prompt = f"""
Route: {route}

Number of retrieved products available to reference: {product_count}

Chat history:
{chat_history}

User query:
{user_query}

Generated SQL:
{generated_sql}

SQL result summary:
{sql_result}

RAG result summary:
{rag_result}

Retrieved products:
{docs_text}

Now write the final answer for the user.
""".strip()

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        final_answer = response.content if hasattr(response, "content") else str(response)
    except Exception:
        final_answer = _fallback_answer(route, normalized_docs, sql_result, rag_result)

    return {
        **state,
        "final_answer": final_answer
    }