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
        review_count = (
            data.get("review_count")
            or data.get("reviews")
            or data.get("review_cnt")
            or data.get("num_reviews")
            or data.get("reviewCount")
            or data.get("rating_count")
            or data.get("total_reviews")
            or "N/A"
        )
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


def _fallback_answer(route: str, retrieved_docs: List[Any], sql_result: str, rag_result: str) -> str:
    if retrieved_docs:
        top = retrieved_docs[:3]
        parts = []

        for item in top:
            data = _doc_to_metadata(item)
            name = data.get("name", "Unknown")
            price = data.get("price", "N/A")
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
        "Try asking about bread, cookies, brownies, cakes, or pancake mixes."
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

    llm = ChatOpenAI(
        model=OPENAI_CHAT_MODEL,
        temperature=0.2
    )

    docs_text = _format_docs_for_prompt(retrieved_docs)
    product_count = len(retrieved_docs)

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
        final_answer = _fallback_answer(route, retrieved_docs, sql_result, rag_result)

    return {
        **state,
        "final_answer": final_answer
    }