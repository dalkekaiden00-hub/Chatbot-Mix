from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from utils.config import OPENAI_CHAT_MODEL

load_dotenv()

llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0)


def format_product_line(doc: dict) -> str:
    name = doc.get("name", "Unknown product")
    price = doc.get("price")
    rating = doc.get("rating")
    review_count = doc.get("review_count")
    category = doc.get("category")

    parts = [name]

    if category:
        parts.append(f"category: {category}")
    if price not in [None, ""]:
        parts.append(f"price: {price}")
    if rating not in [None, ""]:
        if review_count not in [None, ""]:
            parts.append(f"rating: {rating} ({review_count} reviews)")
        else:
            parts.append(f"rating: {rating}")

    return " | ".join(parts)


def recommend_from_docs(retrieved_docs: list[dict], question: str) -> str:
    if not retrieved_docs:
        return ""

    top_docs = retrieved_docs[:3]
    product_lines = [f"- {format_product_line(doc)}" for doc in top_docs]

    return (
        "Here are a few King Arthur Baking products you may want to consider:\n\n"
        + "\n".join(product_lines)
        + "\n\nIf you tell me what you want to bake—such as pancakes, brownies, cake, or gluten-free items—I can narrow it down further."
    )


def sales_fallback_answer(question: str) -> str:
    q = question.lower().strip()

    if "cheese" in q:
        return (
            "I couldn’t find a cheese product in the King Arthur Baking mixes catalog. "
            "I can help with baking mixes such as pancake mixes, brownie mixes, cake mixes, "
            "and gluten-free options. If you want, I can recommend one of those."
        )

    if "which product is good" in q or "what do you recommend" in q or "which one should i buy" in q:
        return (
            "We have several strong baking mix options depending on what you want to make. "
            "I can help you choose a pancake mix, brownie mix, cake mix, or gluten-free option."
        )

    if "product" in q or "products" in q or "mix" in q or "mixes" in q:
        return (
            "I can help with King Arthur Baking products and baking mixes. "
            "We offer items like pancake mixes, brownie mixes, cake mixes, and gluten-free baking options. "
            "Tell me what you’d like to bake, and I’ll recommend a product."
        )

    return (
        "I can help with King Arthur Baking mixes and related baking products. "
        "Tell me what you’d like to bake, and I’ll recommend a product."
    )


SQL_SYNTH_PROMPT = """
You are formatting SQL query results for a user.
User question includes conversation history, so you have to think about this.

User question:
{chat_history}
{question}

SQL result:
{sql_result}

Instructions:
- The SQL result is authoritative
- Answer directly from the SQL result
- If the user asked for rankings, top N, cheapest, most expensive, highest rated, counts, or filters, present the result clearly as a list
- Do not say the information is unavailable if SQL result contains rows
- Do not redirect the user to general categories
- Be concise and factual
"""


RAG_SYNTH_PROMPT = """
You are a friendly sales assistant for King Arthur Baking mixes.


User question:
{chat_history}
{question}

RAG result:
{rag_result}

Instructions:
- Answer like a helpful sales assistant
- Use the retrieved information
- If nothing relevant is found, clearly say that
- Then guide the user toward available King Arthur Baking product categories
- Be concise and helpful
"""


def looks_like_sql_failure(sql_result: str) -> bool:
    if not sql_result:
        return True

    text = str(sql_result).strip().lower()

    return (
        text in ["", "none", "[]"]
        or text.startswith("sql execution error:")
        or text == "no valid sql query could be generated."
    )


def is_empty_rag(rag_result: str) -> bool:
    if not rag_result:
        return True

    text = str(rag_result).strip()
    return text in ["", "None", "[]", "No relevant product information found."]


def synth_node(state):
    question = state.get("user_query", "")
    chat_history = state.get("chat_history","")
    route = state.get("route", "")
    sql_result = state.get("sql_result", "")
    rag_result = state.get("rag_result", "")
    retrieved_docs = state.get("retrieved_docs", [])

    # For RAG queries, prefer product recommendation from retrieved docs
    if route == "rag":
        recommended = recommend_from_docs(retrieved_docs, question)
        if recommended:
            return {
                **state,
                "final_answer": recommended
                }

    # SQL route: if SQL worked, use SQL result directly
    if route == "sql" and not looks_like_sql_failure(sql_result) and sql_result != "No matching products found.":
        response = llm.invoke(
            SQL_SYNTH_PROMPT.format(
                question=question,
                chat_history=chat_history,
                sql_result=sql_result,
            )
        ).content.strip()

        if response:
            return {
                **state,
                "final_answer": response
                }

        return {
            **state,
            "final_answer": sql_result
            }

    # SQL route but no rows
    if route == "sql" and sql_result == "No matching products found.":
        return {
            **state,
            "final_answer": "I couldn’t find any matching products for that request."
            }

    # RAG route
    if route == "rag" and not is_empty_rag(rag_result):
        response = llm.invoke(
            RAG_SYNTH_PROMPT.format(
                question=question,
                rag_result=rag_result,
            )
        ).content.strip()

        if response:
            return {
                **state,
                "final_answer": response
                }

    return {
        **state,
        "final_answer": sales_fallback_answer(question)
        }