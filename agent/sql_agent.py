import sqlite3
import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from utils.config import SQLITE_DB_PATH, OPENAI_CHAT_MODEL

load_dotenv()

llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0)

SQL_PROMPT = """
You are an expert SQLite query generator for a King Arthur Baking product database.
You must interpret the latest question in the context of the full chat history.

Database table:
products(
    id INTEGER,
    name TEXT,
    url TEXT,
    price REAL,
    description TEXT,
    ingredients TEXT,
    size TEXT,
    rating REAL,
    review_count INTEGER,
    image_url TEXT,
    category TEXT
)

Important conversation rule:
- Users may ask short follow-up questions that omit previous constraints.
- You must carry forward relevant structured constraints from chat history.
- Example:
  Chat history: "show me expensive bread products"
  Latest question: "what about cookies then?"
  Meaning: "show me expensive cookie products"

Schema rules:
- Query only the products table.
- Do NOT assume category = 'mixes'.
- If user refers to product type like bread, cookie, brownie, cake, pancake, muffin, scone, pizza, gluten free, etc., use category filtering and optionally name filtering when helpful.

SQL generation rules:
- Generate a valid SQLite SELECT query only.
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, REPLACE, PRAGMA, ATTACH.
- Always include these columns in SELECT:
  id, name, url, price, description, ingredients, size, rating, review_count, image_url, category
- Use case-insensitive matching with LOWER(...)
- For category filters, prefer:
  LOWER(category) LIKE '%cookie%' OR LOWER(name) LIKE '%cookie%'
- For expensive / most expensive: ORDER BY price DESC
- For cheapest / lowest price / affordable: ORDER BY price ASC
- For highest rated / best rated: ORDER BY rating DESC, review_count DESC
- For lowest rated: ORDER BY rating ASC, review_count DESC
- For most reviewed / most popular: ORDER BY review_count DESC
- Use LIMIT 10 unless the user asks for a specific number
- Return only SQL, with no explanation and no markdown

Chat history:
{chat_history}

Latest user question:
{question}
"""


def clean_sql(sql: str) -> str:
    sql = sql.strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()
    sql = sql.rstrip(";").strip()
    return sql


def is_safe_select(sql: str) -> bool:
    sql_upper = sql.upper().strip()

    if not sql_upper.startswith("SELECT"):
        return False

    banned = [
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
        "TRUNCATE", "ATTACH", "PRAGMA", "CREATE", "REPLACE"
    ]
    if any(word in sql_upper for word in banned):
        return False

    if ";" in sql:
        return False

    if "PRODUCTS" not in sql_upper:
        return False

    return True


def ensure_limit(sql: str) -> str:
    if re.search(r"\bLIMIT\s+\d+\b", sql, flags=re.IGNORECASE):
        return sql
    return f"{sql} LIMIT 10"


def execute_sql_query(sql: str):
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def extract_requested_limit(question: str, chat_history: str) -> int:
    text = f"{chat_history} {question}".lower()
    match = re.search(r"\btop\s+(\d+)\b", text)
    if match:
        return int(match.group(1))
    match = re.search(r"\bshow\s+(\d+)\b", text)
    if match:
        return int(match.group(1))
    match = re.search(r"\blimit\s+(\d+)\b", text)
    if match:
        return int(match.group(1))
    return 10


def extract_category(text: str) -> str | None:
    text = text.lower()
    categories = [
        "bread", "cookie", "cookies", "brownie", "brownies", "cake", "cakes",
        "pancake", "pancakes", "muffin", "muffins", "scone", "scones",
        "pizza", "waffle", "waffles", "gluten free", "gluten-free"
    ]
    for cat in categories:
        if cat in text:
            return cat
    return None


def normalize_category(cat: str | None) -> str | None:
    if not cat:
        return None
    mapping = {
        "cookies": "cookie",
        "brownies": "brownie",
        "cakes": "cake",
        "pancakes": "pancake",
        "muffins": "muffin",
        "scones": "scone",
        "waffles": "waffle",
        "gluten-free": "gluten free"
    }
    return mapping.get(cat, cat)


def infer_sort_from_context(question: str, chat_history: str):
    text = f"{chat_history} {question}".lower()

    if any(x in text for x in ["cheapest", "cheap", "cheaper", "lowest price", "affordable", "budget"]):
        return "price ASC"
    if any(x in text for x in ["most expensive", "expensive", "higher price", "premium"]):
        return "price DESC"
    if any(x in text for x in ["highest rated", "best rated", "top rated", "top-rated"]):
        return "rating DESC, review_count DESC"
    if any(x in text for x in ["lowest rated"]):
        return "rating ASC, review_count DESC"
    if any(x in text for x in ["most reviewed", "most popular", "most reviews"]):
        return "review_count DESC"

    return "rating DESC, review_count DESC"


def build_fallback_sql(question: str, chat_history: str) -> str:
    limit_n = extract_requested_limit(question, chat_history)

    q_cat = extract_category(question)
    h_cat = extract_category(chat_history)
    category = normalize_category(q_cat or h_cat)

    order_by = infer_sort_from_context(question, chat_history)

    select_clause = """
SELECT id, name, url, price, description, ingredients, size, rating, review_count, image_url, category
FROM products
""".strip()

    where_clauses = []

    if category:
        cat_value = category.replace("'", "''")
        where_clauses.append(
            f"(LOWER(category) LIKE '%{cat_value}%' OR LOWER(name) LIKE '%{cat_value}%')"
        )

    sql = select_clause

    if where_clauses:
        sql += "\nWHERE " + " AND ".join(where_clauses)

    sql += f"\nORDER BY {order_by}"
    sql += f"\nLIMIT {limit_n}"

    return sql.strip()


def format_rows(rows):
    if not rows:
        return "No matching products found."

    formatted = []
    for row in rows:
        formatted.append(
            f"Name: {row.get('name', 'Unknown')}\n"
            f"Price: {row.get('price', 'N/A')}\n"
            f"Category: {row.get('category', 'N/A')}\n"
            f"Rating: {row.get('rating', 'N/A')}\n"
            f"Review Count: {row.get('review_count', 'N/A')}\n"
            f"URL: {row.get('url', '')}\n"
            f"Image URL: {row.get('image_url', '')}\n"
            f"Description: {row.get('description', '')}"
        )
    return "\n\n---\n\n".join(formatted)


def sql_node(state):
    question = state.get("user_query", "")
    chat_history = state.get("chat_history", "")

    prompt = SQL_PROMPT.format(question=question, chat_history=chat_history)

    try:
        sql = llm.invoke(prompt).content
    except Exception:
        sql = ""

    sql = clean_sql(sql)

    if sql:
        sql = ensure_limit(sql)

    if not sql or not is_safe_select(sql):
        sql = build_fallback_sql(question, chat_history)

    try:
        rows = execute_sql_query(sql)
        result_text = format_rows(rows)

        return {
            **state,
            "route": "sql",
            "generated_sql": sql,
            "sql_result": result_text,
            "retrieved_docs": rows,
        }

    except Exception:
        fallback_sql = build_fallback_sql(question, chat_history)

        try:
            rows = execute_sql_query(fallback_sql)
            result_text = format_rows(rows)

            return {
                **state,
                "route": "sql",
                "generated_sql": fallback_sql,
                "sql_result": result_text,
                "retrieved_docs": rows,
            }
        except Exception as e:
            return {
                **state,
                "route": "sql",
                "generated_sql": fallback_sql,
                "sql_result": f"SQL execution error: {str(e)}",
                "retrieved_docs": [],
            }