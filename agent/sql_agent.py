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

Text matching rules:
- Use case-insensitive matching with LOWER(...).
- For single-word product/category filters like cookie, bread, brownie, cake, muffin, scone, pizza:
  prefer conditions like:
  LOWER(category) LIKE '%cookie%' OR LOWER(name) LIKE '%cookie%'
- For multi-word filters such as "gluten free", "whole wheat", "chocolate chip":
  do NOT require exact phrase matching only.
- Instead, match all important words using AND within a field, for example:
  LOWER(name) LIKE '%gluten%' AND LOWER(name) LIKE '%free%'
- When relevant, search multi-word filters across category, name, and description.
- If the user appears to be asking about a specific named product
  (for example "What is Blueberry Sour Cream Scone Mix?" or "Tell me about Bread Flour"),
  prioritize matching all important words in the product name using AND conditions on LOWER(name).
- For named product lookups, prefer name-based matching over broad category matching.

SQL generation rules:
- Generate a valid SQLite SELECT query only.
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, REPLACE, PRAGMA, ATTACH.
- If the user is asking for product listings, include:
  id, name, url, price, description, ingredients, size, rating, review_count AS review, image_url, category
- If the user is asking for analytics (count, sum, average, min, max, comparison, difference, grouped stats), return only the columns needed for that analysis, with clear aliases.
- Use clear aliases such as:
  COUNT(*) AS count
  SUM(price) AS total_price
  AVG(price) AS avg_price
  MAX(price) AS max_price
  MIN(price) AS min_price
- For expensive / most expensive: ORDER BY price DESC
- For cheapest / lowest price / affordable: ORDER BY price ASC
- For highest rated / best rated: ORDER BY rating DESC, review_count DESC
- For lowest rated: ORDER BY rating ASC, review_count DESC
- For most reviewed / most popular: ORDER BY review_count DESC
- Use LIMIT 10 only for product listing queries unless the user asks for a specific number
- Do not add LIMIT for single-value aggregate queries unless explicitly requested
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


def is_analytic_sql(sql: str) -> bool:
    analytic_patterns = [
        r"\bCOUNT\s*\(",
        r"\bSUM\s*\(",
        r"\bAVG\s*\(",
        r"\bMIN\s*\(",
        r"\bMAX\s*\(",
        r"\bGROUP\s+BY\b",
    ]
    return any(re.search(pattern, sql, flags=re.IGNORECASE) for pattern in analytic_patterns)


def ensure_limit(sql: str) -> str:
    if is_analytic_sql(sql):
        return sql
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
        "pizza", "waffle", "waffles", "gluten free", "gluten-free",
        "whole wheat", "chocolate chip"
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
        "gluten-free": "gluten free",
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


def is_analytic_question(question: str, chat_history: str) -> bool:
    text = f"{chat_history} {question}".lower()

    analytic_phrases = [
        "how many",
        "count",
        "sum",
        "total",
        "average",
        "avg",
        "minimum",
        "maximum",
        "min",
        "max",
        "difference",
        "compare",
        "comparison",
        "group by",
        "per category",
        "by category",
    ]
    return any(phrase in text for phrase in analytic_phrases)


def tokenize_search_phrase(text: str) -> list[str]:
    text = text.lower().strip()
    parts = re.split(r"[\s\-_\/]+", text)
    return [p for p in parts if p]


def escape_sql_like(value: str) -> str:
    return value.replace("'", "''")


def build_all_words_match_for_field(field: str, phrase: str) -> str:
    words = tokenize_search_phrase(phrase)
    if not words:
        return "1=1"

    clauses = [
        f"LOWER({field}) LIKE '%{escape_sql_like(word)}%'"
        for word in words
    ]
    return "(" + " AND ".join(clauses) + ")"


def build_all_words_match_for_fields(fields: list[str], phrase: str) -> str:
    field_clauses = [
        build_all_words_match_for_field(field, phrase)
        for field in fields
    ]
    return "(" + " OR ".join(field_clauses) + ")"


def build_single_word_match(fields: list[str], word: str) -> str:
    safe_word = escape_sql_like(word.lower())
    return "(" + " OR ".join(
        [f"LOWER({field}) LIKE '%{safe_word}%'" for field in fields]
    ) + ")"


def build_category_filter(category: str) -> str:
    fields_basic = ["category", "name"]
    fields_extended = ["category", "name", "description"]

    if " " in category or "-" in category:
        return build_all_words_match_for_fields(fields_extended, category)

    return build_single_word_match(fields_basic, category)


def extract_product_phrase(question: str) -> str | None:
    q = question.strip().lower()

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


def looks_like_specific_product_query(question: str) -> bool:
    q = question.strip().lower()

    if is_analytic_question(question, ""):
        return False

    if extract_product_phrase(question):
        return True

    trigger_phrases = [
        "what is ",
        "what's ",
        "tell me about ",
        "show me ",
        "price of ",
        "what is the price of ",
        "do you have ",
    ]

    return any(q.startswith(p) for p in trigger_phrases)


def build_exact_product_sql(product_phrase: str, limit_n: int = 5) -> str:
    words = tokenize_search_phrase(product_phrase)
    words = [w for w in words if w not in {"the", "a", "an"}]

    if not words:
        return """
SELECT id, name, url, price, description, ingredients, size, rating, review_count AS review, image_url, category
FROM products
ORDER BY rating DESC, review_count DESC
LIMIT 5
""".strip()

    name_match = " AND ".join(
        [f"LOWER(name) LIKE '%{escape_sql_like(word)}%'" for word in words]
    )

    desc_match = " AND ".join(
        [f"LOWER(description) LIKE '%{escape_sql_like(word)}%'" for word in words]
    )

    sql = f"""
SELECT id, name, url, price, description, ingredients, size, rating, review_count AS review, image_url, category
FROM products
WHERE ({name_match}) OR ({desc_match})
ORDER BY
    CASE
        WHEN ({name_match}) THEN 0
        ELSE 1
    END,
    rating DESC,
    review_count DESC
LIMIT {limit_n}
""".strip()

    return sql


def build_fallback_sql(question: str, chat_history: str) -> str:
    limit_n = extract_requested_limit(question, chat_history)

    if looks_like_specific_product_query(question):
        product_phrase = extract_product_phrase(question)
        if product_phrase:
            return build_exact_product_sql(product_phrase, limit_n=5)

    q_cat = extract_category(question)
    h_cat = extract_category(chat_history)
    category = normalize_category(q_cat or h_cat)

    order_by = infer_sort_from_context(question, chat_history)

    where_clauses = []

    if category:
        where_clauses.append(build_category_filter(category))

    if is_analytic_question(question, chat_history):
        select_clause = "SELECT COUNT(*) AS count FROM products"
        sql = select_clause
        if where_clauses:
            sql += "\nWHERE " + " AND ".join(where_clauses)
        return sql.strip()

    select_clause = """
SELECT id, name, url, price, description, ingredients, size, rating, review_count AS review, image_url, category
FROM products
""".strip()

    sql = select_clause

    if where_clauses:
        sql += "\nWHERE " + " AND ".join(where_clauses)

    sql += f"\nORDER BY {order_by}"
    sql += f"\nLIMIT {limit_n}"

    return sql.strip()


def is_product_row(row: dict) -> bool:
    if not isinstance(row, dict) or not row:
        return False

    product_indicators = {
        "id", "name", "url", "price", "description",
        "ingredients", "size", "rating", "review",
        "review_count", "image_url", "category"
    }
    matched = product_indicators.intersection(set(row.keys()))
    return len(matched) >= 3


def is_product_result(rows: list[dict]) -> bool:
    if not rows:
        return False
    return all(is_product_row(row) for row in rows)


def format_product_rows(rows: list[dict]) -> str:
    formatted = []
    for row in rows:
        formatted.append(
            f"Name: {row.get('name', 'Unknown')}\n"
            f"Price: {row.get('price', 'N/A')}\n"
            f"Category: {row.get('category', 'N/A')}\n"
            f"Rating: {row.get('rating', 'N/A')}\n"
            f"Review: {row.get('review', row.get('review_count', 'N/A'))}\n"
            f"URL: {row.get('url', '')}\n"
            f"Image URL: {row.get('image_url', '')}\n"
            f"Description: {row.get('description', '')}"
        )
    return "\n\n---\n\n".join(formatted)


def format_tabular_rows(rows: list[dict]) -> str:
    formatted = []
    for row in rows:
        parts = [f"{key}: {value}" for key, value in row.items()]
        formatted.append("\n".join(parts))
    return "\n\n---\n\n".join(formatted)


def format_rows(rows):
    if not rows:
        return "No matching results found."

    if is_product_result(rows):
        return format_product_rows(rows)

    return format_tabular_rows(rows)


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
        product_rows = rows if is_product_result(rows) else []

        return {
            **state,
            "route": "sql",
            "generated_sql": sql,
            "sql_result": result_text,
            "retrieved_docs": product_rows,
        }

    except Exception:
        fallback_sql = build_fallback_sql(question, chat_history)

        try:
            rows = execute_sql_query(fallback_sql)
            result_text = format_rows(rows)
            product_rows = rows if is_product_result(rows) else []

            return {
                **state,
                "route": "sql",
                "generated_sql": fallback_sql,
                "sql_result": result_text,
                "retrieved_docs": product_rows,
            }
        except Exception as e:
            return {
                **state,
                "route": "sql",
                "generated_sql": fallback_sql,
                "sql_result": f"SQL execution error: {str(e)}",
                "retrieved_docs": [],
            }