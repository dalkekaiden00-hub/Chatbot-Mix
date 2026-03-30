import sqlite3
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from utils.config import SQLITE_DB_PATH, OPENAI_CHAT_MODEL

load_dotenv()

llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0)

SQL_PROMPT = """
You are an expert SQLite query generator.
Customer question includes all conversation history so you have to think about this.

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

Important schema note:
- The products table contains King Arthur Baking products
- The category column represents product type/subcategory, such as bread, scone, cake, brownie, pancake, etc.
- Do NOT assume category = 'mixes'

Rules:
- Generate a valid SQLite SELECT query only
- Query only the products table
- Never use INSERT, UPDATE, DELETE, DROP, ALTER
- Use filters only when relevant to the user's request
- If the user asks for a specific category/type like bread, scone, cake, brownie, etc., filter on category
- For ranking questions like cheapest, most expensive, highest rated, or most reviewed, use ORDER BY appropriately
- Use LOWER(category) for case-insensitive matching when filtering category text
- Always include these columns in the SELECT output:
  id, name, url, price, description, ingredients, size, rating, review_count, image_url, category
- Limit results to 10 unless user asks for more
- Return only SQL, no explanation, no markdown

User question:
{chat_history}
{question}
"""


def clean_sql(sql: str) -> str:
    sql = sql.strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()
    return sql


def is_safe_select(sql: str) -> bool:
    sql_upper = sql.upper().strip()
    if not sql_upper.startswith("SELECT"):
        return False

    banned = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "ATTACH", "PRAGMA"]
    return not any(word in sql_upper for word in banned)


def execute_sql_query(sql: str):
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def sql_node(state):
    question = state["user_query"]
    chat_history = state["chat_history"]

    prompt = SQL_PROMPT.format(question=question, chat_history=chat_history)
    sql = llm.invoke(prompt).content
    sql = clean_sql(sql)

    if not is_safe_select(sql):
        return {
            "route": "sql",
            "generated_sql": sql,
            "sql_result": "No valid SQL query could be generated.",
            "retrieved_docs": []
        }

    try:
        rows = execute_sql_query(sql)

        if not rows:
            result_text = "No matching products found."
        else:
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
            result_text = "\n\n---\n\n".join(formatted)

        return {
            "route": "sql",
            "generated_sql": sql,
            "sql_result": result_text,
            "retrieved_docs": rows
        }

    except Exception as e:
        return {
            "route": "sql",
            "generated_sql": sql,
            "sql_result": f"SQL execution error: {str(e)}",
            "retrieved_docs": []
        }