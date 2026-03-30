import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from utils.config import OPENAI_CHAT_MODEL

load_dotenv()

llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0)

CATEGORY_WORDS = [
    "bread", "cookie", "cookies", "brownie", "brownies", "cake", "cakes",
    "pancake", "pancakes", "muffin", "muffins", "scone", "scones",
    "pizza", "waffle", "waffles", "gluten free", "gluten-free",
    "whole wheat", "chocolate chip"
]

SQL_CUES = [
    "cheap", "cheaper", "cheapest",
    "expensive", "more expensive", "most expensive",
    "highest rated", "lowest rated",
    "top ", "bottom ",
    "under $", "over $",
    "less than", "more than",
    "most reviews", "least reviews",
    "count", "how many",
    "sorted by", "sort by",
    "rating", "reviews",
    "price", "cost"
]

FOLLOWUP_CUES = [
    "what about", "how about", "then", "and", "those", "ones",
    "same for", "same with", "what about those"
]

COMPARATIVE_FOLLOWUP_CUES = [
    "cheaper", "more expensive", "highest rated", "lowest rated",
    "top", "best priced", "under", "over"
]

SPECIFIC_PRODUCT_PATTERNS = [
    r"^what is .+",
    r"^what's .+",
    r"^tell me about .+",
    r"^show me .+",
    r"^price of .+",
    r"^what is the price of .+",
    r"^do you have .+",
]

ROUTER_PROMPT = """
You are a router for a King Arthur Baking products chatbot.
You must use BOTH the current user question and the chat history.

Important rule for follow-up questions:
- Users often ask short follow-ups that omit important constraints from earlier turns.
- You must infer carried-over intent from chat history.
- If the previous turn involved structured constraints like:
  - expensive / cheapest
  - highest rated / lowest rated
  - top N / bottom N
  - under or over a price
  - counts, ranking, sorting
  and the new question changes only the category or product type,
  then route to SQL.

Important rule for specific product lookup:
- If the user appears to be asking about a specific named product,
  route to SQL.
- Examples:
  - "What is Blueberry Sour Cream Scone Mix?"
  - "Tell me about Bread Flour"
  - "Price of Belgian Waffle Mix"
  - "Do you have Gluten-Free Pancake Mix?"

Examples of follow-up behavior:
- "show me the cheapest bread mixes" -> "what about cookies?"
  => SQL
- "top 5 pancake mixes" -> "what about brownies then?"
  => SQL
- "expensive bread options" -> "what about cookies then?"
  => SQL
- "under $10 cake mixes" -> "and muffins?"
  => SQL
- "highest rated brownies" -> "same for cookies"
  => SQL
- "cheapest bread" -> "what about gluten free?"
  => SQL if the user is still asking for structured filtering

Choose one route:

SQL:
- structured filtering, sorting, counting, ranking, numeric constraints
- exact or likely exact product lookup
- category/type filtering such as bread, scone, cake, brownie, pancake, cookie, muffin, pizza, etc.
- follow-up queries that inherit structured constraints from prior turns
- use SQL for:
  - cheapest / most expensive / expensive
  - highest rated / lowest rated
  - top N / bottom N
  - under or over a price
  - review counts
  - averages, counts, min, max
  - sorting by price, rating, or reviews
  - specific product lookup by name

RAG:
- recommendations
- vague product discovery
- product overview questions
- ingredients
- descriptions
- comparisons
- semantic understanding
- examples:
  - Which product is good?
  - What do you recommend?
  - What are your products?
  - Which mix is good for breakfast?

REJECT:
- unrelated to King Arthur Baking products

Return only one word:
SQL
RAG
or
REJECT

Chat history:
{chat_history}

Question:
{question}
"""


def _contains_any(text: str, phrases: list[str]) -> bool:
    text = (text or "").lower()
    return any(phrase in text for phrase in phrases)


def _is_analytic_query(q: str) -> bool:
    q = (q or "").lower()
    phrases = [
        "how many", "count", "sum", "total", "average", "avg",
        "minimum", "maximum", "min", "max",
        "difference", "compare", "comparison",
        "per category", "by category"
    ]
    return any(p in q for p in phrases)


def _is_specific_product_query(q: str) -> bool:
    q = (q or "").strip().lower()
    return any(re.match(pattern, q) for pattern in SPECIFIC_PRODUCT_PATTERNS)


def router_node(state):
    question = state.get("user_query", "")
    chat_history = state.get("chat_history", "")

    q = question.lower()
    h = chat_history.lower()

    try:
        response = llm.invoke(
            ROUTER_PROMPT.format(
                question=question,
                chat_history=chat_history
            )
        )
        route = response.content.strip().upper()
    except Exception:
        route = "RAG"

    if route not in {"SQL", "RAG", "REJECT"}:
        route = "RAG"

    mentions_category = _contains_any(q, CATEGORY_WORDS)
    history_has_sql_intent = _contains_any(h, SQL_CUES)
    question_has_sql_intent = _contains_any(q, SQL_CUES)
    is_followup = _contains_any(q, FOLLOWUP_CUES)
    followup_keeps_structure = _contains_any(q, COMPARATIVE_FOLLOWUP_CUES)
    is_analytic = _is_analytic_query(q)
    is_specific_product = _is_specific_product_query(q)

    if is_analytic:
        route = "SQL"

    if is_specific_product:
        route = "SQL"

    if mentions_category and question_has_sql_intent:
        route = "SQL"

    if (is_followup and mentions_category and history_has_sql_intent) or followup_keeps_structure:
        route = "SQL"

    return {
        **state,
        "route": route.lower()
    }