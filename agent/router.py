from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from utils.config import OPENAI_CHAT_MODEL

load_dotenv()

llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0)

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
  - Tell me about the gluten free pancake mix
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


def router_node(state):
    question = state.get("user_query", "")
    chat_history = state.get("chat_history", "")

    try:
        route = llm.invoke(
            ROUTER_PROMPT.format(
                question=question,
                chat_history=chat_history
            )
        ).content.strip().upper()
    except Exception:
        route = "RAG"

    if route not in {"SQL", "RAG", "REJECT"}:
        route = "RAG"

    q = question.lower()
    h = chat_history.lower()

    category_words = [
        "bread", "cookie", "cookies", "brownie", "brownies", "cake", "cakes",
        "pancake", "pancakes", "muffin", "muffins", "scone", "scones",
        "pizza", "waffle", "waffles", "gluten free", "gluten-free"
    ]

    sql_cues = [
        "cheap", "cheaper", "cheapest",
        "expensive", "more expensive", "most expensive",
        "highest rated", "lowest rated",
        "top ", "bottom ",
        "under $", "over $",
        "less than", "more than",
        "most reviews", "least reviews",
        "count", "how many",
        "sorted by", "sort by",
        "rating", "reviews"
    ]

    followup_cues = [
        "what about", "how about", "then", "and", "those", "ones",
        "same for", "same with", "what about those"
    ]

    comparative_followup_cues = [
        "cheaper", "more expensive", "highest rated", "lowest rated",
        "top", "best priced", "under", "over"
    ]

    is_followup = any(cue in q for cue in followup_cues)
    mentions_category = any(cat in q for cat in category_words)
    history_has_sql_intent = any(cue in h for cue in sql_cues)
    question_has_sql_intent = any(cue in q for cue in sql_cues)
    followup_keeps_structure = any(cue in q for cue in comparative_followup_cues)

    if (is_followup and mentions_category and history_has_sql_intent) or followup_keeps_structure:
        route = "SQL"

    if mentions_category and question_has_sql_intent:
        route = "SQL"

    return {
        **state,
        "route": route.lower()
    }