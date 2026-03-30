from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from utils.config import OPENAI_CHAT_MODEL

load_dotenv()

llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0)


def is_obviously_product_question(q: str) -> bool:
    q = q.lower()
    keywords = [
        "product", "products", "mix", "mixes", "baking",
        "cake", "brownie", "pancake", "bread", "muffin",
        "scone", "pizza", "cookie", "cookies",
        "gluten free", "gluten-free",
        "recommend", "best", "price", "rating", "review",
        "ingredient", "ingredients", "buy", "category", "categories",
        "cheap", "cheapest", "expensive", "top", "lowest", "highest"
    ]
    return any(word in q for word in keywords)


def looks_like_followup(q: str) -> bool:
    q = q.lower().strip()
    followup_cues = [
        "what about", "how about", "then", "and", "those", "ones",
        "what about those", "and those", "what about cookies", "what about brownies"
    ]
    short_category_terms = [
        "bread", "cookie", "cookies", "brownie", "brownies", "cake", "cakes",
        "pancake", "pancakes", "muffin", "muffins", "scone", "scones",
        "pizza", "waffle", "waffles", "gluten free", "gluten-free"
    ]

    return (
        any(cue in q for cue in followup_cues)
        or any(term in q for term in short_category_terms)
    )


def history_suggests_product_domain(chat_history: str) -> bool:
    h = chat_history.lower()
    product_cues = [
        "king arthur", "product", "products", "mix", "mixes", "baking",
        "cake", "brownie", "bread", "cookie", "cookies", "pancake",
        "muffin", "scone", "pizza", "gluten free", "gluten-free",
        "price", "rating", "review", "ingredients", "category",
        "cheap", "cheapest", "expensive", "top", "highest rated", "lowest rated"
    ]
    return any(cue in h for cue in product_cues)


GUARDRAIL_PROMPT = """
You are checking whether a user question is in scope for a King Arthur Baking product assistant.
You are a guardrail classifier for a shopping assistant that only answers questions about baking mixes and closely related baked goods sold in a product catalog.
Question includes conversation history that you have to think about.

Important follow-up rule:
- Short follow-up questions may omit important context.
- If the chat history is clearly about King Arthur Baking products, and the new question appears to be a follow-up like
  "what about cookies then?" or "and brownies?",
  classify it as IN.

Domain scope:
- Baking mixes and related products such as bread mixes, scone mixes, cookie mixes, brownie mixes, cake mixes, pancake mixes, muffin mixes, pizza crust mixes, gluten-free mixes, seasonal mixes, limited-time products, and similar baked-goods products.
- Product attributes such as category, flavor, price, rating, popularity, availability, limited-time status, ingredients, dietary tags, and counts of products.
- Simple comparative and analytical shopping questions that can be answered from catalog data, such as cheapest, most expensive, highest rated, most popular, how many, which category has the most items, and similar.
- Questions about baking mixes or baking products.
- Recommendations about products in the catalog.
- Ingredients, prices, ratings, reviews, categories.

Out of scope:
- weather, politics, coding, medicine, travel, sports
- unrelated retail categories not supported by the catalog

Return only:
IN
or
OUT

Chat history:
{chat_history}

Question:
{question}
"""


def guardrail_node(state):
    question = state.get("user_query", "")
    chat_history = state.get("chat_history", "")

    if is_obviously_product_question(question):
        return {
            **state,
            "out_of_scope": False
        }

    if looks_like_followup(question) and history_suggests_product_domain(chat_history):
        return {
            **state,
            "out_of_scope": False
        }

    try:
        result = llm.invoke(
            GUARDRAIL_PROMPT.format(question=question, chat_history=chat_history)
        ).content.strip().upper()
    except Exception:
        result = "IN" if history_suggests_product_domain(chat_history) else "OUT"

    return {
        **state,
        "out_of_scope": result != "IN"
    }