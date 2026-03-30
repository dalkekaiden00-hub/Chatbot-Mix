from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from utils.config import OPENAI_CHAT_MODEL

load_dotenv()

llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0)


def is_obviously_product_question(q: str) -> bool:
    q = q.lower()
    keywords = [
        "product", "products", "mix", "mixes", "baking",
        "cake", "brownie", "pancake", "gluten free",
        "recommend", "best", "price", "rating", "review",
        "ingredient", "buy"
    ]
    return any(word in q for word in keywords)


GUARDRAIL_PROMPT = """
You are checking whether a user question is in scope for a King Arthur Baking product assistant.
You are a guardrail classifier for a shopping assistant that only answers questions about baking mixes and closely related baked goods sold in a product catalog.
Question includes conversation history that you have to think.

Domain scope:
- Baking mixes and related products such as bread mixes, scone mixes, cookie mixes, brownie mixes, cake mixes, pancake mixes, muffin mixes, pizza crust mixes, gluten-free mixes, seasonal mixes, limited-time products, and similar baked-goods products.
- Product attributes such as category, flavor, price, rating, popularity, availability, limited-time status, ingredients, dietary tags, and counts of products.
- Simple comparative and analytical shopping questions that can be answered from catalog data, such as cheapest, most expensive, highest rated, most popular, how many, which category has the most items, and similar.
- questions about baking mixes or baking products
- recommendations about products in the catalog
- ingredients, prices, ratings, reviews, categories

Out of scope:
- weather, politics, coding, medicine, travel, sports
- unrelated retail categories not supported by the catalog

Return only:
IN
or
OUT

Question:
{chat_history}
{question}

"""


def guardrail_node(state):
    question = state["user_query"]
    chat_history = state["chat_history"]

    if is_obviously_product_question(question):
        return {"out_of_scope": False}

    result = llm.invoke(GUARDRAIL_PROMPT.format(question=question,chat_history=chat_history)).content.strip().upper()

    return {"out_of_scope": result != "IN"}