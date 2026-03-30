from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from utils.config import OPENAI_CHAT_MODEL

load_dotenv()

llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0)

ROUTER_PROMPT = """
You are a router for a King Arthur Baking products chatbot.
Question includes conversation history, so you have to think about this.

Choose one route:

SQL:
- structured filtering, sorting, counting, ranking, numeric constraints
- includes questions about category/type filtering such as bread, scone, cake, brownie, pancake, etc.
- use SQL for:
  - cheapest / most expensive
  - highest rated / lowest rated
  - top N / bottom N
  - under or over a price
  - review counts
  - averages, counts, min, max
  - sorting by price, rating, or reviews
- examples:
  - under $10
  - top 5 highest rated
  - most reviews
  - how many products
  - cheapest mix
  - what is the cheapest bread
  - show the top 3 scones by rating
  - give me the most expensive 10 products by price

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
- unrelated to baking products

Return only one word:
SQL
RAG
or
REJECT

Question:
{chat_history}
{question}

"""


def router_node(state):
    question = state["user_query"]
    chat_history = state["chat_history"]

    route = llm.invoke(ROUTER_PROMPT.format(question=question,chat_history=chat_history)).content.strip().upper()

    if route not in {"SQL", "RAG", "REJECT"}:
        route = "RAG"

    return {"route": route.lower()}