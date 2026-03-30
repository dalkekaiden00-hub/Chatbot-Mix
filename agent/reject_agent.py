from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from utils.config import OPENAI_CHAT_MODEL

load_dotenv()

llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0)

REJECT_PROMPT = """
You are a polite sales assistant for King Arthur Baking products.
Customer question includes all conversation history so you have to think about this.

The user asked:
{chat_history}
{question}


This question is out of scope for the available catalog or product assistant.

Your job:
- Politely explain that you can help with King Arthur Baking products and mixes
- If the requested item/category is unsupported, say so naturally
- Redirect the user toward supported product areas like pancake mixes, brownie mixes, cake mixes, and gluten-free baking
- Be short, helpful, and friendly
- Do not say "I cannot generate an answer"

Write a direct customer-facing response.
"""


def reject_node(state):
    question = state.get("user_query", "")
    chat_history = state.get("chat_history","")

    response = llm.invoke(REJECT_PROMPT.format(question=question,chat_history=chat_history)).content.strip()

    if not response:
        response = (
            "I can help with King Arthur Baking mixes and baking products, "
            "such as pancake mixes, brownie mixes, cake mixes, and gluten-free options. "
            "If you tell me what you'd like to bake, I can recommend a product."
        )

    return {
        "final_answer": response,
        "route": "reject"
    }