from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from utils.config import OPENAI_CHAT_MODEL

load_dotenv()

llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0)

REJECT_PROMPT = """
You are a polite customer-facing assistant for King Arthur Baking products.

The user's request is outside the supported scope of this assistant.
You should respond briefly, naturally, and helpfully.

You can help with:
- King Arthur Baking mixes
- baking products
- pancake mixes
- brownie mixes
- cake mixes
- bread mixes
- cookie mixes
- gluten-free baking products

Instructions:
- Politely explain that you help with King Arthur Baking products and mixes
- If the user asked for something unsupported, say so naturally
- Redirect them toward supported baking product questions
- Be short, friendly, and helpful
- Do not mention internal systems, routing, or scope classification
- Do not say "I cannot generate an answer"

Chat history:
{chat_history}

User question:
{question}

Write a direct response to the user.
"""


def reject_node(state):
    question = state.get("user_query", "")
    chat_history = state.get("chat_history", "")

    try:
        response = llm.invoke(
            REJECT_PROMPT.format(
                question=question,
                chat_history=chat_history
            )
        ).content.strip()
    except Exception:
        response = ""

    if not response:
        response = (
            "I can help with King Arthur Baking mixes and baking products, "
            "including pancake mixes, brownie mixes, cake mixes, bread mixes, "
            "cookie mixes, and gluten-free options. If you tell me what you'd like "
            "to bake, I can recommend a product."
        )

    return {
        **state,
        "final_answer": response,
        "route": "reject"
    }