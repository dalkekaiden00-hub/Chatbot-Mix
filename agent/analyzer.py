import json
from langchain_openai import ChatOpenAI
from agent.state import AgentState
from utils.config import OPENAI_API_KEY

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=OPENAI_API_KEY
)

SYSTEM_PROMPT = """
You are a sales assistant for King Arthur Baking Mixes products only.

Your job is to analyze the user's question and decide whether to:
1. answer
2. clarify
3. reject

Rules:
- Answer if the query is clearly about King Arthur Baking mixes, including product discovery,
  prices, ingredients, descriptions, comparisons, recommendations, dietary preferences,
  sizes, ratings, and shopping guidance.
- Clarify if the query is relevant but too vague to answer well.
- Reject if the query is unrelated to King Arthur Baking mixes, asks for unsupported general knowledge,
  or falls outside the product catalog domain.

If decision is "answer", also choose route:
- sql: for structured/product database questions like price, count, cheapest, list, filters
- rag: for descriptive/semantic questions like ingredients, features, suitability, summaries
- both: if the query needs both structured data and descriptive context

Return valid JSON only in this format:
{
  "decision": "answer | clarify | reject",
  "reason": "short explanation",
  "route": "sql | rag | both | null",
  "clarification_question": "question to ask user if decision=clarify, else null"
}
"""

def analyze_query(state: AgentState) -> AgentState:
    user_query = state["user_query"]
    chat_history = state["chat_history"]

    prompt = f"""
User query: {chat_history,user_query}
"""

    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ])

    content = response.content.strip()

    try:
        result = json.loads(content)
    except Exception:
        result = {
            "decision": "reject",
            "reason": "Could not classify the request safely.",
            "route": None,
            "clarification_question": None
        }

    state["decision"] = result.get("decision")
    state["reason"] = result.get("reason")
    state["route"] = result.get("route")
    state["clarification_question"] = result.get("clarification_question")
    state["steps"] = state.get("steps", []) + [f"Analyzer: {state['decision']} ({state['route']})"]

    return state