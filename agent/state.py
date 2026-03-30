from typing import TypedDict, List, Any


class AgentState(TypedDict, total=False):
    user_query: str
    chat_history: str
    out_of_scope: bool
    route: str
    generated_sql: str
    sql_result: str
    rag_result: str
    retrieved_docs: List[Any]
    final_answer: str