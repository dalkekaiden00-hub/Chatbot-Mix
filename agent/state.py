from typing import Any, Dict, List, TypedDict


class AgentState(TypedDict, total=False):
    user_query: str
    chat_history: str

    out_of_scope: bool
    route: str

    generated_sql: str
    sql_result: str

    rag_result: str
    retrieved_docs: List[Any]

    rag_retrieved_count: int
    rag_selected_count: int
    rag_analysis: Dict[str, Any]
    retrieval_query: str

    final_answer: str