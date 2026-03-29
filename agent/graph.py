from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.guardrails import guardrail_node
from agent.router import router_node
from agent.sql_agent import sql_node
from agent.rag_agent import rag_node
from agent.synth_agent import synth_node
from agent.reject_agent import reject_node


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("guardrail", guardrail_node)
    workflow.add_node("router", router_node)
    workflow.add_node("sql", sql_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("synthesis", synth_node)
    workflow.add_node("reject", reject_node)

    workflow.set_entry_point("guardrail")

    # guardrail branch
    workflow.add_conditional_edges(
        "guardrail",
        lambda state: "reject" if state.get("out_of_scope") else "router",
        {
            "router": "router",
            "reject": "reject",
        },
    )

    # router branch
    workflow.add_conditional_edges(
        "router",
        lambda state: state.get("route", "reject"),
        {
            "sql": "sql",
            "rag": "rag",
            "reject": "reject",
        },
    )

    workflow.add_edge("sql", "synthesis")
    workflow.add_edge("rag", "synthesis")

    workflow.add_edge("synthesis", END)
    workflow.add_edge("reject", END)

    return workflow.compile()

graph = build_graph()