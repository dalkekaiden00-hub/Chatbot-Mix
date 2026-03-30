import os
import sys
import html
import re

import requests
import streamlit as st
from dotenv import load_dotenv



PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv()

st.set_page_config(
    page_title="King Arthur Mixes Chatbot",
    page_icon="🍰",
    layout="wide"
)

# ---------- Ensure knowledge base exists BEFORE importing graph ----------
DB_PATH = os.path.join(PROJECT_ROOT, "data", "db", "mixes.db")
VECTORSTORE_DIR = os.path.join(PROJECT_ROOT, "data", "vectorstore")
FAISS_FILE = os.path.join(VECTORSTORE_DIR, "index.faiss")
PKL_FILE = os.path.join(VECTORSTORE_DIR, "index.pkl")


def ensure_knowledge_base():
    db_exists = os.path.exists(DB_PATH)
    vector_exists = os.path.exists(FAISS_FILE) and os.path.exists(PKL_FILE)

    if db_exists and vector_exists:
        return True

    with st.spinner("Preparing knowledge base for first run..."):
        try:
            from build_kb import main as build_kb_main
            build_kb_main()
            return True
        except Exception as e:
            st.error("Failed to build knowledge base.")
            st.exception(e)
            return False


def format_chat_history(messages):
    lines = []
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


kb_ready = ensure_knowledge_base()

graph = None
graph_import_error = None

if kb_ready:
    try:
        from agent.graph import graph
    except Exception as e:
        graph_import_error = str(e)
else:
    graph_import_error = "Knowledge base could not be built."

# ---------- Custom CSS ----------
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}

.chat-shell {
    background: #ffffff;
    border: 1px solid #ececec;
    border-radius: 18px;
    padding: 1rem;
    box-shadow: 0 6px 20px rgba(0,0,0,0.04);
}

.chat-card {
    border-radius: 18px;
    padding: 14px 16px;
    margin-bottom: 12px;
    line-height: 1.6;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}

.user-card {
    background: #f3f4f6;
    border: 1px solid #e5e7eb;
}

.assistant-card {
    background: #fff8f1;
    border: 1px solid #fde7d3;
}

.product-native-card {
    background: #ffffff;
    border: 1px solid #ececec;
    border-radius: 16px;
    padding: 14px;
    margin-bottom: 16px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.04);
}

.product-title {
    font-weight: 700;
    font-size: 1rem;
    color: #1f2937;
    margin-top: 0.5rem;
    margin-bottom: 0.5rem;
}

.meta-row {
    font-size: 0.93rem;
    color: #4b5563;
    margin-bottom: 0.2rem;
}

.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.8rem;
    margin-right: 6px;
    margin-top: 6px;
    background: #f3f4f6;
    border: 1px solid #e5e7eb;
    color: #374151;
}

.side-card {
    background: #ffffff;
    border: 1px solid #ececec;
    border-radius: 16px;
    padding: 1rem;
    box-shadow: 0 4px 14px rgba(0,0,0,0.04);
    margin-bottom: 1rem;
}

.small-muted {
    color: #6b7280;
    font-size: 0.92rem;
}

.stChatInputContainer {
    border-top: none !important;
    padding-top: 0.6rem;
}

div.stButton > button {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


def build_workflow_graph(selected_route=None, rejected=False):
    active_route = "reject" if rejected else selected_route

    def node_attrs(name):
        base = {
            "shape": "box",
            "style": "rounded,filled",
            "fillcolor": "white",
            "color": "gray60",
            "fontcolor": "black",
            "penwidth": "1.2",
        }

        if active_route == "sql" and name in ["UserQuery", "Guardrail", "Router", "SQL", "Synthesis", "FinalAnswer"]:
            base.update({
                "fillcolor": "lightskyblue",
                "color": "deepskyblue4",
                "penwidth": "2.2",
            })

        elif active_route == "rag" and name in ["UserQuery", "Guardrail", "Router", "RAG", "Synthesis", "FinalAnswer"]:
            base.update({
                "fillcolor": "palegreen",
                "color": "darkgreen",
                "penwidth": "2.2",
            })

        elif active_route == "reject" and name in ["UserQuery", "Guardrail", "RejectLLM", "FinalAnswer"]:
            base.update({
                "fillcolor": "mistyrose",
                "color": "firebrick4",
                "fontcolor": "black",
                "penwidth": "2.2",
            })

        elif name == "FinalAnswer":
            base.update({
                "fillcolor": "plum1",
                "color": "purple4",
            })

        return ", ".join(f'{k}="{v}"' for k, v in base.items())

    def edge_attrs(src, dst, label=""):
        base = {
            "color": "gray70",
            "fontcolor": "gray40",
            "penwidth": "1.2",
        }

        active_edges = set()
        active_color = None

        if active_route == "sql":
            active_edges = {
                ("UserQuery", "Guardrail"),
                ("Guardrail", "Router"),
                ("Router", "SQL"),
                ("SQL", "Synthesis"),
                ("Synthesis", "FinalAnswer"),
            }
            active_color = "deepskyblue4"

        elif active_route == "rag":
            active_edges = {
                ("UserQuery", "Guardrail"),
                ("Guardrail", "Router"),
                ("Router", "RAG"),
                ("RAG", "Synthesis"),
                ("Synthesis", "FinalAnswer"),
            }
            active_color = "darkgreen"

        elif active_route == "reject":
            active_edges = {
                ("UserQuery", "Guardrail"),
                ("Guardrail", "RejectLLM"),
                ("RejectLLM", "FinalAnswer"),
            }
            active_color = "firebrick4"

        if (src, dst) in active_edges:
            base.update({
                "color": active_color,
                "fontcolor": active_color,
                "penwidth": "2.4",
            })

        if label:
            base["label"] = label

        return ", ".join(f'{k}="{v}"' for k, v in base.items())

    return f"""
    digraph LangGraphFlow {{
        rankdir=TB;
        labelloc="t";
        fontsize=20;
        fontname="Arial";

        node [fontname="Arial", fontsize=12];
        edge [fontname="Arial", fontsize=11];

        UserQuery   [{node_attrs("UserQuery")} label="User Query"];
        Guardrail   [{node_attrs("Guardrail")} label="Guardrail"];
        Router      [{node_attrs("Router")} label="Router"];
        SQL         [{node_attrs("SQL")} label="SQL Node"];
        RAG         [{node_attrs("RAG")} label="RAG Node"];
        Synthesis   [{node_attrs("Synthesis")} label="Final Synthesis"];
        RejectLLM   [{node_attrs("RejectLLM")} label="Reject / Out of Scope (LLM)"];
        FinalAnswer [{node_attrs("FinalAnswer")} label="Final Answer"];

        UserQuery -> Guardrail   [{edge_attrs("UserQuery", "Guardrail")}];
        Guardrail -> Router      [{edge_attrs("Guardrail", "Router", "in scope")}];
        Guardrail -> RejectLLM   [{edge_attrs("Guardrail", "RejectLLM", "out of scope")}];
        Router -> SQL            [{edge_attrs("Router", "SQL", "sql")}];
        Router -> RAG            [{edge_attrs("Router", "RAG", "rag")}];
        Router -> RejectLLM      [{edge_attrs("Router", "RejectLLM", "reject")}];
        SQL -> Synthesis         [{edge_attrs("SQL", "Synthesis")}];
        RAG -> Synthesis         [{edge_attrs("RAG", "Synthesis")}];
        Synthesis -> FinalAnswer [{edge_attrs("Synthesis", "FinalAnswer")}];
        RejectLLM -> FinalAnswer [{edge_attrs("RejectLLM", "FinalAnswer")}];
    }}
    """


def render_chat_bubble(role: str, content: str):
    safe_content = html.escape(str(content)).replace("\n", "<br>")
    css_class = "user-card" if role == "user" else "assistant-card"
    st.markdown(
        f'<div class="chat-card {css_class}">{safe_content}</div>',
        unsafe_allow_html=True
    )


def clean_description_text(text):
    if not text:
        return ""

    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    match = re.search(r"Description:\s*(.*)", text, re.IGNORECASE)
    if match:
        text = match.group(1).strip()

    text = re.split(
        r"\b(Product Name|Name|Category|Price|Rating|Review Count|Reviews|URL|Image URL|Image Url|Ingredients|Size):",
        text,
        flags=re.IGNORECASE
    )[0].strip()

    return text


def clean_assistant_answer(answer: str) -> str:
    if not answer:
        return ""

    text = str(answer).strip()
    text = text.replace("**", "")
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[([^\]]+)\]\((.*?)\)', r'\1', text)

    cleaned_lines = []
    skip_description_block = False

    for line in text.splitlines():
        stripped = line.strip()

        if re.match(r"^-?\s*(Image|URL)\s*:", stripped, re.IGNORECASE):
            continue

        if re.match(r"^-?\s*(Price|Category|Rating|Review Count|Reviews)\s*:", stripped, re.IGNORECASE):
            continue

        if re.match(r"^-?\s*Description\s*:", stripped, re.IGNORECASE):
            skip_description_block = True
            continue

        if skip_description_block:
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if paragraphs:
        return paragraphs[0]

    return text


def normalize_price(price):
    if isinstance(price, (int, float)):
        return f"${price:.2f}"
    if price not in [None, "", "N/A"]:
        price = str(price).strip()
        return price if price.startswith("$") else f"${price}"
    return "N/A"


@st.cache_data(show_spinner=False)
def fetch_image_bytes(url: str, timeout: int = 15):
    if not url:
        return None

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Referer": "https://shop.kingarthurbaking.com/",
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "image" not in content_type.lower():
            return None

        return response.content
    except Exception:
        return None


def render_badges(category, rating, price_text):
    badge_html = '<div style="margin-top:8px; margin-bottom:8px;">'
    if category not in [None, "", "N/A"]:
        badge_html += f'<span class="badge">{html.escape(str(category))}</span>'
    if rating not in [None, "", "N/A"]:
        badge_html += f'<span class="badge">⭐ {html.escape(str(rating))}</span>'
    if price_text not in [None, "", "N/A"]:
        badge_html += f'<span class="badge">{html.escape(str(price_text))}</span>'
    badge_html += '</div>'
    st.markdown(badge_html, unsafe_allow_html=True)

def render_product_cards(retrieved_docs):
    if not retrieved_docs:
        return

    st.markdown("##### Suggested products")
    cols = st.columns(2)

    for i, doc in enumerate(retrieved_docs[:6]):
        data = doc.metadata if hasattr(doc, "metadata") else doc

        name = data.get("name", "Unknown")
        price = data.get("price")
        category = data.get("category", "N/A")
        rating = (
            data.get("rating")
            or data.get("average_rating")
            or data.get("stars")
            or "N/A"
        )

        review_count = (
            data.get("review_count")
            or data.get("reviews")
            or data.get("review_cnt")
            or data.get("num_reviews")
            or data.get("reviewCount")
            or data.get("rating_count")
            or data.get("total_reviews")
            or "N/A"
        )

        url = data.get("url", "#")
        image_url = (
            data.get("image_url")
            or data.get("image")
            or data.get("img_url")
            or data.get("thumbnail")
            or data.get("imageUrl")
            or ""
        )

        description = clean_description_text(
            data.get("description", "") or data.get("content", "")
        )

        price_text = normalize_price(price)
        short_desc = (
            description[:140] + ("..." if len(description) > 140 else "")
            if description else ""
        )

        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(f"**{name}**")

                st.write(f"**Price:** {price_text}")
                st.write(f"**Category:** {category}")
                st.write(f"**Rating:** {rating}")
                st.write(f"**Reviews:** {review_count}")

                if image_url:
                    st.image(image_url, width=90)

                if short_desc:
                    st.caption(short_desc)

                tags = []
                if category not in [None, "", "N/A"]:
                    tags.append(category)
                if rating not in [None, "", "N/A"]:
                    tags.append(f"⭐ {rating}")
                if price_text not in [None, "", "N/A"]:
                    tags.append(str(price_text))

                if tags:
                    st.caption(" • ".join(tags))

                if url and url != "#":
                    try:
                        st.link_button("View product", url, use_container_width=True)
                    except:
                        st.markdown(f"[View product]({url})")





if graph_import_error:
    st.error("Could not load chatbot agent.")
    st.code(graph_import_error)

left_col, right_col = st.columns([2.2, 1])

with left_col:
    st.subheader("Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None

    st.markdown('<div class="chat-shell">', unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🧁"):
            render_chat_bubble(msg["role"], msg["content"])
            if msg["role"] == "assistant" and msg.get("products"):
                render_product_cards(msg["products"])

    user_query = st.chat_input("Ask about King Arthur Baking mixes...")

    if not user_query and st.session_state.pending_query:
        user_query = st.session_state.pending_query
        st.session_state.pending_query = None

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message("user", avatar="👤"):
            render_chat_bubble("user", user_query)

        with st.chat_message("assistant", avatar="🧁"):
            with st.spinner("Thinking..."):
                if graph_import_error:
                    result = {
                        "final_answer": "The chatbot agent is not available right now.",
                        "route": "error"
                    }
                else:
                    try:
                        chat_history_text = format_chat_history(st.session_state.messages)
                        result = graph.invoke({
                            "user_query": user_query,
                            "chat_history": chat_history_text
                        })
                    except Exception as e:
                        result = {
                            "final_answer": f"Error: {str(e)}",
                            "route": "error"
                        }

                raw_final_answer = result.get("final_answer", "Sorry, I could not generate an answer.")
                retrieved_docs = result.get("retrieved_docs", [])

                final_answer = clean_assistant_answer(raw_final_answer)

                if retrieved_docs:
                    top_doc = retrieved_docs[0].metadata if hasattr(retrieved_docs[0], "metadata") else retrieved_docs[0]
                    top_name = top_doc.get("name", "this product")
                    top_price = top_doc.get("price")

                    if result.get("route") == "sql":
                        if top_price not in [None, "", "N/A"]:
                            final_answer = f'The top result I found is "{top_name}" priced at ${top_price}.'
                        else:
                            final_answer = f'The top result I found is "{top_name}".'
                    elif not final_answer or len(final_answer) > 300:
                        final_answer = f'The best match I found is "{top_name}".'

                render_chat_bubble("assistant", final_answer)
                render_product_cards(retrieved_docs)

                st.session_state.last_result = result
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_answer,
                    "products": retrieved_docs
                })

    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="side-card">', unsafe_allow_html=True)
    st.subheader("Workflow Graph")
    result = st.session_state.get("last_result", None)

    if result:
        route = result.get("route")
        out_of_scope = result.get("out_of_scope", False)
        rejected = out_of_scope or route == "reject"
        st.graphviz_chart(build_workflow_graph(selected_route=route, rejected=rejected))
    else:
        st.graphviz_chart(build_workflow_graph())

    st.markdown('</div>', unsafe_allow_html=True)

st.divider()