from kb.vector_store import similarity_search


def safe_float(v, default=0.0):
    try:
        return float(str(v).replace("$", "").strip())
    except:
        return default


def safe_int(v, default=0):
    try:
        return int(str(v).replace(",", "").strip())
    except:
        return default


def rank_docs(query: str, items: list[dict]) -> list[dict]:
    q = query.lower()

    if any(x in q for x in ["recommend", "good", "best", "top rated", "top-rated", "which product"]):
        return sorted(
            items,
            key=lambda d: (
                safe_float(d.get("rating", 0)),
                safe_int(d.get("review_count", 0))
            ),
            reverse=True
        )

    if any(x in q for x in ["cheap", "cheaper", "lowest price", "budget", "affordable"]):
        return sorted(
            items,
            key=lambda d: safe_float(d.get("price", 999999))
        )

    return items


def clean_page_content_description(text: str) -> str:
    if not text:
        return ""

    text = str(text).strip()

    import re
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


def rag_node(state):
    query = state["user_query"]

    docs = similarity_search(query, k=8)

    if not docs:
        return {
            "route": "rag",
            "rag_result": "No relevant product information found.",
            "retrieved_docs": []
        }

    retrieved_docs = []
    for doc in docs:
        metadata = doc.metadata or {}

        raw_description = metadata.get("description", "")
        if not raw_description:
            raw_description = clean_page_content_description(doc.page_content)

        item = {
            "id": metadata.get("id"),
            "name": metadata.get("name", "Unknown"),
            "price": metadata.get("price", ""),
            "rating": metadata.get("rating", ""),
            "review_count": metadata.get("review_count", ""),
            "category": metadata.get("category", ""),
            "url": metadata.get("url", ""),
            "image_url": metadata.get("image_url", ""),
            "description": raw_description,
            "ingredients": metadata.get("ingredients", ""),
            "size": metadata.get("size", "")
        }
        retrieved_docs.append(item)

    ranked_docs = rank_docs(query, retrieved_docs)

    doc_summaries = []
    for item in ranked_docs[:4]:
        summary = f"""
Name: {item['name']}
Price: {item['price']}
Rating: {item['rating']}
Review Count: {item['review_count']}
Category: {item['category']}
URL: {item['url']}
Image Url: {item['image_url']}
Description: {item['description']}
"""
        doc_summaries.append(summary.strip())

    rag_result = "\n\n---\n\n".join(doc_summaries)

    return {
        "route": "rag",
        "rag_result": rag_result,
        "retrieved_docs": ranked_docs
    }