import json
from langchain_core.documents import Document
from utils.config import INPUT_JSON


def load_clean_products():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def product_to_text(product: dict) -> str:
    return f"""
Product Name: {product.get('name', '')}
Category: {product.get('subcategory', 'mixes')}
Price: {product.get('price', '')}
Description: {product.get('description', '')}
Ingredients: {product.get('ingredients', '')}
Size: {product.get('size', '')}
Rating: {product.get('rating', '')}
Review Count: {product.get('review_count', '')}
URL: {product.get('url', '')}
""".strip()


def build_documents():
    products = load_clean_products()
    docs = []

    for p in products:
        text = product_to_text(p)
        metadata = {
            "name": p.get("name"),
            "url": p.get("url"),
            "price": p.get("price"),
            "rating": p.get("rating"),
            "category": p.get("category", "mixes"),
            "size": p.get("size"),
        }
        docs.append(Document(page_content=text, metadata=metadata))

    return docs


if __name__ == "__main__":
    documents = build_documents()
    print(f"Built {len(documents)} documents.")
    if documents:
        print("\nSample document:\n")
        print(documents[0].page_content)
        print("\nMetadata:\n", documents[0].metadata)