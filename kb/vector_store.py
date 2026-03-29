import os
import time
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from kb.documents import build_documents
from utils.config import VECTOR_STORE_DIR, OPENAI_EMBED_MODEL

load_dotenv()


def ensure_vector_dir():
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)


def get_embeddings():
    return OpenAIEmbeddings(
        model=OPENAI_EMBED_MODEL,
        check_embedding_ctx_length=False,
        tiktoken_enabled=False,
        request_timeout=120,
        max_retries=5,
    )


def build_faiss_index():
    ensure_vector_dir()
    documents = build_documents()

    if not documents:
        raise ValueError("No documents returned by build_documents().")

    embeddings = get_embeddings()

    batch_size = 20
    vectorstore = None

    for start in range(0, len(documents), batch_size):
        batch = documents[start:start + batch_size]
        print(f"Processing batch {start + 1}-{start + len(batch)} of {len(documents)}")

        success = False
        for attempt in range(3):
            try:
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(batch, embeddings)
                else:
                    vectorstore.add_documents(batch)
                success = True
                break
            except Exception as e:
                print(f"Batch failed on attempt {attempt + 1}/3: {e}")
                time.sleep(5)

        if not success:
            raise RuntimeError(f"Failed to process batch starting at document {start + 1}")

    vectorstore.save_local(VECTOR_STORE_DIR)
    return len(documents)


def load_vector_store():
    index_file = os.path.join(VECTOR_STORE_DIR, "index.faiss")
    pkl_file = os.path.join(VECTOR_STORE_DIR, "index.pkl")

    if not os.path.exists(index_file) or not os.path.exists(pkl_file):
        raise FileNotFoundError(
            f"Vector store not found in '{VECTOR_STORE_DIR}'. "
            f"Run: python build_kb.py"
        )

    embeddings = get_embeddings()
    return FAISS.load_local(
        VECTOR_STORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )


def similarity_search(query: str, k: int = 4):
    vector_store = load_vector_store()
    return vector_store.similarity_search(query, k=k)


if __name__ == "__main__":
    total = build_faiss_index()
    print(f"FAISS index built successfully with {total} documents.")