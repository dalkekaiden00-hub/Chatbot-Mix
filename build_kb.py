from kb.sqlite_loader import build_sqlite_db
from kb.vector_store import build_faiss_index


def main():
    print("Building SQLite database...")
    total_sql = build_sqlite_db()
    print(f"SQLite database ready with {total_sql} products.\n")

    print("Building FAISS vector index...")
    total_docs = build_faiss_index()
    print(f"FAISS vector store ready with {total_docs} documents.\n")

    print("Knowledge base build complete.")


if __name__ == "__main__":
    main()