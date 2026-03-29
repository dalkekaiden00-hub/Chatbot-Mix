import os
import json
import sqlite3
from utils.config import INPUT_JSON, SQLITE_DB_PATH


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    url TEXT UNIQUE,
    price REAL,
    description TEXT,
    ingredients TEXT,
    size TEXT,
    rating REAL,
    review_count INTEGER,
    image_url TEXT,
    category TEXT
);
"""


INSERT_SQL = """
INSERT OR REPLACE INTO products (
    name, url, price, description, ingredients, size, rating, review_count, image_url, category
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""


def ensure_db_dir():
    os.makedirs(os.path.dirname(SQLITE_DB_PATH), exist_ok=True)


def load_clean_json():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def create_connection():
    ensure_db_dir()
    conn = sqlite3.connect(SQLITE_DB_PATH)
    return conn


def create_table(conn):
    conn.execute(CREATE_TABLE_SQL)
    conn.commit()


def insert_products(conn, products):
    rows = []
    for p in products:
        rows.append((
            p.get("name"),
            p.get("url"),
            p.get("price"),
            p.get("description"),
            p.get("ingredients"),
            p.get("size"),
            p.get("rating"),
            p.get("review_count"),
            p.get("image_url"),
            p.get("subcategory", "mixes"),
        ))

    conn.executemany(INSERT_SQL, rows)
    conn.commit()


def count_products(conn):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM products;")
    return cur.fetchone()[0]


def build_sqlite_db():
    products = load_clean_json()
    conn = create_connection()
    create_table(conn)
    insert_products(conn, products)
    total = count_products(conn)
    conn.close()
    return total


if __name__ == "__main__":
    total = build_sqlite_db()
    print(f"SQLite DB created successfully with {total} products.")