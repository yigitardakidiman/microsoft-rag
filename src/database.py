import sqlite3
import json
import os

# Veritabanını proje kök dizininde rag_database.db olarak tutuyoruz
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_NAME = os.path.join(BASE_DIR, "rag_database.db")

def get_connection():
    return sqlite3.connect(DB_NAME)

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            embedding TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'Bilinmiyor'
        )
    ''')
    conn.commit()
    conn.close()

def reset_db():
    """Veritabanındaki documents tablosunu siler ve yeniden oluşturur."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS documents')
    conn.commit()
    conn.close()
    init_db()
    print("Veritabanı sıfırlandı ve yeniden oluşturuldu.")

def insert_chunk(content, embedding_vector, source="Bilinmiyor"):
    conn = get_connection()
    cursor = conn.cursor()
    # Save embedding as a JSON string
    embedding_json = json.dumps(embedding_vector)
    cursor.execute('INSERT INTO documents (content, embedding, source) VALUES (?, ?, ?)', (content, embedding_json, source))
    conn.commit()
    conn.close()

def get_all_chunks():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT content, embedding, source FROM documents')
    rows = cursor.fetchall()
    conn.close()
    
    # Parse JSON embeddings back to lists
    results = []
    for row in rows:
        content = row[0]
        embedding = json.loads(row[1])
        source = row[2] if len(row) > 2 else "Bilinmiyor"
        results.append({"content": content, "embedding": embedding, "source": source})
    
    return results

if __name__ == "__main__":
    init_db()
    print("Veritabanı (SQLite) başarıyla oluşturuldu.")
