import sqlite3
import numpy as np
import os
import logging
from config import DATABASE_PATH

logger = logging.getLogger(__name__)

def get_db_connection():
    """Establish connection to SQLite database with Row factory."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    # Enable Foreign Key constraints
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    """Initialize database schema if tables do not exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Table: persons
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Table: face_embeddings
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (person_id) REFERENCES persons (id) ON DELETE CASCADE
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully at %s", DATABASE_PATH)

def add_person(name):
    """Add a new person or return existing person ID."""
    name = name.strip()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT id FROM persons WHERE LOWER(name) = LOWER(?)", (name,))
    existing = cursor.fetchone()
    if existing:
        person_id = existing['id']
    else:
        cursor.execute("INSERT INTO persons (name) VALUES (?)", (name,))
        conn.commit()
        person_id = cursor.lastrowid
        
    conn.close()
    return person_id

def add_face_embedding(person_id, embedding_np):
    """
    Store a 512-D ArcFace float32 embedding as BLOB.
    """
    if not isinstance(embedding_np, np.ndarray):
        embedding_np = np.array(embedding_np, dtype=np.float32)
        
    # Ensure standard float32 array
    embedding_np = embedding_np.astype(np.float32)
    blob_data = embedding_np.tobytes()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO face_embeddings (person_id, embedding) VALUES (?, ?)",
        (person_id, sqlite3.Binary(blob_data))
    )
    conn.commit()
    conn.close()

def get_all_embeddings():
    """
    Retrieve all face embeddings stored in the database.
    Returns list of dicts: [{'person_id': int, 'name': str, 'embedding': np.ndarray}]
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = '''
        SELECT fe.id, fe.person_id, p.name, fe.embedding 
        FROM face_embeddings fe
        JOIN persons p ON fe.person_id = p.id
    '''
    cursor.execute(query)
    rows = cursor.fetchall()
    
    results = []
    for row in rows:
        emb_bytes = row['embedding']
        emb_array = np.frombuffer(emb_bytes, dtype=np.float32)
        results.append({
            'embedding_id': row['id'],
            'person_id': row['person_id'],
            'name': row['name'],
            'embedding': emb_array
        })
        
    conn.close()
    return results

def get_all_persons_with_counts():
    """
    Retrieve all registered persons along with their stored sample count.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = '''
        SELECT p.id, p.name, p.created_at, COUNT(fe.id) as sample_count
        FROM persons p
        LEFT JOIN face_embeddings fe ON p.id = fe.person_id
        GROUP BY p.id, p.name, p.created_at
        ORDER BY p.name ASC
    '''
    cursor.execute(query)
    rows = cursor.fetchall()
    
    persons = [dict(row) for row in rows]
    conn.close()
    return persons

def delete_person(person_id):
    """Delete a registered person and all their embeddings."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM persons WHERE id = ?", (person_id,))
    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()
    print("Database init test completed successfully.")
