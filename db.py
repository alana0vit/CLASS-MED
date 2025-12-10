# db.py
import sqlite3
import numpy as np
import os

def init_db(db_path):
    db_path = str(db_path)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT UNIQUE,
        embedding BLOB
    )
    """)
    conn.commit()
    conn.close()

def insert_image_record(db_path, image_path, embedding):
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    # store embedding as bytes
    emb_bytes = embedding.tobytes()
    c.execute("INSERT OR IGNORE INTO images (path, embedding) VALUES (?, ?)", (image_path, emb_bytes))
    conn.commit()
    conn.close()

def load_all_embeddings(db_path):
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    rows = c.execute("SELECT id, path, embedding FROM images").fetchall()
    conn.close()
    ids, paths, embs = [], [], []
    for r in rows:
        ids.append(r[0])
        paths.append(r[1])
        emb = np.frombuffer(r[2], dtype=np.float32)
        embs.append(emb)
    return ids, paths, np.vstack(embs) if embs else ([], [], None)