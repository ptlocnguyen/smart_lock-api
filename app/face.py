import requests
import numpy as np
from app.db import get_connection
import os

API_URL = os.getenv("AI_URL")


def normalize(v):
    v = np.array(v)
    return v / np.linalg.norm(v)


def cosine(a, b):
    a = normalize(a)
    b = normalize(b)
    return float(np.dot(a, b))


def get_embedding(file_bytes):
    try:
        res = requests.post(
            API_URL,
            files={
                "file": ("image.jpg", file_bytes, "image/jpeg")
            },
            timeout=5
        )

        if res.status_code != 200:
            print("API ERROR:", res.text)
            return None

        data = res.json()

        if "embedding" not in data:
            print("NO EMBEDDING:", data)
            return None

        emb = data["embedding"]

        if not emb or len(emb) == 0:
            print("EMPTY EMBEDDING")
            return None

        print("Embedding OK:", len(emb))
        return emb

    except Exception as e:
        print("EXCEPTION:", e)
        return None


def recognize_face(file_bytes):
    emb = get_embedding(file_bytes)
    if emb is None:
        return None, 0

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, avg_embedding FROM face_user_vector")

        best_user = None
        best_score = 0

        for row in cursor.fetchall():
            score = cosine(emb, row[1])
            if score > best_score:
                best_score = score
                best_user = row[0]

    return best_user, best_score
