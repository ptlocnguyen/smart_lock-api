import requests
import numpy as np
from app.db import get_connection

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
            files={"file": ("img.jpg", file_bytes)},
            timeout=5
        )
        if res.status_code == 200:
            return res.json().get("embedding")
    except:
        return None
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
