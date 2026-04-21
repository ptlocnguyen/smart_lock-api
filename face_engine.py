import requests
import numpy as np
from config import BUFFALO_API_URL

def get_embedding(image_bytes):
    try:
        resp = requests.post(
            BUFFALO_API_URL,
            files={"file": ("image.jpg", image_bytes, "image/jpeg")},
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            emb = data.get("embedding")

            if emb and len(emb) == 512:
                return emb

    except Exception as e:
        print("Embedding error:", e)

    return None


def cosine_similarity(a, b):
    if len(a) != len(b):
        return 0.0

    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)
