from flask import Flask, request, jsonify
import requests
import numpy as np

app = Flask(__name__)

# ===== DATABASE giả lập (sau thay Databricks)
faces_db = []

# ===== AI CALL =====
def get_embedding(image_bytes):
    url = "https://bufalo-api-973102760389.asia-southeast1.run.app/predict"

    files = {
        "file": ("img.jpg", image_bytes, "image/jpeg")
    }

    res = requests.post(url, files=files)

    if res.status_code != 200:
        return None

    data = res.json()
    return data.get("embedding")


# ===== COSINE =====
def cosine(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))


# ===== REGISTER =====
@app.route("/register", methods=["POST"])
def register():

    name = request.form.get("name")
    file = request.files.get("file")

    emb = get_embedding(file.read())

    if emb is None:
        return "No face", 400

    faces_db.append({
        "name": name,
        "embedding": emb
    })

    return "Saved"


# ===== RECOGNIZE =====
@app.route("/recognize", methods=["POST"])
def recognize():

    emb = request.json.get("embedding")

    best_name = "Unknown"
    best_score = 0

    for f in faces_db:

        score = cosine(emb, f["embedding"])

        if score > best_score:
            best_score = score
            best_name = f["name"]

    if best_score > 0.5:
        return jsonify({"name": best_name})

    return jsonify({"name": "Unknown"})


# ===== RUN =====
if __name__ == "__main__":
    app.run()
