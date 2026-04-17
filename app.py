from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import numpy as np
import os
import time
from databricks import sql

app = Flask(__name__)
CORS(app)

# ================= ENV =================
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
AI_URL = os.getenv("AI_URL")

# ================= DEBUG ENV =================
print("HOST:", DATABRICKS_HOST)
print("HTTP PATH:", DATABRICKS_HTTP_PATH)
print("TOKEN:", str(DATABRICKS_TOKEN)[:5] if DATABRICKS_TOKEN else None)
print("AI URL:", AI_URL)

# ================= DB CONNECT =================
def get_conn():
    return sql.connect(
        server_hostname=DATABRICKS_HOST,
        http_path=DATABRICKS_HTTP_PATH,
        access_token=DATABRICKS_TOKEN
    )

# ================= AI CALL (RETRY) =================
def get_embedding(image_bytes):

    files = {
        "file": ("img.jpg", image_bytes, "image/jpeg")
    }

    for i in range(3):  # retry 3 lần
        try:
            res = requests.post(AI_URL, files=files, timeout=10)

            if res.status_code == 200:
                data = res.json()
                return data.get("embedding")

        except Exception as e:
            print("AI ERROR:", e)

        time.sleep(1)

    return None

# ================= COSINE =================
def cosine(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

# ================= REGISTER =================
@app.route("/register", methods=["POST"])
def register():

    try:
        name = request.form.get("name")
        file = request.files.get("file")

        if not name or not file:
            return "Missing data", 400

        print("Register:", name)

        emb = get_embedding(file.read())

        if emb is None:
            return "No face detected", 400

        conn = get_conn()
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO face_db.faces VALUES (?, ?)",
            (name, emb)
        )

        cursor.close()
        conn.close()

        return "Saved"

    except Exception as e:
        print("REGISTER ERROR:", e)
        return "Server error", 500


# ================= RECOGNIZE =================
@app.route("/recognize", methods=["POST"])
def recognize():

    try:
        data = request.get_json()
        emb = data.get("embedding")

        if emb is None:
            return jsonify({"name": "Error"}), 400

        conn = get_conn()
        cursor = conn.cursor()

        cursor.execute("SELECT name, embedding FROM face_db.faces")

        best_name = "Unknown"
        best_score = 0

        for row in cursor.fetchall():

            name = row[0]
            emb_db = row[1]

            score = cosine(emb, emb_db)

            if score > best_score:
                best_score = score
                best_name = name

        cursor.close()
        conn.close()

        print("Best:", best_name, best_score)

        if best_score > 0.5:
            return jsonify({"name": best_name})

        return jsonify({"name": "Unknown"})

    except Exception as e:
        print("RECOGNIZE ERROR:", e)
        return jsonify({"name": "Error"}), 500


# ================= HEALTH CHECK =================
@app.route("/")
def home():
    return "Face API Running"


# ================= RUN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
