from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import numpy as np
import os
import json
import datetime
from databricks import sql

app = Flask(__name__)

# ===== CORS =====
CORS(app, resources={r"/*": {"origins": "*"}})

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', '*')
    response.headers.add('Access-Control-Allow-Methods', '*')
    return response

# ===== ENV =====
HOST = os.getenv("DATABRICKS_HOST")
PATH = os.getenv("DATABRICKS_HTTP_PATH")
TOKEN = os.getenv("DATABRICKS_TOKEN")
AI_URL = os.getenv("AI_URL")

print("HOST:", HOST)
print("PATH:", PATH)
print("AI:", AI_URL)

# ===== CACHE =====
faces_cache = []

# ===== DB CONNECT =====
def get_conn():
    return sql.connect(
        server_hostname=HOST,
        http_path=PATH,
        access_token=TOKEN
    )

# ===== LOAD CACHE =====
def load_faces():
    global faces_cache

    print("Loading faces from DB...")

    try:
        conn = get_conn()
        cur = conn.cursor()

        cur.execute("SELECT name, embedding FROM face_db.faces")

        faces_cache = []

        for row in cur.fetchall():
            try:
                faces_cache.append((row[0], json.loads(row[1])))
            except:
                continue

        cur.close()
        conn.close()

        print("Loaded:", len(faces_cache), "faces")

    except Exception as e:
        print("LOAD ERROR:", e)

# load 1 lần khi start
load_faces()

# ===== AI CALL =====
def get_embedding(image_bytes):

    files = {
        "file": ("img.jpg", image_bytes, "image/jpeg")
    }

    try:
        res = requests.post(AI_URL, files=files, timeout=10)

        if res.status_code != 200:
            print("AI ERROR:", res.text)
            return None

        return res.json().get("embedding")

    except Exception as e:
        print("AI FAIL:", e)
        return None

# ===== COSINE =====
def cosine(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    if len(a) != len(b):
        return 0

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ===== MATCH =====
def match_face(emb):

    global faces_cache

    best_name = "Unknown"
    best_score = 0

    for name, emb_db in faces_cache:

        score = cosine(emb, emb_db)

        if score > best_score:
            best_score = score
            best_name = name

    print("Best:", best_name, best_score)

    if best_score > 0.5:
        return best_name

    return "Unknown"

# ================= REGISTER =================
@app.route("/register", methods=["POST"])
def register():

    print("CALL /register")

    name = request.form.get("name")
    file = request.files.get("file")

    if not name or not file:
        return "Missing data", 400

    emb = get_embedding(file.read())

    if emb is None:
        return "No face", 400

    try:
        conn = get_conn()
        cur = conn.cursor()

        cur.execute(
            "INSERT INTO face_db.faces VALUES (?, ?)",
            (name, json.dumps(emb))
        )

        cur.close()
        conn.close()

        # update cache
        faces_cache.append((name, emb))

        return "Saved"

    except Exception as e:
        print("REGISTER ERROR:", e)
        return "DB Error", 500

# ================= RECOGNIZE (WEB) =================
@app.route("/recognize_image", methods=["POST"])
def recognize_image():

    print("CALL /recognize_image")

    file = request.files.get("file")

    emb = get_embedding(file.read())

    if emb is None:
        return jsonify({"name": "No face"})

    name = match_face(emb)

    if name != "Unknown":

        conn = get_conn()
        cur = conn.cursor()
    
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
        cur.execute(
            "INSERT INTO face_db.logs VALUES (?, ?)",
            (name, now)
        )
    
        cur.close()
        conn.close()

    return jsonify({"name": name})

# ================= RECOGNIZE (ESP32) =================
@app.route("/recognize", methods=["POST"])
def recognize():

    print("CALL /recognize")

    file = request.files.get("file")

    emb = get_embedding(file.read())

    if emb is None:
        return jsonify({"name": "No face"})

    name = match_face(emb)

    # lưu log nếu nhận diện được
    if name != "Unknown":
        try:
            conn = get_conn()
            cur = conn.cursor()

            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            cur.execute(
                "INSERT INTO face_db.logs VALUES (?, ?)",
                (name, now)
            )

            cur.close()
            conn.close()

        except Exception as e:
            print("LOG ERROR:", e)

    return jsonify({"name": name})

# ================= LOGS =================
@app.route("/logs")
def logs():

    print("CALL /logs")

    try:
        conn = get_conn()
        cur = conn.cursor()

        cur.execute("SELECT name, time FROM face_db.logs LIMIT 20")

        data = []

        for row in cur.fetchall():
            data.append({
                "name": row[0],
                "time": row[1]
            })

        cur.close()
        conn.close()

        return jsonify(data)

    except Exception as e:
        print("LOGS ERROR:", e)
        return jsonify([])

# ================= RELOAD CACHE =================
@app.route("/reload")
def reload_cache():

    load_faces()
    return "Reloaded"

# ================= HOME =================
@app.route("/")
def home():
    return "API OK"
