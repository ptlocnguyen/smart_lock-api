from fastapi import FastAPI, UploadFile, File, Form
import numpy as np
from PIL import Image
import io
import os
import threading
from db import insert_log
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # cho phép tất cả (dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# GLOBAL STATE
# =============================
MODEL_DIR = "/tmp/insightface"
os.makedirs(MODEL_DIR, exist_ok=True)

model = None
embeddings_cache = None

# lock chống load model nhiều lần
init_lock = threading.Lock()

# =============================
# INIT SYSTEM (LAZY LOAD)
# =============================
def init_system():
    global model, embeddings_cache

    if model is not None:
        return

    with init_lock:
        if model is not None:
            return

        print("INIT SYSTEM...")

        import insightface
        from db import load_embeddings

        # load model
        m = insightface.app.FaceAnalysis(
            name="buffalo_l",
            root=MODEL_DIR
        )
        m.prepare(ctx_id=0, det_size=(320, 320))

        print("MODEL LOADED")

        # load DB
        try:
            cache = load_embeddings()
            print("DB LOADED:", len(cache))
        except Exception as e:
            print("DB ERROR:", e)
            cache = []

        model = m
        embeddings_cache = cache

        print("SYSTEM READY")

# =============================
# HEALTH CHECK
# =============================
@app.get("/")
def root():
    return {"status": "running"}

# =============================
# COSINE SIMILARITY
# =============================
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# =============================
# RECOGNIZE
# =============================
@app.post("/recognize")
async def recognize(
    file: UploadFile = File(...),
    device_id: str = Form("esp32")
):
    try:
        # nếu chưa load thì load
        init_system()

        if model is None:
            return {"status": "warming_up"}

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # resize để tăng tốc + ổn định
        # image = image.resize((240, 240))
        img = np.array(image)

        faces = model.get(img)

        if len(faces) == 0:
            insert_log("unknown", "face", "fail", device_id)
            return {"status": "no_face"}

        emb = faces[0].embedding

        best_user = None
        best_score = 0

        if embeddings_cache:
            for item in embeddings_cache:
                score = cosine_similarity(emb, item["embedding"])
                if score > best_score:
                    best_score = score
                    best_user = item["user_id"]

        print("BEST SCORE:", best_score)

        if best_score > 0.5:
            # ghi log thành công
            insert_log(best_user, "face", "success", device_id)

            return {
                "status": "success",
                "user_id": best_user,
                "score": float(best_score)
            }
        else:
            # ghi log thất bại
            insert_log("unknown", "face", "fail", device_id)

            return {"status": "unknown"}

    except Exception as e:
        return {"status": "error", "message": str(e)}

# =============================
# REGISTER
# =============================
@app.post("/register")
async def register(
    user_id: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        init_system()

        if model is None:
            return {"status": "warming_up"}

        from db import update_embedding

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img = np.array(image)

        faces = model.get(img)

        if len(faces) == 0:
            return {"status": "no_face"}

        # embedding mới
        new_emb = [float(x) for x in faces[0].embedding.tolist()]

        # update DB
        update_embedding(user_id, new_emb)

        # =============================
        # UPDATE CACHE REALTIME
        # =============================
        global embeddings_cache

        if embeddings_cache is None:
            embeddings_cache = []

        found = False

        for item in embeddings_cache:
            if item["user_id"] == user_id:
                old_emb = [float(x) for x in item["embedding"]]
                avg = [(o + n) / 2 for o, n in zip(old_emb, new_emb)]
                item["embedding"] = avg
                found = True
                break

        if not found:
            embeddings_cache.append({
                "user_id": user_id,
                "embedding": new_emb
            })

        print("CACHE SIZE:", len(embeddings_cache))

        return {"status": "registered"}

    except Exception as e:
        return {"status": "error", "message": str(e)}

# =============================
# GET FACE EMBEDDINGS (CHECK)
# =============================
@app.get("/face/all")
async def get_all_face():
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT user_id FROM smart_door.face_embeddings
        """)

        rows = cursor.fetchall()

        cursor.close()
        conn.close()

        data = [r[0] for r in rows]

        return {"data": data}

    except Exception as e:
        return {"status": "error", "message": str(e)}

# =============================
# USER MANAGEMENT APIs
# =============================

from db import get_connection

# =============================
# CREATE USER
# =============================
@app.post("/user/create")
async def create_user(user_id: str = Form(...), name: str = Form(...)):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(f"""
            INSERT INTO smart_door.users VALUES (
                '{user_id}',
                '{name}',
                from_utc_timestamp(current_timestamp(), 'Asia/Ho_Chi_Minh'),
                from_utc_timestamp(current_timestamp(), 'Asia/Ho_Chi_Minh'),
                true
            )
        """)

        cursor.close()
        conn.close()

        return {"status": "created"}

    except Exception as e:
        insert_log("unknown", "face", "fail", device_id)
        return {"status": "error", "message": str(e)}


# =============================
# UPDATE USER
# =============================
@app.put("/user/update")
async def update_user(user_id: str = Form(...), name: str = Form(...)):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(f"""
            UPDATE smart_door.users
            SET name = '{name}',
                updated_at = from_utc_timestamp(current_timestamp(), 'Asia/Ho_Chi_Minh')
            WHERE user_id = '{user_id}'
        """)

        cursor.close()
        conn.close()

        return {"status": "updated"}

    except Exception as e:
        return {"status": "error", "message": str(e)}


# =============================
# DELETE USER (CASCADE)
# =============================
@app.delete("/user/delete")
async def delete_user(user_id: str = Form(...)):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # xóa embedding
        cursor.execute(f"""
            DELETE FROM smart_door.face_embeddings
            WHERE user_id = '{user_id}'
        """)

        # xóa fingerprint
        cursor.execute(f"""
            DELETE FROM smart_door.fingerprints
            WHERE user_id = '{user_id}'
        """)

        # xóa logs
        cursor.execute(f"""
            DELETE FROM smart_door.access_logs
            WHERE user_id = '{user_id}'
        """)

        # xóa user
        cursor.execute(f"""
            DELETE FROM smart_door.users
            WHERE user_id = '{user_id}'
        """)

        cursor.close()
        conn.close()

        # ===== update cache =====
        global embeddings_cache
        if embeddings_cache:
            embeddings_cache = [
                x for x in embeddings_cache if x["user_id"] != user_id
            ]

        return {"status": "deleted"}

    except Exception as e:
        return {"status": "error", "message": str(e)}


# =============================
# DELETE EMBEDDING ONLY
# =============================
@app.delete("/user/delete-embedding")
async def delete_embedding(user_id: str = Form(...)):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(f"""
            DELETE FROM smart_door.face_embeddings
            WHERE user_id = '{user_id}'
        """)

        cursor.close()
        conn.close()

        # update cache
        global embeddings_cache
        if embeddings_cache:
            embeddings_cache = [
                x for x in embeddings_cache if x["user_id"] != user_id
            ]

        return {"status": "embedding_deleted"}

    except Exception as e:
        return {"status": "error", "message": str(e)}


# =============================
# GET USERS
# =============================
@app.get("/users")
async def get_users():
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT user_id, name, created_at, is_active
            FROM smart_door.users
        """)

        rows = cursor.fetchall()

        cursor.close()
        conn.close()

        data = []
        for r in rows:
            data.append({
                "user_id": r[0],
                "name": r[1],
                "created_at": str(r[2]),
                "is_active": r[3]
            })

        return {"users": data}

    except Exception as e:
        return {"status": "error", "message": str(e)}


# =============================
# GET ACCESS LOGS
# =============================
@app.get("/logs")
async def get_logs(limit: int = 20):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT user_id, method, status, device_id, created_at
            FROM smart_door.access_logs
            ORDER BY created_at DESC
            LIMIT {limit}
        """)

        rows = cursor.fetchall()

        cursor.close()
        conn.close()

        data = []
        for r in rows:
            data.append({
                "user_id": r[0],
                "method": r[1],
                "status": r[2],
                "device_id": r[3],
                "time": str(r[4])
            })

        return {"logs": data}

    except Exception as e:
        return {"status": "error", "message": str(e)}



# =============================
# FINGERPRINT APIs
# =============================

# =============================
# REGISTER FINGERPRINT (ONLINE)
# =============================
@app.post("/fingerprint/register")
async def register_fingerprint(
    user_id: str = Form(...),
    fingerprint_id: int = Form(...)
):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # nếu đã có thì update
        cursor.execute(f"""
            DELETE FROM smart_door.fingerprints
            WHERE user_id = '{user_id}'
        """)

        cursor.execute(f"""
            INSERT INTO smart_door.fingerprints VALUES (
                '{user_id}',
                {fingerprint_id},
                from_utc_timestamp(current_timestamp(), 'Asia/Ho_Chi_Minh')
            )
        """)

        cursor.close()
        conn.close()

        return {"status": "fingerprint_registered"}

    except Exception as e:
        return {"status": "error", "message": str(e)}


# =============================
# SYNC OFFLINE FINGERPRINTS
# =============================
@app.post("/fingerprint/sync")
async def sync_fingerprint(data: list):
    """
    ESP32 gửi danh sách:
    [
        {"user_id": "...", "fingerprint_id": 1},
        {"user_id": "...", "fingerprint_id": 2}
    ]
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()

        for item in data:
            user_id = item["user_id"]
            fid = item["fingerprint_id"]

            # xóa cũ
            cursor.execute(f"""
                DELETE FROM smart_door.fingerprints
                WHERE user_id = '{user_id}'
            """)

            # insert mới
            cursor.execute(f"""
                INSERT INTO smart_door.fingerprints VALUES (
                    '{user_id}',
                    {fid},
                    from_utc_timestamp(current_timestamp(), 'Asia/Ho_Chi_Minh')
                )
            """)

        cursor.close()
        conn.close()

        return {"status": "synced"}

    except Exception as e:
        return {"status": "error", "message": str(e)}


# =============================
# GET ALL FINGERPRINTS
# =============================
@app.get("/fingerprint/all")
async def get_all_fingerprint():
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT user_id, fingerprint_id
            FROM smart_door.fingerprints
        """)

        rows = cursor.fetchall()

        cursor.close()
        conn.close()

        data = []
        for r in rows:
            data.append({
                "user_id": r[0],
                "fingerprint_id": r[1]
            })

        return {"data": data}

    except Exception as e:
        return {"status": "error", "message": str(e)}


# =============================
# DELETE FINGERPRINT
# =============================
@app.delete("/fingerprint/delete")
async def delete_fingerprint(user_id: str = Form(...)):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(f"""
            DELETE FROM smart_door.fingerprints
            WHERE user_id = '{user_id}'
        """)

        cursor.close()
        conn.close()

        return {"status": "deleted"}

    except Exception as e:
        return {"status": "error", "message": str(e)}