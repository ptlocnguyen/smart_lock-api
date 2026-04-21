from fastapi import FastAPI, UploadFile, File
import time

from face_engine import get_embedding, cosine_similarity
from cache import get_cache, refresh_cache
from config import SIM_THRESHOLD
from db import insert_log

app = FastAPI()


# =====================
# STARTUP
# =====================
@app.on_event("startup")
def startup():
    refresh_cache()


# =====================
# HEALTH
# =====================
@app.get("/")
def health():
    return {"status": "ok"}


# =====================
# RECOGNIZE FACE (ESP32 dùng)
# =====================
@app.post("/recognize")
async def recognize(file: UploadFile = File(...), device_code: str = "esp32"):

    start = time.time()

    image = await file.read()

    embedding = get_embedding(image)

    if embedding is None:
        return {"success": False, "message": "no_face"}

    cache = get_cache()

    best_user = None
    best_score = 0

    for user in cache:
        score = cosine_similarity(embedding, user["embedding"])
        if score > best_score:
            best_score = score
            best_user = user

    if best_score >= SIM_THRESHOLD:
        result = {
            "success": True,
            "user_code": best_user["user_code"],
            "full_name": best_user["full_name"],
            "similarity": best_score
        }

        insert_log({
            "user_id": best_user["user_id"],
            "user_code": best_user["user_code"],
            "full_name": best_user["full_name"],
            "method": "face",
            "channel": "esp32",
            "result": "success",
            "device_code": device_code,
            "similarity": best_score
        })

    else:
        result = {"success": False, "message": "unknown"}

    result["process_time"] = time.time() - start

    return result


# =====================
# FORCE REFRESH CACHE
# =====================
@app.post("/refresh-cache")
def refresh():
    refresh_cache()
    return {"status": "refreshed"}
