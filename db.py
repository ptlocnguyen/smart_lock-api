import os
from databricks import sql

def get_connection():
    return sql.connect(
        server_hostname=os.environ["DATABRICKS_SERVER"],
        http_path=os.environ["DATABRICKS_HTTP_PATH"],
        access_token=os.environ["DATABRICKS_TOKEN"]
    )

# =============================
# LOAD EMBEDDINGS
# =============================
def load_embeddings():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT user_id, embedding FROM smart_door.face_embeddings")
    rows = cursor.fetchall()

    data = []
    for r in rows:
        data.append({
            "user_id": r[0],
            "embedding": r[1]
        })

    cursor.close()
    conn.close()
    return data

# =============================
# UPDATE EMBEDDING (AVG)
# =============================
def update_embedding(user_id, new_emb):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(f"""
        SELECT embedding FROM smart_door.face_embeddings 
        WHERE user_id = '{user_id}'
    """)
    row = cursor.fetchone()

    # Convert new_emb về float
    new_emb = [float(x) for x in new_emb]

    if row:
        # Convert old embedding về float
        old_emb = [float(x) for x in row[0]]

        # Tính trung bình
        avg = [(o + n) / 2 for o, n in zip(old_emb, new_emb)]

        # Convert sang string để insert SQL
        avg_str = ",".join([str(x) for x in avg])

        cursor.execute(f"""
            UPDATE smart_door.face_embeddings
            SET embedding = array({avg_str}),
                updated_at = from_utc_timestamp(current_timestamp(), 'Asia/Ho_Chi_Minh')
            WHERE user_id = '{user_id}'
        """)
    else:
        new_str = ",".join([str(x) for x in new_emb])

        cursor.execute(f"""
            INSERT INTO smart_door.face_embeddings VALUES (
                '{user_id}',
                array({new_str}),
                from_utc_timestamp(current_timestamp(), 'Asia/Ho_Chi_Minh')
            )
        """)

    cursor.close()
    conn.close()

# =============================
# INSERT LOG
# =============================
import uuid

def insert_log(user_id, method, status, device_id):
    conn = get_connection()
    cursor = conn.cursor()

    log_id = str(uuid.uuid4())

    cursor.execute(f"""
        INSERT INTO smart_door.access_logs VALUES (
            '{log_id}',
            '{user_id}',
            '{method}',
            '{status}',
            '{device_id}',
            from_utc_timestamp(current_timestamp(), 'Asia/Ho_Chi_Minh')
        )
    """)

    cursor.close()
    conn.close()