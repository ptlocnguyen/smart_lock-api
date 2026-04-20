from app.db import get_connection


def sync_fingerprint(user_id: str, fingerprint_id: int, device_id: str):
    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM fingerprints WHERE fingerprint_id = ? AND device_id = ?",
            (fingerprint_id, device_id)
        )

        if cursor.fetchone():
            return {"status": "exists"}

        cursor.execute(
            "INSERT INTO fingerprints VALUES (?, ?, ?, current_timestamp())",
            (fingerprint_id, user_id, device_id)
        )

    return {"status": "ok"}


def verify_fingerprint(fingerprint_id: int, device_id: str):
    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            "SELECT user_id FROM fingerprints WHERE fingerprint_id = ? AND device_id = ?",
            (fingerprint_id, device_id)
        )

        row = cursor.fetchone()

        if row:
            return row[0], True

    return None, False