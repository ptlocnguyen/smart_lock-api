import os
from dotenv import load_dotenv
from databricks import sql

load_dotenv()

def get_connection():
    host = os.getenv("DB_HOST")
    path = os.getenv("DB_HTTP_PATH")
    token = os.getenv("DB_TOKEN")

    if not host or not path or not token:
        raise Exception("Thiếu biến môi trường Databricks")

    conn = sql.connect(
        server_hostname=host,
        http_path=path,
        access_token=token
    )

    conn.autocommit = True
    return conn