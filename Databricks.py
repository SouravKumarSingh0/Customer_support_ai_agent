import os
from databricks import sql
from dotenv import load_dotenv

load_dotenv()

host = os.getenv("DATABRICKS_HOST")
token = os.getenv("DATABRICKS_TOKEN")

print(f"DEBUG: Host from .env = {host}")
print(f"DEBUG: Token from .env = {'Token Found' if token else 'Token NOT FOUND'}")

if not host:
    raise ValueError("DATABRICKS_HOST not found in environment variables. Check your .env file.")

con = sql.connect(
    server_hostname=host.replace("https://",""),
    http_path=os.getenv("DATABRICKS_HTTP_PATH"),
    access_token=token,
)

with con.cursor() as cur:
    # basic connection test
    cur.execute("SELECT current_date()")
    result = cur.fetchall()
    print("✅ Databricks connected, current_date =", result)

    # list tables (limit 20)
    print("\n📋 Listing up to 20 tables:")
    cur.execute("SHOW TABLES")
    tables = cur.fetchall()
    for row in tables[:20]:
        print(row)

con.close()
