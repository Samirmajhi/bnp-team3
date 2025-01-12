from fastapi import FastAPI, HTTPException
import sqlite3

app = FastAPI()

# Database connection function
def get_db_connection():
    conn = sqlite3.connect("stock_data.db")
    conn.row_factory = sqlite3.Row  # Allows fetching rows as dictionaries
    return conn

@app.get("/")
def root():
    return {"message": "Welcome to the Financial Data API"}

@app.get("/data/{table_name}")
def get_table_data(table_name: str, limit: int = 10):
    conn = get_db_connection()
    try:
        query = f"SELECT * FROM {table_name} LIMIT ?"
        cursor = conn.execute(query, (limit,))
        rows = cursor.fetchall()
        conn.close()
        if not rows:
            raise HTTPException(status_code=404, detail="Table or data not found")
        return {"data": [dict(row) for row in rows]}
    except sqlite3.Error as e:
        raise HTTPException(status_code=400, detail=str(e))
