import sqlite3

def get_db_connection(db_name="stock_data.db"):
    return sqlite3.connect(db_name)
