import sqlite3
import pandas as pd

# Paths to the CSV files
csv_files = {
    "fundamentals": "data/fundamentals.csv",
    "prices": "data/prices.csv",
    "prices_split_adjusted": "data/prices-split-adjusted.csv",
    "securities": "data/securities.csv",
}

# Create or connect to SQLite database
db_name = "stock_data.db"
connection = sqlite3.connect(db_name)
cursor = connection.cursor()

# Function to load CSV files into SQLite
def load_csv_to_sqlite(file_path, table_name, conn):
    # Load CSV into DataFrame
    df = pd.read_csv(file_path)
    # Write DataFrame to SQLite table
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    print(f"Table '{table_name}' has been created/updated.")

# Load each CSV into the database
for table_name, file_path in csv_files.items():
    load_csv_to_sqlite(file_path, table_name, connection)

# Commit changes and close the connection
connection.commit()
connection.close()
print(f"All tables loaded into '{db_name}' successfully.")
