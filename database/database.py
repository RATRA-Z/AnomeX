import pandas as pd
import sqlite3

# -----------------------------
# 1. File paths
# -----------------------------

CSV_PATH = r"C:\Users\pc\AnomeX\data\anomex_dataset.csv"
DB_PATH = r"C:\Users\pc\AnomeX\database\anomex.db"


# -----------------------------
# 2. Load CSV using Pandas
# -----------------------------

df = pd.read_csv(CSV_PATH)

print("CSV loaded successfully!")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# -----------------------------
# 3. Connect to SQLite
# -----------------------------

connection = sqlite3.connect(DB_PATH)

print("Connected to SQLite database!")


# -----------------------------
# 4. Create/replace table
# -----------------------------

df.to_sql(
    "components",
    connection,
    if_exists="replace",
    index=False
)

print("Data transferred to 'components' table!")


# -----------------------------
# 5. Verify the database
# -----------------------------

cursor = connection.cursor()

cursor.execute("SELECT COUNT(*) FROM components")

row_count = cursor.fetchone()[0]

print("Rows in database:", row_count)


# -----------------------------
# 6. Show first 5 records
# -----------------------------

cursor.execute("SELECT * FROM components LIMIT 5")

rows = cursor.fetchall()

print("\nFirst 5 records:")

for row in rows:
    print(row)


# -----------------------------
# 7. Close database
# -----------------------------

connection.close()

print("\nDatabase connection closed.")
print("SQLite setup completed successfully!")