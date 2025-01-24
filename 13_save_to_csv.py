import pandas as pd
import sqlite3

db_file = "D:\\workspace\\sqlite-tools-win-x64-3480000\\bis.db"
# query = "SELECT * FROM laundering_detail"
query = "SELECT * FROM laundering_full"
csv_file = "D:\\GoogleDrive\\내 드라이브\\4_BIS\\Analytics Challenge 2025\\laundering.csv"

conn = sqlite3.connect(db_file)
df = pd.read_sql_query(query, conn)
df.to_csv(csv_file, index=False)
conn.close()

print(f"Complete to save csv")
