import pandas as pd
import sqlite3

parquet_file = "D:\\GoogleDrive\\내 드라이브\\4_BIS\\Analytics Challenge 2025\\train_target_dataset.parquet"
db_file = "D:\\workspace\\sqlite-tools-win-x64-3480000\\bis.db"
table_name = "laundering"

df = pd.read_parquet(parquet_file)
conn = sqlite3.connect(db_file)
df.to_sql(table_name, conn, if_exists="replace", index=False)
conn.close()

print(f"Complete to save db")
