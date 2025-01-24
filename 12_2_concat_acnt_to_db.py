import dask.dataframe as dd
import pandas as pd
import sqlite3

db_file = "D:\\workspace\\sqlite-tools-win-x64-3480000\\bis.db"
query = """
SELECT *
FROM laundering_detail
"""
parquet_file = "D:\\GoogleDrive\\내 드라이브\\4_BIS\\Analytics Challenge 2025\\account_dataset.parquet"

conn = sqlite3.connect(db_file)

laundering_data = pd.read_sql_query(query, conn)
dask_df = dd.read_parquet(parquet_file)
merged = dask_df.merge(laundering_data, on=["account_id"], how="right")

result_df = merged.compute()
result_df.to_sql("laundering_full", conn, if_exists="replace", index=False)
conn.close()

print("Complete to process join")
