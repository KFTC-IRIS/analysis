import pandas as pd
import sqlite3

parquet_file = "D:\\GoogleDrive\\내 드라이브\\4_BIS\\Analytics Challenge 2025\\transaction_dataset.parquet"

df = pd.read_parquet(parquet_file)

# print(f"read completed")
# print(f"- account_id 컬럼의 고유 데이터 수: {df['account_id'].nunique()}")
#
# unique_combinations = df[['category_0', 'category_1', 'category_2']].drop_duplicates()
# combination_count = unique_combinations.shape[0]
# print(f"- category_0, category_1, category_2 조합의 고유 개수: {combination_count}")
#
# - account_id 컬럼의 고유 데이터 수: 1809646
# - category_0, category_1, category_2 조합의 고유 개수: 126

max_account_id = pd.to_numeric(df['account_id'], errors='coerce').max()
print(f"- account_id 컬럼에서 가장 큰 값: {max_account_id}")
