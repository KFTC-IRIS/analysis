import json
import pandas as pd
from sentence_transformers import SentenceTransformer

# 모델 로드
model = SentenceTransformer("all-MiniLM-L6-v2")

# 1. 텍스트 데이터와 임베딩 생성
file_path = "D:\\GoogleDrive\\내 드라이브\\4_BIS\\Analytics Challenge 2025\\Transaction categories.xlsx"
sheet_name = "Sheet1"
df = pd.read_excel(file_path, sheet_name=sheet_name)

df['combined_text'] = df['level_0'] + " " + df['level_1'] + " " + df['level_2']

model = SentenceTransformer("all-MiniLM-L6-v2")
df['embeddings'] = df['combined_text'].apply(lambda x: model.encode(x).tolist())
embedding_dict = dict(zip(df['combined_text'], df['embeddings']))

# 2. JSON 파일로 저장
output_path = "D:\\GoogleDrive\\내 드라이브\\4_BIS\\Analytics Challenge 2025\\embeddings.json"
with open(output_path, "w") as f:
    json.dump(embedding_dict, f)

print(f"Finished to embed categories")
