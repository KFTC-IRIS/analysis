import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = 'D:\\GoogleDrive\\내 드라이브\\4_BIS\\Analytics Challenge 2025\\laundering.csv'
data = pd.read_csv(file_path)

data['date'] = pd.to_datetime({'year': 2025, 'month': data['month'], 'day': data['day']})
date_differences = data.groupby('laundering_schema_id')['date'].agg(['min', 'max'])
date_differences['days_difference'] = (date_differences['max'] - date_differences['min']).dt.days

# 일수 차이별 건수 계산
days_difference_counts = date_differences['days_difference'].value_counts().sort_index()
bins = np.arange(0, date_differences['days_difference'].max() + 5, 5)

plt.figure(figsize=(12, 6))
plt.bar(days_difference_counts.index, days_difference_counts.values, color='skyblue', width=0.8)
plt.xlabel('Days Difference', fontsize=12)
plt.ylabel('Number of Laundering Schemas', fontsize=12)
plt.title('Day window statistics', fontsize=14)
plt.xticks(bins)
plt.tight_layout()
plt.savefig(f"D:\\GoogleDrive\\내 드라이브\\4_BIS\\day_window.png")

# 각 사건별 최대 금액 계산assas
max_amount_per_schema = data.groupby('laundering_schema_id')['amount'].max()

date_differences = date_differences.merge(max_amount_per_schema, left_index=True, right_index=True)
average_max_amounts = date_differences.groupby('days_difference')['amount'].min()

plt.figure(figsize=(12, 6))
plt.bar(average_max_amounts.index, average_max_amounts.values, color='skyblue', width=0.8)
plt.xlabel('Days Difference', fontsize=12)
plt.ylabel('Min of Maximum Amounts', fontsize=12)
plt.title('Average Amounts statistics', fontsize=14)
plt.xticks(bins)
plt.tight_layout()
plt.savefig(f"D:\\GoogleDrive\\내 드라이브\\4_BIS\\min_max_amount.png")
plt.close()

print("Complete to draw statistics")
