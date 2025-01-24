import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = 'D:\\GoogleDrive\\내 드라이브\\4_BIS\\Analytics Challenge 2025\\laundering.csv'
data = pd.read_csv(file_path)

# 동일한 사건에서 가장 많이 나타난 account_id 선택
most_common_account_id = (
    data.groupby(['laundering_schema_id', 'account_id'])
    .size()
    .reset_index(name='count')
    .sort_values(['laundering_schema_id', 'count'], ascending=[True, False])
    .drop_duplicates('laundering_schema_id')
)

# 가장 많이 나타난 account_id만 필터링
filtered_data = data.merge(
    most_common_account_id[['laundering_schema_id', 'account_id']],
    on=['laundering_schema_id', 'account_id']
)

filtered_data['date'] = pd.to_datetime({'year': 2025, 'month': filtered_data['month'], 'day': filtered_data['day']})
date_differences = filtered_data.groupby('laundering_schema_id')['date'].agg(['min', 'max'])
date_differences['days_difference'] = (date_differences['max'] - date_differences['min']).dt.days

# 일수 차이별 건수 계산
days_difference_counts = date_differences['days_difference'].value_counts().sort_index()
bins = np.arange(0, date_differences['days_difference'].max() + 5, 5)

plt.figure(figsize=(12, 6))
plt.bar(days_difference_counts.index, days_difference_counts.values, color='skyblue', width=0.8)
plt.xlabel('Days Difference', fontsize=12)
plt.ylabel('Number of Laundering Schemas by center node', fontsize=12)
plt.title('Day window statistics', fontsize=14)
plt.xticks(bins)
plt.tight_layout()
plt.savefig(f"D:\\GoogleDrive\\내 드라이브\\4_BIS\\day_window_by_center.png")

# 각 사건별 최대 금액 계산
max_amount_per_schema = filtered_data.groupby('laundering_schema_id')['amount'].max()

date_differences = date_differences.merge(max_amount_per_schema, left_index=True, right_index=True)
average_max_amounts = date_differences.groupby('days_difference')['amount'].min()

plt.figure(figsize=(12, 6))
plt.bar(average_max_amounts.index, average_max_amounts.values, color='skyblue', width=0.8)
plt.xlabel('Days Difference', fontsize=12)
plt.ylabel('Min of Maximum Amounts', fontsize=12)
plt.title('Average Amounts statistics by center node', fontsize=14)
plt.xticks(bins)
plt.tight_layout()
plt.savefig(f"D:\\GoogleDrive\\내 드라이브\\4_BIS\\min_max_amount_by_center.png")
plt.close()

print("Complete to draw statistics")
