import pandas as pd
import matplotlib.pyplot as plt

variable_names = ['weekday', 'payment_system',
                  'channel', 'category_0']

# file_path = 'D:\\GoogleDrive\\내 드라이브\\4_BIS\\Analytics Challenge 2025\\laundering.csv'
# data = pd.read_csv(file_path)

parquet_file = "D:\\GoogleDrive\\내 드라이브\\4_BIS\\Analytics Challenge 2025\\transaction_dataset.parquet"
data = pd.read_parquet(parquet_file)

for variable_name in variable_names:
    weekday_distribution = data[variable_name].value_counts()

    # 그래프 설정 및 출력
    plt.figure(figsize=(8, 5))
    weekday_distribution.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(variable_name, fontsize=14)
    plt.xlabel(variable_name, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"D:\\GoogleDrive\\내 드라이브\\4_BIS\\dist_origin_{variable_name}.png")
