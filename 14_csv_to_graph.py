import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

def draw_graph(scheme_id, group_df):
    node_attributes = {}
    for _, row in group_df.iterrows():
        account_id = str(row['account_id'])
        if account_id not in node_attributes:
            node_attributes[account_id] = {
                'age': row['age'],
                'initial_balance': row['initial_balance'],
                'assigned_bank_type': row['assigned_bank_type'],
                'assigned_bank': row['assigned_bank']
            }

    group_df = group_df.drop_duplicates(subset='transaction_id')

    G = nx.MultiDiGraph()
    for _, row in group_df.iterrows():
        account_id = str(row['account_id'])
        counterpart_id = str(row['counterpart_id'])

        # 엣지 그리기
        if row['transaction_direction'] == 'outbound':
            G.add_edge(account_id, counterpart_id,
                       amount=row['amount'],
                       datetime=f'{row["month"]}/{row["day"]} {row["hour"]}:{row["min"]}:{row["sec"]}',
                       channel=row['channel'],
                       payment=row['payment_system'],
                       category=f'{row["category_0"]}/{row["category_1"]}/{row["category_2"]}')
        else:
            G.add_edge(counterpart_id, account_id,
                       amount=row['amount'],
                       datetime=f'{row["month"]}/{row["day"]} {row["hour"]}:{row["min"]}:{row["sec"]}',
                       channel=row['channel'],
                       payment=row['payment_system'],
                       category=f'{row["category_0"]}/{row["category_1"]}/{row["category_2"]}')

    # 노드 속성을 그래프에 추가
    nx.set_node_attributes(G, node_attributes)

    # 그래프 시각화
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=1)  # 레이아웃 설정

    # 노드와 엣지 그리기
    nx.draw(G, pos, with_labels=False, node_color='lightblue', node_size=3000, font_size=10, font_weight='bold')

    # 노드 속성 출력
    node_labels = {
        node: (
            f"{node}\n"
            f"Age:{attr['age']}\n"
            f"Bal:{attr['initial_balance']}\n"
            f"{attr['assigned_bank_type']}\n"
            f"{attr['assigned_bank']}"
            if 'age' in attr and
               'initial_balance' in attr and
               'assigned_bank_type' in attr and
               'assigned_bank' in attr
            else f"{node}"
        )
        for node, attr in G.nodes(data=True)}
    node_labels = {node: label for node, label in node_labels.items() if label}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

    # 엣지 표시
    from collections import defaultdict
    from adjustText import adjust_text

    edge_labels_dict = defaultdict(list)
    for u, v, d in G.edges(data=True):
        edge_labels_dict[(u, v)].append(
            f"{d['datetime']} / {d['amount']} / {d['channel']} / {d['payment']} / {d['category']}")

    # \n으로 연결된 라벨 생성
    edge_labels = {edge: "\n".join(labels) for edge, labels in edge_labels_dict.items()}

    # # 엣지 라벨 출력
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    texts = []
    for (u, v), label in edge_labels.items():
        x_start, y_start = pos[u]
        x_end, y_end = pos[v]

        # 중심점 계산
        x_label = (x_start + x_end) / 2
        y_label = (y_start + y_end) / 2

        # 노드와 겹치지 않도록 약간 위로 이동
        offset = 0.05  # 위치 오프셋
        dx = x_end - x_start
        dy = y_end - y_start
        dist = (dx ** 2 + dy ** 2) ** 0.5
        x_label += offset * (dy / dist)  # y 방향으로 오프셋 적용
        y_label -= offset * (dx / dist)  # x 방향으로 오프셋 적용

        # 라벨 출력
        texts.append(plt.text(
            x_label, y_label, label,
            fontsize=8, color="black", ha="center", va="center", rotation=0,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7)
        ))

    adjust_text(texts, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

    plt.title(f"Graph for Laundering Scheme ID: {scheme_id}", fontsize=14)
    plt.savefig(f"D:\\GoogleDrive\\내 드라이브\\4_BIS\\graph2\\{scheme_id}.png")
    plt.close()

# Laundering Scheme ID 별 데이터 그룹화
parquet_file = ("D:\\GoogleDrive\\내 드라이브\\4_BIS\\Analytics Challenge 2025\\laundering.csv")
df = pd.read_csv(parquet_file)
scheme_groups = df.groupby('laundering_schema_id')

for scheme_id, group_df in scheme_groups:
    draw_graph(scheme_id, group_df)

print("Complete to draw graphs")
