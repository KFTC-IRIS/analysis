import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

def add_edge_with_amount(G, u, v, amount, direction):
    if direction == 'outbound' and G.has_edge(u, v):
        G[u][v]['amount'] += round(amount)
    elif direction == 'inbound' and G.has_edge(v, u):
        G[v][u]['amount'] += round(amount)
    else:
        if direction == 'outbound':
            G.add_edge(u, v, amount=round(amount))
        else:
            G.add_edge(v, u, amount=round(amount))

def draw_graph(scheme_id, group_df):
    group_df = group_df.drop_duplicates(subset='transaction_id')

    G = nx.DiGraph()
    for _, row in group_df.iterrows():
        account_id = str(row['account_id'])
        counterpart_id = str(row['counterpart_id'])
        amount = row['amount']

        add_edge_with_amount(G, account_id, counterpart_id, amount, direction=row['transaction_direction'])

    # 그래프 시각화
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)

    nx.draw(
        G, pos, with_labels=True, node_size=700, node_color="skyblue",
        font_size=10, edge_color="gray", arrowsize=20
    )

    edge_labels = {
        (u, v): f"${d['amount']}"
        for u, v, d in G.edges(data=True)
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.5)

    plt.title(f"Graph for Laundering Scheme ID: {scheme_id}", fontsize=14)
    plt.savefig(f"D:\\GoogleDrive\\내 드라이브\\4_BIS\\graph\\{scheme_id}.png")
    plt.close()

# Laundering Scheme ID 별 데이터 그룹화
parquet_file = ("D:\\GoogleDrive\\내 드라이브\\4_BIS\\Analytics Challenge 2025\\laundering.csv")
df = pd.read_csv(parquet_file)
scheme_groups = df.groupby('laundering_schema_id')

for scheme_id, group_df in scheme_groups:
    draw_graph(scheme_id, group_df)

print("Complete to draw graphs")
