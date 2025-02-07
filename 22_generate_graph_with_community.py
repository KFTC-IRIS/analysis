import matplotlib.patches as mpatches
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from community import community_louvain
from collections import defaultdict

##################################################
# 1. 데이터 전처리
##################################################
def prepare_data(group_df):
    """
    (1) 거래 ID 기준 중복 제거
    """
    # 중복 제거
    group_df = group_df.drop_duplicates(subset='transaction_id')
    
    return group_df

##################################################
# 2. 방향성 멀티그래프 생성 및 에지 추가
##################################################
def build_multidigraph(group_df):
    """
    Parquet/CSV 등에서 불러온 group_df를 이용하여
    방향성 MultiDiGraph를 생성하고, 에지를 추가해 반환합니다.
    """
    G = nx.MultiDiGraph()
    
    for _, row in group_df.iterrows():
        source = str(row['account_id'])
        target = str(row['counterpart_id'])
        
        # 시간 정보 문자열로 구성
        time_info = f"{row['month']:02}-{row['day']:02} {row['hour']:02}:{row['min']:02}:{row['sec']:02}"
        
        if row['transaction_direction'] == 'inbound':
            G.add_edge(
                target, source,
                weight=row['amount'],
                payment_system=row['payment_system'],
                time=time_info,
                weekday=row['weekday']
            )
        else:
            G.add_edge(
                source, target,
                weight=row['amount'],
                payment_system=row['payment_system'],
                time=time_info,
                weekday=row['weekday']
            )
    return G

##################################################
# 3. 레이아웃 설정
##################################################
def get_layout(G):
    """
    그래프의 레이아웃을 계산하여 반환합니다.
    layout_type에 따라 다른 레이아웃 함수를 쓸 수도 있습니다.
    """

    pos = nx.shell_layout(G)
    
    return pos

##################################################
# 4. PageRank 계산
##################################################
def compute_pagerank(G, weight='weight'):
    """
    가중치(weight)를 고려하여 PageRank를 계산합니다.
    """
    page_rank = nx.pagerank(G, weight=weight)
    return page_rank

##################################################
# 5. 커뮤니티 탐지 (무방향 그래프 변환 후)
##################################################
def detect_communities(G):
    """
    방향성 그래프를 무방향 그래프로 변환 후, Louvain 알고리즘을 사용해 커뮤니티 탐지
    """
    UG = G.to_undirected()
    partition = community_louvain.best_partition(UG)
    
    # 커뮤니티를 딕셔너리 형태로 변환
    communities = defaultdict(list)
    for node, comm_id in partition.items():
        communities[comm_id].append(node)
    
    hub_nodes = [max(comm, key=lambda x: UG.degree(x)) for comm in communities.values()]
    
    return list(communities.values()), hub_nodes

    
##################################################
# 5-2. 그래프 정보 (커뮤니티, PageRank, 허브 여부, Degree) 추가
##################################################
def add_graph_metrics_to_df(df, G, page_rank, communities, hub_nodes):
    """
    방향성 네트워크 G에서 추가적인 네트워크 분석 정보를 df에 추가
    - 커뮤니티 ID
    - PageRank 값
    - Hub Node 여부 (허브 노드는 True, 나머지는 False)
    - Degree (연결된 엣지 수)
    """
    # 1) 커뮤니티 탐지
    # communities, hub_nodes = detect_communities(G)

    # 노드별 커뮤니티 매핑
    node_community_map = {}
    for comm_id, nodes in enumerate(communities):
        for node in nodes:
            node_community_map[node] = comm_id  # 각 노드가 속한 커뮤니티 ID

    # 2) Degree 정보 추가
    node_degree_map = dict(G.degree())

    # 3) 허브 노드 여부 추가
    hub_node_set = set(hub_nodes)  # 검색 속도 최적화

    # 4) DataFrame에 매핑
    df['community'] = df['account_id'].astype(str).map(node_community_map).fillna(-1).astype(int)
    df['pagerank'] = df['account_id'].astype(str).map(page_rank).fillna(0)
    df['hub_node'] = df['account_id'].astype(str).apply(lambda x: x in hub_node_set)
    df['degree'] = df['account_id'].astype(str).map(node_degree_map).fillna(0).astype(int)

    return df




##################################################
# 6. 노드 & 커뮤니티 색상 매핑
##################################################
def get_community_colors(communities, cmap_name='Set2'):
    """
    주어진 커뮤니티 리스트(communities)에 대해
    컬러맵을 적용하여 node -> color 매핑 딕셔너리를 생성
    """
    cmap = cm.get_cmap(cmap_name, len(communities))
    community_colors = {}
    for i, comm in enumerate(communities):
        color_rgba = cmap(i)  # (R, G, B, A)
        for node in comm:
            community_colors[node] = color_rgba
    return community_colors

##################################################
# 7. 노드 시각화
##################################################
def draw_nodes(G, pos, page_rank, communities, hub_nodes, community_colors):
    """
    그래프 상의 노드를 PageRank와 커뮤니티 정보를 바탕으로 시각화
    """
    # 허브 / 비허브 노드 분리
    normal_nodes = [n for n in G.nodes() if n not in hub_nodes]
    hub_nodes    = [n for n in hub_nodes if n in G.nodes()]  # 혹시 제거된 노드가 있다면 필터

    # 1) 허브 아닌 노드
    normal_node_colors = [community_colors.get(n, 'lightgray') for n in normal_nodes]
    normal_node_sizes  = [page_rank.get(n, 0) * 3000 for n in normal_nodes]
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=normal_nodes,
        node_size=normal_node_sizes,
        node_color=normal_node_colors,
        alpha=0.9
    )

    # 2) 허브 노드 (윤곽선 강조)
    hub_node_colors = [community_colors.get(n, 'lightgray') for n in hub_nodes]
    hub_node_sizes  = [page_rank.get(n, 0) * 3000 for n in hub_nodes]
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=hub_nodes,
        node_size=hub_node_sizes,
        node_color=hub_node_colors,
        alpha=1.0,
        linewidths=2,
        edgecolors='black'
    )

    # 노드 라벨
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

##################################################
# 8. 엣지 시각화
##################################################
def draw_edges(G, pos, payment_system_colors=None):
    """
    MultiDiGraph의 모든 엣지(곡선 처리 포함)를 그립니다.
    결제 수단 컬러 매핑 딕셔너리(payment_system_colors)를 적용합니다.
    """
    if payment_system_colors is None:
        payment_system_colors = {
            'OTHER': 'skyblue',
            'FPS': 'green',
            'CHAPS': 'purple',
            'Visa/Mastercard': 'gold',
            'BACS': 'orange',
            'LINK': 'red'
        }
    
    for (u, v, key, data) in G.edges(data=True, keys=True):
        # 멀티 에지 곡률
        rad = 0.15 * (key + 1)
        
        # 엣지 색상
        edge_color = payment_system_colors.get(data['payment_system'], 'gray')

        # (예) CHAPS/BACS/ICS 이 주말이면 표시 안 함
        if data['payment_system'] in ['CHAPS', 'BACS', 'ICS'] and data['weekday'] in ['Saturday', 'Sunday']:
            continue
            
        # 1) 엣지 그리기 (곡률)
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            connectionstyle=f'arc3,rad={rad}',
            arrows=True,
            arrowsize=15,
            edge_color=edge_color,
            width=2
        )

        # 2) 엣지 라벨(곡률 반영)
        x_u, y_u = pos[u]
        x_v, y_v = pos[v]
        dx, dy = x_v - x_u, y_v - y_u
        dist = np.sqrt(dx**2 + dy**2)

        mid_x = (x_u + x_v) / 2
        mid_y = (y_u + y_v) / 2

        perp_x = -dy
        perp_y = dx
        offset_factor = 0.3
        offset = offset_factor * rad * dist
        
        if dist != 0:
            norm_perp_x = perp_x / dist
            norm_perp_y = perp_y / dist
            label_x = mid_x + offset * norm_perp_x
            label_y = mid_y + offset * norm_perp_y
        else:
            label_x, label_y = x_u, y_u

        label_text = f"${data['weight']:.2f}"
        plt.text(
            label_x, label_y,
            label_text,
            fontsize=8,
            color='darkred',
            ha='center',
            va='center',
            bbox=dict(
                facecolor='white',
                alpha=0.5,
                boxstyle='round,pad=0.2'
            )
        )

##################################################
# 9. 범례 구성
##################################################
def add_legends(communities, payment_system_colors, cmap_name='Set2'):
    """
    엣지 결제수단(색) 범례, 커뮤니티(노드 색) 범례를 추가합니다.
    """
    # 1) 결제수단(엣지 색) 범례
    edge_legend_patches = [
        mpatches.Patch(color=color, label=payment) 
        for payment, color in payment_system_colors.items()
    ]
    legend_edges = plt.legend(
        handles=edge_legend_patches,
        title="Payment Systems (Edges)",
        loc='upper left',
        fontsize=10,
        title_fontsize=12
    )
    plt.gca().add_artist(legend_edges)

    # 2) 커뮤니티(노드 색) 범례
    cmap = cm.get_cmap(cmap_name, len(communities))
    community_legend_patches = []
    for i in range(len(communities)):
        c_color = cmap(i)
        community_legend_patches.append(
            mpatches.Patch(color=c_color, label=f"Community {i+1}")
        )

    legend_nodes = plt.legend(
        handles=community_legend_patches,
        title="Communities (Nodes)",
        loc='upper right',
        fontsize=10,
        title_fontsize=12
    )
    plt.gca().add_artist(legend_nodes)

def draw_graph(scheme_id, group_df):
    """
    종합적으로 그래프를 그려주는 메인 함수.
    """
    # 1) 데이터 전처리
    group_df = prepare_data(group_df)

    # 2) 방향성 멀티그래프 생성
    G = build_multidigraph(group_df)

    # 3) 레이아웃 계산
    pos = get_layout(G)

    # 4) PageRank 계산
    page_rank = compute_pagerank(G, weight='weight')

    # 5) 무방향 그래프 만들기 + 커뮤니티 탐지
    communities, hub_nodes = detect_communities(G)

    # **그래프 분석 정보를 데이터프레임에 추가**
    group_df = add_graph_metrics_to_df(group_df, G, page_rank, communities, hub_nodes)

    # 6) 커뮤니티별 색상 매핑
    community_colors = get_community_colors(communities, cmap_name='Set2')

    # 7) 시각화 시작
    plt.figure(figsize=(12, 8))

    # 8) 노드 그리기
    draw_nodes(G, pos, page_rank, communities, hub_nodes, community_colors)

    # 9) 엣지 그리기
    payment_system_colors = {
        'OTHER': 'skyblue',
        'FPS': 'green',
        'CHAPS': 'purple',
        'Visa/Mastercard': 'gold',
        'BACS': 'orange',
        'LINK': 'red'
    }
    draw_edges(G, pos, payment_system_colors)

    # 10) 범례
    add_legends(communities, payment_system_colors, cmap_name='Set2')

    # 마무리
    plt.title("MultiDiGraph with Distinct Curved Edges & Community Detection", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return group_df

if __name__ == "__main__":
    # 1. 특정 id만 보고싶을때
    test_df = df[df['laundering_schema_id'] == 'laundering_schema_1_id7']
    community_df = draw_graph("", test_df)

    # # 2. scheme_id별 전체 데이터를 보고싶으면 해당 코드 실행
    # scheme_groups = df.groupby('laundering_schema_id')
    # for scheme_id, group_df in scheme_groups:
    #     draw_graph(scheme_id, group_df)


    # # 3. 전체 데이터를 한번에 볼때
    # community_df = draw_graph("", df) # 사용할 데이터프레임
    # community_df
