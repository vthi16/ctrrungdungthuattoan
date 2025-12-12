import streamlit as st
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import heapq
from io import BytesIO

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Đồ án CTDL&GT Full", layout="wide", page_icon="🎓")


# ==============================================================================
# PHẦN 1: THƯ VIỆN THUẬT TOÁN TỰ VIẾT (MANUAL IMPLEMENTATION)
# (Phần này để lấy điểm thuật toán & trả lời vấn đáp)
# ==============================================================================

def my_bfs(G, start_node):
    visited = set()
    queue = [start_node]
    visited.add(start_node)
    path_order = []
    edges_path = []

    while queue:
        u = queue.pop(0)
        path_order.append(u)
        neighbors = sorted(list(G.neighbors(u)))  # Sort để thứ tự duyệt nhất quán
        for v in neighbors:
            if v not in visited:
                visited.add(v)
                queue.append(v)
                edges_path.append((u, v))
    return edges_path, path_order


def my_dfs(G, start_node):
    visited = set()
    stack = [start_node]
    path_order = []
    edges_path = []

    while stack:
        u = stack.pop()
        if u not in visited:
            visited.add(u)
            path_order.append(u)
            neighbors = sorted(list(G.neighbors(u)), reverse=True)  # Reverse để pop ra theo thứ tự nhỏ->lớn
            for v in neighbors:
                if v not in visited:
                    stack.append(v)
                    edges_path.append((u, v))
    return edges_path, path_order


def my_dijkstra(G, start_node, end_node):
    distances = {node: float('infinity') for node in G.nodes()}
    distances[start_node] = 0
    pq = [(0, start_node)]
    parent = {node: None for node in G.nodes()}

    while pq:
        d, u = heapq.heappop(pq)
        if u == end_node: break
        if d > distances[u]: continue

        for v in G.neighbors(u):
            weight = G[u][v].get('weight', 1)
            if distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                parent[v] = u
                heapq.heappush(pq, (distances[v], v))

    path = []
    curr = end_node
    if distances[end_node] == float('infinity'): return None, 0
    while curr is not None:
        path.insert(0, curr)
        curr = parent[curr]
    return path, distances[end_node]


def check_bipartite_manual(G):
    """Kiểm tra đồ thị 2 phía bằng BFS tô màu (0 và 1)"""
    color = {}
    for node in G.nodes():
        if node not in color:
            color[node] = 0
            queue = [node]
            while queue:
                u = queue.pop(0)
                for v in G.neighbors(u):
                    if v not in color:
                        color[v] = 1 - color[u]
                        queue.append(v)
                    elif color[v] == color[u]:
                        return False, {}  # Hai đỉnh kề nhau cùng màu -> Không phải 2 phía
    return True, color


def my_prim(G):
    if G.is_directed(): return None, "Prim chỉ dùng cho đồ thị Vô hướng!"
    if not nx.is_connected(G): return None, "Đồ thị không liên thông!"

    start_node = list(G.nodes())[0]
    mst_edges = []
    visited = {start_node}
    edges_heap = []

    for v in G.neighbors(start_node):
        w = G[start_node][v].get('weight', 1)
        heapq.heappush(edges_heap, (w, start_node, v))

    total_w = 0
    while len(mst_edges) < len(G.nodes()) - 1 and edges_heap:
        w, u, v = heapq.heappop(edges_heap)
        if v not in visited:
            visited.add(v)
            mst_edges.append((u, v))
            total_w += w
            for next_n in G.neighbors(v):
                if next_n not in visited:
                    new_w = G[v][next_n].get('weight', 1)
                    heapq.heappush(edges_heap, (new_w, v, next_n))
    return mst_edges, total_w


def my_kruskal(G):
    edges = sorted([(data.get('weight', 1), u, v) for u, v, data in G.edges(data=True)])
    parent = {n: n for n in G.nodes()}

    def find(n):
        if parent[n] != n: parent[n] = find(parent[n])
        return parent[n]

    def union(u, v):
        r1, r2 = find(u), find(v)
        if r1 != r2: parent[r1] = r2; return True
        return False

    mst = []
    total_w = 0
    for w, u, v in edges:
        if union(u, v):
            mst.append((u, v))
            total_w += w
    return mst, total_w


def my_ford_fulkerson(G, source, sink):
    """Thuật toán Edmonds-Karp (BFS) để tìm luồng cực đại"""
    if not G.is_directed(): return None, "Max Flow cần đồ thị CÓ HƯỚNG!"

    # Tạo đồ thị thặng dư (Residual Graph)
    R = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        capacity = data.get('weight', 1)  # Coi trọng số là sức chứa
        R.add_edge(u, v, capacity=capacity)
        R.add_edge(v, u, capacity=0)  # Cạnh ngược ban đầu = 0

    max_flow = 0
    path_flow_details = []

    while True:
        # Tìm đường đi tăng luồng bằng BFS trên đồ thị thặng dư
        parent = {node: None for node in R.nodes()}
        queue = [source]
        path_found = False
        while queue:
            u = queue.pop(0)
            if u == sink:
                path_found = True
                break
            for v in R.neighbors(u):
                if parent[v] is None and R[u][v]['capacity'] > 0:
                    parent[v] = u
                    queue.append(v)

        if not path_found: break  # Không còn đường tăng luồng

        # Tìm bottleneck (dòng chảy nhỏ nhất trên đường đi tìm được)
        path_flow = float('inf')
        v = sink
        path = []
        while v != source:
            u = parent[v]
            path.insert(0, v);
            path.insert(0, u)  # Lưu vết
            path_flow = min(path_flow, R[u][v]['capacity'])
            v = u

        max_flow += path_flow
        path_flow_details.append((list(dict.fromkeys(path)), path_flow))  # Xóa duplicate node để hiện thị đẹp

        # Cập nhật đồ thị thặng dư
        v = sink
        while v != source:
            u = parent[v]
            R[u][v]['capacity'] -= path_flow
            R[v][u]['capacity'] += path_flow
            v = u

    return max_flow, path_flow_details


def my_hierholzer(G):
    """Tìm chu trình Euler (Hierholzer)"""
    if not nx.is_connected(G.to_undirected()): return None, "Đồ thị không liên thông!"

    # Kiểm tra bậc chẵn (Vô hướng) hoặc Bán bậc ra = Bán bậc vào (Có hướng)
    if not G.is_directed():
        odd_nodes = [v for v, d in G.degree() if d % 2 != 0]
        if odd_nodes: return None, "Không có chu trình Euler (Tồn tại đỉnh bậc lẻ)."
    else:
        for v in G.nodes():
            if G.out_degree(v) != G.in_degree(v):
                return None, "Không có chu trình Euler (Bán bậc ra != Bán bậc vào)."

    # Copy đồ thị để xóa cạnh dần
    temp_G = G.copy()
    if G.is_directed():
        temp_G = nx.MultiDiGraph(G)
    else:
        temp_G = nx.MultiGraph(G)

    stack = [list(temp_G.nodes())[0]]
    circuit = []

    while stack:
        u = stack[-1]
        if temp_G.degree(u) > 0:
            # Lấy 1 đỉnh kề và xóa cạnh
            v = list(temp_G.neighbors(u))[0]
            temp_G.remove_edge(u, v)
            stack.append(v)
        else:
            circuit.append(stack.pop())

    return circuit[::-1], "Thành công"


# ==============================================================================
# PHẦN 2: HÀM HỖ TRỢ & GIAO DIỆN
# ==============================================================================

def ve_do_thi(G, highlight_edges=None, highlight_nodes=None, title="", color_map=None):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 5))

    node_colors = 'lightblue'
    if color_map:
        node_colors = [color_map.get(node, 'gray') for node in G.nodes()]
        # Map 0 -> red, 1 -> blue
        node_colors = ['#ff7675' if c == 0 else '#74b9ff' if c == 1 else 'gray' for c in node_colors]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600)
    nx.draw_networkx_labels(G, pos, font_weight='bold')

    nx.draw_networkx_edges(G, pos, edge_color='#b2bec3', width=1, arrows=G.is_directed(), arrowsize=15)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    if highlight_edges:
        nx.draw_networkx_edges(G, pos, edgelist=highlight_edges, edge_color='#e17055', width=3)
    if highlight_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=highlight_nodes, node_color='#fab1a0', node_size=650)

    plt.title(title, fontsize=14)
    plt.axis('off')
    return plt


st.title("🎓 ĐỒ ÁN MÔN HỌC: ỨNG DỤNG ĐỒ THỊ")
st.markdown("---")

# --- SIDEBAR: NHẬP LIỆU ---
with st.sidebar:
    st.header("1. Cấu trúc dữ liệu")
    type_g = st.radio("Loại:", ["Vô hướng", "Có hướng"])
    input_text = st.text_area("Nhập cạnh (u v w):", "A B 4\nA C 2\nB C 5\nB D 10\nC E 3\nD F 11\nE D 4")

    if st.button("🚀 Khởi tạo Đồ thị", type="primary"):
        G = nx.DiGraph() if type_g == "Có hướng" else nx.Graph()
        try:
            for line in input_text.strip().split('\n'):
                parts = line.split()
                if len(parts) >= 2:
                    w = int(parts[2]) if len(parts) > 2 else 1
                    G.add_edge(parts[0], parts[1], weight=w)
            st.session_state['graph'] = G
            st.session_state['input_raw'] = input_text
            st.success("Đã nạp dữ liệu!")
        except:
            st.error("Lỗi định dạng!")

    # Chức năng 2: Lưu đồ thị
    if 'input_raw' in st.session_state:
        st.download_button("💾 Lưu file Graph.txt", st.session_state['input_raw'], "graph.txt")

# --- MAIN SCREEN ---
if 'graph' in st.session_state:
    G = st.session_state['graph']

    tab1, tab2, tab3 = st.tabs(["🖼️ Trực quan & Thuật toán", "📊 Biểu diễn dữ liệu", "🔍 Kiểm tra tính chất"])

    # TAB 1: THUẬT TOÁN (Gồm Cơ bản 3,4 và Nâng cao 7.x)
    with tab1:
        c1, c2 = st.columns([1, 2])
        with c1:
            algo = st.selectbox("Chọn thuật toán:",
                                ["BFS", "DFS", "Dijkstra (Shortest Path)",
                                 "Prim (MST)", "Kruskal (MST)",
                                 "Ford-Fulkerson (Max Flow)", "Hierholzer (Euler Circuit)"])

            start = st.selectbox("Đỉnh bắt đầu:", G.nodes())
            end = st.selectbox("Đỉnh đích:", G.nodes(), index=len(G.nodes()) - 1)

            run_btn = st.button("Chạy mô phỏng")

        with c2:
            fig = None
            msg = ""
            if run_btn:
                try:
                    if algo == "BFS":
                        edges, order = my_bfs(G, start)
                        fig = ve_do_thi(G, highlight_edges=edges, title=f"BFS từ {start}")
                        msg = f"Thứ tự duyệt: {order}"

                    elif algo == "DFS":
                        edges, order = my_dfs(G, start)
                        fig = ve_do_thi(G, highlight_edges=edges, title=f"DFS từ {start}")
                        msg = f"Thứ tự duyệt: {order}"

                    elif algo == "Dijkstra (Shortest Path)":
                        path, dist = my_dijkstra(G, start, end)
                        if path:
                            edges = list(zip(path, path[1:]))
                            fig = ve_do_thi(G, highlight_edges=edges, title=f"Dijkstra Cost: {dist}")
                            msg = f"Đường đi: {' → '.join(path)}"
                        else:
                            st.error("Không có đường đi")

                    elif algo == "Prim (MST)":
                        mst, w = my_prim(G)
                        if mst:
                            fig = ve_do_thi(G, highlight_edges=mst, title=f"Prim MST Cost: {w}")
                            msg = f"Các cạnh MST: {mst}"
                        else:
                            st.error(w)

                    elif algo == "Kruskal (MST)":
                        mst, w = my_kruskal(G)
                        if mst:
                            fig = ve_do_thi(G, highlight_edges=mst, title=f"Kruskal MST Cost: {w}")
                            msg = f"Các cạnh MST: {mst}"
                        else:
                            st.error(w)

                    elif algo == "Ford-Fulkerson (Max Flow)":
                        val, details = my_ford_fulkerson(G, start, end)
                        if val is not None:
                            fig = ve_do_thi(G, title=f"Max Flow: {val}")
                            msg = f"Luồng cực đại: {val}. Chi tiết đường tăng luồng: {details}"
                        else:
                            st.error(details)  # Lỗi

                    elif algo == "Hierholzer (Euler Circuit)":
                        path, err = my_hierholzer(G)
                        if path:
                            edges = list(zip(path, path[1:]))
                            fig = ve_do_thi(G, highlight_edges=edges, title="Chu trình Euler")
                            msg = f"Chu trình: {' → '.join(map(str, path))}"
                        else:
                            st.error(err)

                except Exception as e:
                    st.error(f"Lỗi runtime: {e}")

            else:
                fig = ve_do_thi(G, title="Đồ thị ban đầu")

            st.pyplot(fig)
            if msg: st.info(msg)

    # TAB 2: BIỂU DIỄN DỮ LIỆU (Yêu cầu Cơ bản 6)
    with tab2:
        st.subheader("Chuyển đổi các dạng biểu diễn")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write("**1. Danh sách cạnh (Edge List)**")
            df_edges = nx.to_pandas_edgelist(G)
            st.dataframe(df_edges, hide_index=True)
        with c2:
            st.write("**2. Ma trận kề (Adjacency Matrix)**")
            matrix = nx.adjacency_matrix(G).todense()
            st.dataframe(pd.DataFrame(matrix, index=G.nodes(), columns=G.nodes()))
        with c3:
            st.write("**3. Danh sách kề (Adjacency List)**")
            adj_dict = {n: list(G.neighbors(n)) for n in G.nodes()}
            st.json(adj_dict)

    # TAB 3: KIỂM TRA TÍNH CHẤT (Yêu cầu Cơ bản 5)
    with tab3:
        st.subheader("Kiểm tra Đồ thị 2 phía (Bipartite)")
        is_bi, color_map = check_bipartite_manual(G)

        c1, c2 = st.columns([1, 2])
        with c1:
            if is_bi:
                st.success("✅ Đây LÀ đồ thị 2 phía!")
                st.write("Phân hoạch tập đỉnh:")
                set_0 = [n for n, c in color_map.items() if c == 0]
                set_1 = [n for n, c in color_map.items() if c == 1]
                st.write(f"Tập A (Đỏ): {set_0}")
                st.write(f"Tập B (Xanh): {set_1}")
            else:
                st.error("❌ Đây KHÔNG PHẢI đồ thị 2 phía.")
        with c2:
            if is_bi:
                fig_bi = ve_do_thi(G, title="Mô phỏng 2 phía", color_map=color_map)
                st.pyplot(fig_bi)

else:
    st.info("👈 Mời bạn nhập dữ liệu ở thanh bên trái để bắt đầu.")