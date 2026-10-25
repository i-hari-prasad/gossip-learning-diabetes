import random
import networkx as nx


def create_topology(patients: list, topology_type: str = 'ring') -> nx.Graph:
    """
    Creates a communication graph among patient nodes.

    Supported topologies: 'ring', 'fully_connected', 'random', 'small_world'
    """
    n = len(patients)
    G = nx.Graph()
    G.add_nodes_from(patients)

    if n <= 1:
        return G

    if topology_type == 'ring':
        for i in range(n):
            G.add_edge(patients[i], patients[(i + 1) % n])

    elif topology_type == 'fully_connected':
        for i in range(n):
            for j in range(i + 1, n):
                G.add_edge(patients[i], patients[j])

    elif topology_type in ('random', 'sparse_peer_to_peer'):
        k = max(2, min(3, n - 1))
        # Ensure k*n is even for random_regular_graph
        if (k * n) % 2 != 0:
            k = max(2, k - 1)
        try:
            tmp = nx.random_regular_graph(k, n, seed=42)
            mapping = dict(zip(range(n), patients))
            G = nx.relabel_nodes(tmp, mapping)
        except Exception:
            # Fallback: ring
            for i in range(n):
                G.add_edge(patients[i], patients[(i + 1) % n])

    elif topology_type == 'small_world':
        k = max(2, min(4, n - 1))
        tmp = nx.watts_strogatz_graph(n, k, 0.3, seed=42)
        mapping = dict(zip(range(n), patients))
        G = nx.relabel_nodes(tmp, mapping)

    else:
        # Default: ring
        for i in range(n):
            G.add_edge(patients[i], patients[(i + 1) % n])

    return G


def get_neighbors(G: nx.Graph, node) -> list:
    return list(G.neighbors(node)) if node in G else []
