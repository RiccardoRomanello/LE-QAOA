import networkx as nx
import random

def generate_graphs(num_graphs=5, num_nodes=8, max_degree=4, seed=None):
    """
    Generates a list of random graphs with specified parameters.
    Each graph will have `num_nodes` nodes and edges added randomly while ensuring that no node exceeds `max_degree`. The `seed` parameter can be set for reproducibility.
    """
    if seed is not None:
        random.seed(seed)
    
    graphs = []
    for _ in range(num_graphs):
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        possible_edges = [(i,j) for i in range(num_nodes) for j in range(i+1, num_nodes)]
        random.shuffle(possible_edges)
        for u,v in possible_edges:
            if G.degree(u) < max_degree and G.degree(v) < max_degree:
                G.add_edge(u,v)
        graphs.append(G)
    return graphs