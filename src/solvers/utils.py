
import networkx as nx

def find_k_hop_neighbors(graph, node, k):
    """
    Find the k-hop neighbors of a given node in the graph.

    :param graph: A NetworkX graph
    :param node: Node for which to find neighbors
    :param k: Number of hops
    :return: A list of k-hop neighbors sorted by path cost
    """
    # Find k-hop neighbors
    neighbors = set(
        n for n, dist in nx.single_source_shortest_path_length(graph, node).items() if dist == k
    )

    # Calculate path costs for each neighbor
    neighbor_cost_pairs = []
    for neighbor in neighbors:
        path_cost = nx.shortest_path_length(graph, source=node, target=neighbor, weight='reverse_bw')
        neighbor_cost_pairs.append((neighbor, path_cost))

    # Sort neighbors by path cost
    sorted_neighbors = sorted(neighbor_cost_pairs, key=lambda x: x[1])

    # Extract and return only the sorted neighbors (without costs)
    return [neighbor for neighbor, cost in sorted_neighbors]
