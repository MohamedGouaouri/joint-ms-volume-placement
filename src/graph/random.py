import networkx as nx
import random
import matplotlib.pyplot as plt

class RandomMicroserviceEdgeEnvironment:
    def __init__(self, num_microservices=10, num_edge_servers=10, block_size=1):
        self.num_microservices = num_microservices
        self.num_edge_servers = num_edge_servers
        self.block_size = block_size

        self.microservice_graph = nx.DiGraph()
        self.edge_network = nx.erdos_renyi_graph(num_edge_servers, 0.5, seed=123, directed=False)

        self._create_microservice_graph()
        self._create_edge_network()

    def _create_microservice_graph(self):
        for m in range(self.num_microservices):
            self.microservice_graph.add_node(m,
                                            requested_cpu=random.randint(100, 500),
                                            requested_memory=random.randint(512, 1024),
                                            requested_disk=random.randint(self.block_size, 4 * self.block_size),
                                            data_bw=random.randint(10, 100)
                                            )

        # Add edges ensuring no cycles (DAG)
        possible_edges = [(i, j) for i in range(self.num_microservices) for j in range(i + 1, self.num_microservices)]
        for _ in range(self.num_microservices * 2):  # Add twice the number of nodes as edges
            if not possible_edges:
                break
            m1, m2 = random.choice(possible_edges)
            possible_edges.remove((m1, m2))
            bandwidth = random.randint(10, 100)
            self.microservice_graph.add_edge(m1, m2, bw_min=bandwidth)

    def _create_edge_network(self):
        for n in range(self.num_edge_servers):
            self.edge_network.add_node(n,
                                      available_cpu=random.randint(4000, 8000),
                                      available_memory=random.randint(4096, 2 * 4096),
                                      available_disk=random.randint(4 * self.block_size, 16 * self.block_size),
                                      energy_per_cpu_unit=random.random(),
                                      cpu_frequency=random.choice([1000, 2000, 3000, 4000])

                                      )

        for n1, n2 in self.edge_network.edges():
            bandwidth = random.randint(100, 1000)
            self.edge_network.add_edge(n1, n2, bw=bandwidth, reverse_bw=1 / bandwidth)

    def plot_graphs(self):
        plt.figure(figsize=(16, 8))

        # Microservice Graph
        plt.subplot(1, 2, 1)
        pos_ms = nx.spring_layout(self.microservice_graph)
        nx.draw(self.microservice_graph, pos_ms, with_labels=True, node_color='skyblue', node_size=1200)
        labels_ms = {node: f"CPU: {self.microservice_graph.nodes[node]['requested_cpu']}\nMem: {self.microservice_graph.nodes[node]['requested_memory']}\nDisk: {self.microservice_graph.nodes[node]['requested_disk']}" for node in self.microservice_graph.nodes}
        nx.draw_networkx_labels(self.microservice_graph, pos_ms, labels=labels_ms, font_size=8)

        edge_labels_ms = {(u, v): f"BW: {data['bw_min']}" for u, v, data in self.microservice_graph.edges(data=True)}
        nx.draw_networkx_edge_labels(self.microservice_graph, pos_ms, edge_labels=edge_labels_ms, font_size=8)
        plt.title("Microservice Graph (Demands)")

        # Edge Network Graph
        plt.subplot(1, 2, 2)
        pos_edge = nx.spring_layout(self.edge_network)
        nx.draw(self.edge_network, pos_edge, with_labels=True, node_color='lightgreen', node_size=1200)
        labels_edge = {node: f"CPU: {self.edge_network.nodes[node]['available_cpu']}\nMem: {self.edge_network.nodes[node]['available_memory']}\nDisk: {self.edge_network.nodes[node]['available_disk']}" for node in self.edge_network.nodes}
        nx.draw_networkx_labels(self.edge_network, pos_edge, labels=labels_edge, font_size=8)

        edge_labels_edge = {(u, v): f"BW: {data['bw']}" for u, v, data in self.edge_network.edges(data=True)}
        nx.draw_networkx_edge_labels(self.edge_network, pos_edge, edge_labels=edge_labels_edge, font_size=8)
        plt.title("Edge Network Graph (Capacities)")

        plt.tight_layout()
        plt.show()

