import networkx as nx
import random
import matplotlib.pyplot as plt


class BaseMicroserviceEdgeEnvironment():
    def __init__(self, num_edge_servers=10, block_size=1):
        self.app_name = ''
        self.block_size = block_size
        self.edge_network = nx.erdos_renyi_graph(num_edge_servers, 0.5, seed=123, directed=False)
        self.num_edge_servers = num_edge_servers
        self._build_edge_network()

    def _build_edge_network(self):
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

    def _build_ms_graph(self):
        raise NotImplementedError()

    def _add_service(self, name, cpu, memory, disk, data_bw):
        self.microservice_graph.add_node(name, requested_cpu=cpu, requested_memory=memory, requested_disk=disk, data_bw=data_bw)

    def _add_dependency(self, src, dest, bandwidth):
        self.microservice_graph.add_edge(src, dest, bw_min=bandwidth)

    def plot_graph(self):
        self._plot(self.app_name)

    def _plot(self, title):
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(self.microservice_graph)
        nx.draw(self.microservice_graph, pos, with_labels=True, node_color='lightblue', node_size=1200)
        labels = {node: f"{node}\nCPU: {data['requested_cpu']}\nMem: {data['requested_memory']}" for node, data in self.microservice_graph.nodes(data=True)}
        nx.draw_networkx_labels(self.microservice_graph, pos, labels=labels, font_size=8)
        edge_labels = {(u, v): f"BW: {data['bw_min']}" for u, v, data in self.microservice_graph.edges(data=True)}
        nx.draw_networkx_edge_labels(self.microservice_graph, pos, edge_labels=edge_labels, font_size=8)
        plt.title(title)
        plt.show()