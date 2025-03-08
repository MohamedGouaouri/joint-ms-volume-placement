import numpy as np
import networkx as nx

class EdgeNetworkOptimization:
    def __init__(self, microservice_graph, edge_network):
        self.microservice_graph = microservice_graph
        self.edge_network = edge_network

        self.num_microservices = len(self.microservice_graph.nodes)
        self.num_edge_servers = len(self.edge_network.nodes)

    def get_num_microservices(self):
        return self.num_microservices

    def get_num_edge_servers(self):
        return self.num_edge_servers
        
    def compute_placement_cost(self, x, y, verbose=False):
            # print(x.shape)
            x = (x >= np.max(x, axis=1).reshape(-1, 1).repeat(self.num_edge_servers, axis=1)).astype(int)
            # print(x)
            L_ms = 0
            L_energy = 0
            L_vol = 0

            for m1, m2 in self.microservice_graph.edges():
                cpu_demands_m1 = self.microservice_graph.nodes[m1]["requested_cpu"]
                cpu_demands_m2 = self.microservice_graph.nodes[m2]["requested_cpu"]
                min_bw = self.microservice_graph.edges[(m1, m2)]["bw_min"]
                for n1 in range(self.num_edge_servers):
                    for n2 in range(self.num_edge_servers):
                        path_cost = 0
                        if x[m1, n1] * x[m2, n2] > 0:
                            if n1 != n2:
                                try:
                                    path = nx.shortest_path(self.edge_network, source=n1, target=n2, weight='reverse_bw')
                                    # path_cost = sum(1 / self.edge_network[u][v]['bw'] for u, v in zip(path[:-1], path[1:]))
                                    # path_cost = nx.shortest_path_length(self.edge_network, source=n1 - 1, target=n2 - 1, weight='reverse_bw')
                                    for (u, v) in zip(path[:-1], path[1:]):
                                        # path_cost += min_bw / (self.edge_network.edges[(u, v)]['bw'] + 0.0001)
                                        path_cost += min_bw / self.edge_network.edges[(u, v)]['bw']
                                        # self.edge_network.edges[(u, v)]['bw'] = max(0.1, self.edge_network.edges[(u, v)]['bw'] - min_bw)
                                        # self.edge_network.edges[(u, v)]['reverse_bw'] = 1 / self.edge_network.edges[(u, v)]['bw']
                                except:
                                    pass
                        
                        computation_cost = x[m1, n1] * cpu_demands_m1 / self.edge_network.nodes[n1]["cpu_frequency"] + x[m2, n2] * cpu_demands_m2 / self.edge_network.nodes[n2]["cpu_frequency"]
                        energy_cost = x[m1, n1] * self.edge_network.nodes[n1]["energy_per_cpu_unit"] * cpu_demands_m1 + x[m2, n2] * self.edge_network.nodes[n2]["energy_per_cpu_unit"] * cpu_demands_m2
                        # min_bw_cost = max(0, (x[m1, n1] * x[m2, n2] * min_bw - bw_uv) ** 2)
                        # path_cost =  min_bw / (bw_uv + 0.0001)
                        communication_cost = x[m1, n1] * x[m2, n2] * path_cost
                        # L_ms += data['bw_min'] * x[m1, n1] * x[m2, n2] * path_cost
                        L_ms += communication_cost + computation_cost
                        L_energy += energy_cost
            

            for m in range(self.num_microservices):
                data_bw = self.microservice_graph.nodes[m]["data_bw"]
                for n1 in range(self.num_edge_servers):
                    for n2 in range(self.num_edge_servers):
                        path_cost = 0
                        if n1 != n2:
                            try:
                                path = nx.shortest_path(self.edge_network, source=n1, target=n2, weight='reverse_bw')
                                for (u, v) in zip(path[:-1], path[1:]):
                                    # path_cost += min_bw / (self.edge_network.edges[(u, v)]['bw'] + 0.0001)
                                    path_cost += data_bw / self.edge_network.edges[(u, v)]['bw']
                                    # self.edge_network.edges[(u, v)]['bw'] = max(0.1, self.edge_network.edges[(u, v)]['bw'] - data_bw)
                                    # self.edge_network.edges[(u, v)]['reverse_bw'] = 1 / self.edge_network.edges[(u, v)]['bw']
                            except:
                                pass
                        L_vol += x[m, n1] * y[m, n2] * path_cost


            constraint_violations = 0

            # Placement constraint: Each microservice must be assigned to one edge server
            for m in range(self.num_microservices):
                if np.sum(x[m, :]) != 1:
                    print("Placement constraint violation")
                    constraint_violations += 1

            # Resource constraints per edge server
            for n in range(self.num_edge_servers):
                cpu_usage = sum(x[m, n] * self.microservice_graph.nodes[m]['requested_cpu'] for m in range(self.num_microservices))
                memory_usage = sum(x[m, n] * self.microservice_graph.nodes[m]['requested_memory'] for m in range(self.num_microservices))
                disk_usage = sum(y[m, n] for m in range(self.num_microservices))

                if cpu_usage > self.edge_network.nodes[n]['available_cpu']:
                    print("CPU violation")
                    constraint_violations += 1
                if memory_usage > self.edge_network.nodes[n]['available_memory']:
                    print("Memory violation")
                    
                    constraint_violations += 1
                if disk_usage > self.edge_network.nodes[n]['available_disk']:
                    print(f"Disk violation {disk_usage}, available: {self.edge_network.nodes[n]['available_disk']}")
                    
                    constraint_violations += 1

            # Disk demand constraint
            for m in range(self.num_microservices):
                if np.sum(y[m, :]) != self.microservice_graph.nodes[m]['requested_disk']:
                    print(f"Disk demand constraint violation: allocated: {np.sum(y[m, :])} requested {self.microservice_graph.nodes[m]['requested_disk']}")
                    
                    constraint_violations += 1

            penalty = 10000 * constraint_violations
            if verbose:
                print(f"MS placement cost: {L_ms}, Energy cost {L_energy} , Data placement cost: {L_vol}, Penalty: {penalty}")

            return L_ms + L_energy + L_vol + penalty, L_ms, L_energy, L_vol, penalty
