
import numpy as np
import networkx as nx
from copy import deepcopy
from .utils import find_k_hop_neighbors

# Algorithm based on node ranks and centrality
class GraphCentralityRankBasedHeuristic:

    def __init__(self, sim):
        self.sim = sim
        
        self.microservice_graph = self.sim.microservice_graph
        self.edge_network = self.sim.edge_network

        self.num_microservices = len(self.microservice_graph.nodes)
        self.num_edge_servers = len(self.edge_network.nodes)

        self.ms_centrality = nx.degree_centrality(self.microservice_graph)
        self.ms_weighted_degrees = dict(self.microservice_graph.degree(weight='bw_min'))

        self.rank_cache = {}
        
    def _get_candidate_server(self, ms, edge_network):
        cpu_demand, memory_demand = self.microservice_graph.nodes[ms]['requested_cpu'], self.microservice_graph.nodes[ms]['requested_memory']
        candidate_edge_servers = []
        for edge_node, edge_attr in edge_network.nodes(data=True):
            available_cpu, available_memory = edge_attr['available_cpu'], edge_attr['available_memory']
            if available_cpu > cpu_demand and available_memory > memory_demand:
                candidate_edge_servers.append(edge_node)
        
        return candidate_edge_servers
    def _calculate_computation_cost(self, ms, edge_network):
        # Compute average computation cost on the available edge servers
        # Filter the edge servers based on the available resources
        candidate_edge_servers = self._get_candidate_server(ms, edge_network)
        # If no candidate edge server was found, return 0
        if len(candidate_edge_servers) == 0:
            return 0
        cpu_demand = self.microservice_graph.nodes[ms]['requested_cpu']
        avg = 0
        for candidate in candidate_edge_servers:
            avg += sum([cpu_demand / self.edge_network.nodes[candidate]["cpu_frequency"]])
        avg /= len(candidate_edge_servers)
        return avg
    
    def _calculate_communication_cost(self, m1, m2, edge_network):
        # Compute average communication cost between two microservices
        min_bw = self.microservice_graph.edges[(m1, m2)]['bw_min']
        candidate_edge_servers1 = self._get_candidate_server(m1, edge_network)
        candidate_edge_servers2 = self._get_candidate_server(m2, edge_network)
        if len(candidate_edge_servers1) == 0 or len(candidate_edge_servers2) == 0:
            return 0
        
        avg = 0
        for c1 in candidate_edge_servers1:
            for c2 in candidate_edge_servers2:
                path_cost = 0
                try:
                    path = nx.shortest_path(self.edge_network, source=c1, target=c2, weight='reverse_bw')
                    for u, v in zip(path[:-1], path[1:]):
                        path_cost += min_bw / self.edge_network.edges[(u, v)]['bw']
                except:
                    pass

                avg += path_cost
        avg /= len(candidate_edge_servers1) * len(candidate_edge_servers2)
        return avg
    def _calculate_rank(self, ms, edge_network):
        # If rank is already computed, return it from cache
        if ms in self.rank_cache:
            return self.rank_cache[ms]

        # Compute computation cost for the current microservice
        computation_cost = self._calculate_computation_cost(ms, edge_network)

        # Initialize max_neighbor_rank to 0
        max_neighbor_rank = 0

        # Iterate over all neighbors of the current microservice
        for neighbor in self.microservice_graph.neighbors(ms):
            # Recursively calculate the rank of the neighbor
            neighbor_rank = self._calculate_rank(neighbor, edge_network)
            # Calculate communication cost between current microservice and neighbor
            communication_cost = self._calculate_communication_cost(ms, neighbor, edge_network)
            # Update max_neighbor_rank
            max_neighbor_rank = max(max_neighbor_rank, neighbor_rank + communication_cost)

        # Calculate the rank of the current microservice
        rank = computation_cost + max_neighbor_rank + self.ms_centrality[ms]

        # Cache the computed rank
        self.rank_cache[ms] = rank

        return rank
    def solve(self, verbose=False):
        ms_placement = np.zeros((self.num_microservices, self.num_edge_servers))
        data_placement = np.zeros((self.num_microservices, self.num_edge_servers))

        edge_network = deepcopy(self.edge_network)
        
        # Calculate node degree centrality for microservice graph
        # ms_centrality = nx.degree_centrality(microservice_graph)
        # ms_weighted_degrees = dict(microservice_graph.degree(weight='bw_min'))
        server_centrality = nx.degree_centrality(edge_network)
        server_weighted_degrees = dict(edge_network.degree(weight='bw'))
        # server_centrality = {key: server_centrality[key] * server_weighted_degrees[key] for key in server_centrality}
        # sorted_servers = sorted(server_centrality, key=lambda k: -server_centrality[k])
        # Recursively calculate microservice graph nodes ranks using
        # rank(m) = computation_cost(m) + max(rank(n) + communication_cost(m, n) where n is a neighbor of m)
        ranks = {}
        for ms in self.microservice_graph.nodes:
            ranks[ms] = self._calculate_rank(ms, edge_network)
        
        # Sort microservices based on rank
        sorted_ms = sorted(ranks, key=lambda k: -ranks[k])
        for ms in sorted_ms:
            cpu_demand = self.microservice_graph.nodes[ms]['requested_cpu']
            memory_demand = self.microservice_graph.nodes[ms]['requested_memory']
            candidate_edge_servers = self._get_candidate_server(ms, edge_network)
            if len(candidate_edge_servers) == 0:
                print('No candidate edge server was found')
                break
            # Score based on energy consumption
            energy_scores = {}
            for candidate in candidate_edge_servers:
                energy_scores[candidate] = -edge_network.nodes[candidate]["energy_per_cpu_unit"] * cpu_demand

            
            highest_score = -np.inf
            highest_candidate = None
            # print(f"Microservice {ms}")
            for candidate in candidate_edge_servers:
                score = server_centrality[candidate] + energy_scores[candidate]
                # print(f"Candidate {candidate}: {score}")
                if score >= highest_score:
                    highest_score = score
                    highest_candidate = candidate
            
            ms_placement[ms, highest_candidate] = 1
            
            # Update resources
            edge_network.nodes[highest_candidate]['available_cpu'] -= cpu_demand
            edge_network.nodes[highest_candidate]['available_memory'] -= memory_demand
        
        
        # Data placement
        # Sort microservices by data bandwidth
        bw_sorted_ms = sorted(self.microservice_graph.nodes, key=lambda k: -self.microservice_graph.nodes[k]['data_bw'])
        for ms in bw_sorted_ms:
            disk_demand = self.sim.microservice_graph.nodes[ms]['requested_disk']
            placed_on = ms_placement.argmax(axis=1)[ms]
            k = 0
            while k < 10 and disk_demand > 0: # TODO: To be changed (k)
                k_hop_neighbors = find_k_hop_neighbors(edge_network, placed_on, k)
                # Sort k hop neighbors by bandwith
                k_hop_neighbors = sorted(k_hop_neighbors, key=lambda x: -server_centrality[x])
                
                for neighbor in k_hop_neighbors:
                    available = edge_network.nodes[neighbor]['available_disk']
                    if disk_demand > 0 and available > 0:
                        if disk_demand <= available:
                            allocation = disk_demand
                            disk_demand = 0
                        else:
                            allocation = available
                            disk_demand -= allocation
                        
                        edge_network.nodes[neighbor]['available_disk'] -= allocation
                        data_placement[ms, neighbor] = allocation
                k += 1
        return ms_placement, data_placement