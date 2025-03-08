import numpy as np
import networkx as nx
from copy import deepcopy
from .utils import find_k_hop_neighbors

class TopologicalSortingHeuristic:
    cloud_resources = {'available_disk': float('inf')}  # Cloud has virtually infinite disk space
    CLOUD_NODE = 'cloud'
    
    def __init__(self, sim):
        self.sim = sim
        
        self.microservice_graph = self.sim.microservice_graph
        self.edge_network = self.sim.edge_network

        self.num_microservices = len(self.microservice_graph.nodes)
        self.num_edge_servers = len(self.edge_network.nodes)
    
    def solve(self, verbose=False):
        edge_network = deepcopy(self.sim.edge_network)
        
        sorted_ms = list(nx.topological_sort(self.microservice_graph))
        server_degree_centrality = nx.degree_centrality(edge_network)
        sorted_servers = sorted(server_degree_centrality, key=lambda k: -server_degree_centrality[k])
        ms_placement = np.zeros((len(sorted_ms), len(sorted_servers) ))
        data_placement = np.zeros((len(sorted_ms), len(sorted_servers) ))
        
        for ms in sorted_ms:
            # print(ms)
            # Filter edges based on the resources, ie edge servers that can't accomodate the ms are filtered out
            cpu_demand, memory_demand, disk_demand = self.microservice_graph.nodes[ms]['requested_cpu'], self.microservice_graph.nodes[ms]['requested_memory'], self.microservice_graph.nodes[ms]['requested_disk']
            candidate_edge_servers = []
            # Filter
            for edge_node, edge_attr in edge_network.nodes(data=True):
                available_cpu, available_memory = edge_attr['available_cpu'], edge_attr['available_memory']
                if available_cpu > cpu_demand and available_memory > memory_demand:
                    candidate_edge_servers.append(edge_node)
        
            if len(candidate_edge_servers) == 0:
                print('No candidate edge server was found')
                break
            
            # Score based on energy consumption
            energy_scores = {}
            for candidate in candidate_edge_servers:
                # energy_scores[candidate] = -edge_network.nodes[candidate]["energy_per_cpu_unit"] * cpu_demand
                energy_scores[candidate] = -edge_network.nodes[candidate]["energy_per_cpu_unit"] * cpu_demand

            
            highest_score = -np.inf
            highest_candidate = None
            for candidate in candidate_edge_servers:
                score = 0.5 * server_degree_centrality[candidate] + 0.5 * energy_scores[candidate]
                if score >= highest_score:
                    highest_score = 0.5 * server_degree_centrality[candidate] + 0.5 * energy_scores[candidate]
                    highest_candidate = candidate
            
            ms_placement[ms, highest_candidate] = 1
            
            # Update resources
            edge_network.nodes[highest_candidate]['available_cpu'] -= cpu_demand
            edge_network.nodes[highest_candidate]['available_memory'] -= memory_demand
        
            
        # Data placement
        for ms in self.sim.microservice_graph.nodes:
            disk_demand = self.sim.microservice_graph.nodes[ms]['requested_disk']
            placed_on = ms_placement.argmax(axis=1)[ms]
            k = 0
            while k < 5 and disk_demand > 0: # TODO: To be changed (k)
                k_hop_neighbors = find_k_hop_neighbors(edge_network, placed_on, k)
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

