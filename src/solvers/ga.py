import numpy as np
import networkx as nx
import cvxpy as cp
import matplotlib.pyplot as plt
from pyomo.environ import *
from collections import deque
from copy import deepcopy
from .utils import find_k_hop_neighbors


class GASolver:
    def __init__(self, sim, population_size=50, generations=100, mutation_rate=0.1, crossover_rate=0.8):
        self.sim = sim
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_microservices = sim.get_num_microservices()
        self.num_edge_servers = sim.get_num_edge_servers()
    
    def initialize_population(self):
        return [np.random.randint(0, self.num_edge_servers, self.num_microservices) for _ in range(self.population_size)]

    def fitness(self, chromosome):
        x = np.zeros((self.num_microservices, self.num_edge_servers))
        for m, n in enumerate(chromosome):
            x[m, n] = 1

        # TODO: Use heuristic for data allocation
        y = self.allocate_data(x)

        # compute cost
        cost = self.compute_cost(x, y)

        return cost

    def allocate_data(self, x):
        y = np.zeros((self.num_microservices, self.num_edge_servers))
        edge_network = deepcopy(self.sim.edge_network)
        for ms in self.sim.microservice_graph.nodes:
            disk_demand = self.sim.microservice_graph.nodes[ms]['requested_disk']
            placed_on = x.argmax(axis=1)[ms]
            k = 0
            while k < 5 and disk_demand > 0: # TODO: To be changed (k)
                k_hop_neighbors = find_k_hop_neighbors(self.sim.edge_network, placed_on, k)
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
                        y[ms, neighbor] = allocation
                k += 1
        return y

    def compute_cost(self, x, y):
        return self.sim.compute_placement_cost(x, y)


    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, self.num_microservices - 1)
            child1 = np.concatenate((parent1[:point], parent2[point:]))
            child2 = np.concatenate((parent2[:point], parent1[point:]))
            return child1, child2
        return parent1, parent2

    def mutate(self, chromosome):
        if np.random.rand() < self.mutation_rate:
            m = np.random.randint(self.num_microservices)
            chromosome[m] = np.random.randint(self.num_edge_servers)
        return chromosome

    def solve(self):
        population = self.initialize_population()
        best_solution = None
        best_fitness = float('inf')
        best_fitness_hist = deque(maxlen=5)
    
        for generation in range(self.generations):
            fitness_scores = [self.fitness(chromosome) for chromosome in population]
            fitness_scores = np.array(fitness_scores)

            # Get sorted indices
            sorted_indices = np.argsort(fitness_scores)

            # Sort the population by fitness from lowest cost to highest cost
            sorted_population = [population[i] for i in sorted_indices]
            
            best_solution = sorted_population[0]
            best_fitness = min(fitness_scores)
    
            # Prepare the next generation
            next_generation = []
    
            while len(next_generation) < self.population_size:
                # Select parents
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
    
                # Crossover
                child1, child2 = self.crossover(parent1, parent2)
    
                # Mutate
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
    
                # Add offspring to the next generation
                next_generation.extend([child1, child2])
    
            # Update the population
            population = next_generation[:self.population_size]
            print(f"Generation {generation+1}: Best Fitness = {best_fitness}")
            
            best_fitness_hist.append(best_fitness)
            if len(best_fitness_hist) == 3 and len(set(best_fitness_hist)) == 1:
                break

        x = np.zeros((self.num_microservices, self.num_edge_servers))
        for m, n in enumerate(best_solution):
            x[m, n] = 1
        return x, self.allocate_data(x)

        
    def tournament_selection(self, population, fitness_scores, tournament_size=3):
        """Select a parent using tournament selection."""
        participants = np.random.choice(len(population), tournament_size, replace=False)
        best_participant = min(participants, key=lambda i: fitness_scores[i])
        return population[best_participant]
