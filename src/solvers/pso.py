
import numpy as np
import networkx as nx
import cvxpy as cp
import matplotlib.pyplot as plt
from pyomo.environ import *
from collections import deque
from copy import deepcopy
from .utils import find_k_hop_neighbors


class PSOSolver:
    def __init__(self, sim, num_particles=50, generations=100, w=0.5, c1=1.5, c2=1.5):
        """
        Initialize the PSO solver.

        :param sim: Simulation object
        :param num_particles: Number of particles in the swarm
        :param generations: Number of iterations
        :param w: Inertia weight
        :param c1: Cognitive parameter (local best influence)
        :param c2: Social parameter (global best influence)
        """
        self.sim = sim
        self.num_particles = num_particles
        self.generations = generations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.num_microservices = sim.get_num_microservices()
        self.num_edge_servers = sim.get_num_edge_servers()

    def initialize_particles(self):
        """
        Initialize the particles' positions and velocities.
        """
        positions = np.random.randint(0, self.num_edge_servers, (self.num_particles, self.num_microservices))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.num_microservices))
        return positions, velocities

    def fitness(self, particle):
        """
        Compute the fitness of a particle.
        """
        x = np.zeros((self.num_microservices, self.num_edge_servers))
        for m, n in enumerate(particle):
            x[m, n] = 1

        y = self.allocate_data(x)
        cost = self.compute_cost(x, y)
        return cost

    def allocate_data(self, x):
        """
        Allocate data using the heuristic.
        """
        y = np.zeros((self.num_microservices, self.num_edge_servers))
        edge_network = deepcopy(self.sim.edge_network)
        for ms in self.sim.microservice_graph.nodes:
            disk_demand = self.sim.microservice_graph.nodes[ms]['requested_disk']
            placed_on = x.argmax(axis=1)[ms]
            k = 0
            while k < 5 and disk_demand > 0:  # TODO: To be changed (k)
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
        """
        Compute the cost of a solution.
        """
        return self.sim.compute_placement_cost(x, y)

    def solve(self):
        """
        Run the PSO algorithm.
        """
        # Initialize particles
        positions, velocities = self.initialize_particles()
        personal_best_positions = positions.copy()
        personal_best_scores = np.array([self.fitness(p) for p in positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        best_fitness_hist = deque(maxlen=5)

        for generation in range(self.generations):
            for i in range(self.num_particles):
                # Update velocity
                r1, r2 = np.random.rand(2)
                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * r1 * (personal_best_positions[i] - positions[i])
                    + self.c2 * r2 * (global_best_position - positions[i])
                )

                # Update position
                positions[i] = np.clip(positions[i] + velocities[i], 0, self.num_edge_servers - 1).astype(int)

                # Evaluate fitness
                fitness_score = self.fitness(positions[i])

                # Update personal best
                if fitness_score < personal_best_scores[i]:
                    personal_best_scores[i] = fitness_score
                    personal_best_positions[i] = positions[i]

                # Update global best
                if fitness_score < global_best_score:
                    global_best_score = fitness_score
                    global_best_position = positions[i]

            print(f"Generation {generation + 1}: Best Fitness = {global_best_score}")

            # Early stopping condition
            best_fitness_hist.append(global_best_score)
            if len(best_fitness_hist) == 3 and len(set(best_fitness_hist)) == 1:
                break

        # Return the best solution
        x = np.zeros((self.num_microservices, self.num_edge_servers))
        for m, n in enumerate(global_best_position):
            x[m, n] = 1
        return x, self.allocate_data(x)