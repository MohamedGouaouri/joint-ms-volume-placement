from solvers.sim import EdgeNetworkOptimization
from solvers.rank_based import GraphCentralityRankBasedHeuristic
from solvers.ga import GASolver
from solvers.pso import PSOSolver
from solvers.milp import ExactSolver
from solvers.top_sort import TopologicalSortingHeuristic

from graph.random import RandomMicroserviceEdgeEnvironment
from graph.smart_home import SmartHomeMicroserviceGraph
from graph.smart_traffic import SmartTrafficMicroserviceGraph
import time
import random
import csv

seed = 2
random.seed(2)

env = RandomMicroserviceEdgeEnvironment(num_microservices=30, num_edge_servers=50, block_size=1)
# env = SmartTrafficMicroserviceGraph()
sim =  EdgeNetworkOptimization(env.microservice_graph, env.edge_network)

solvers = [ExactSolver(sim), GraphCentralityRankBasedHeuristic(sim), TopologicalSortingHeuristic(sim)]
# solvers = [GASolver(sim)]
experiment_name = f"random_ms_{env.num_microservices}_es_{env.num_edge_servers}"
with open(f'{experiment_name}.csv', mode='a') as file:
    # TODO: add header row
    writer = csv.writer(file)
    for solver in solvers:
        st = time.time()
        ms_placement, disk_placement = solver.solve(verbose=False)
        et = time.time()
        total_cost, compute_cost, energy_cost, data_cost, penalty = sim.compute_placement_cost(ms_placement, disk_placement, verbose=True)
        # Create a csv file to store the results
        writer.writerow([solver.__class__.__name__, total_cost, compute_cost, energy_cost, data_cost, penalty, et-st])