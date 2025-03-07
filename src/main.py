from solvers.sim import EdgeNetworkOptimization
from solvers.rank_based import GraphCentralityRankBasedHeuristic
from solvers.milp import ExactSolver
from graph.random import RandomMicroserviceEdgeEnvironment
import random

seed = 2
random.seed(2)

env = RandomMicroserviceEdgeEnvironment(num_microservices=10, num_edge_servers=10, block_size=1)
microservice_graph = env.microservice_graph
edge_network = env.edge_network

sim =  EdgeNetworkOptimization(microservice_graph, edge_network)


solver = ExactSolver(sim)

ms_placement, disk_placement = solver.solve()

cost = sim.compute_placement_cost(ms_placement, disk_placement, verbose=True)
print(cost)