import numpy as np
import networkx as nx
import random
import cvxpy as cp
import matplotlib.pyplot as plt
from pyomo.environ import *


def softmax(matrix, axis):
    exp_matrix = np.exp(matrix - np.max(matrix, axis=axis, keepdims=True))
    return exp_matrix / np.sum(exp_matrix, axis=axis, keepdims=True)

class EdgeNetworkOptimization:
    def __init__(self, microservice_graph, edge_network):
        self.microservice_graph = microservice_graph
        self.edge_network = edge_network
        
        self.num_microservices = len(self.microservice_graph.nodes)
        self.num_edge_servers = len(self.edge_network.nodes)
        


    def solve(self):
        model = ConcreteModel()
        model.microservices = RangeSet(1, self.num_microservices)  # 1-based indexing
        model.edge_servers = RangeSet(1, self.num_edge_servers)  # 1-based indexing

        # Variables
        model.x = Var(model.microservices, model.edge_servers, domain=Binary)
        model.y = Var(model.microservices, model.edge_servers, domain=NonNegativeReals)

        def cost(model):
            L_ms = 0
            L_vol = 0

            for m1, m2, data in self.microservice_graph.edges(data=True):
                for n1 in model.edge_servers:
                    for n2 in model.edge_servers:
                        path_cost = 0
                        if n1 != n2:
                            path = nx.shortest_path(self.edge_network, source=n1 - 1, target=n2 - 1, weight='reverse_bw')
                            path_cost = sum(1 / self.edge_network[u][v]['bw'] for u, v in zip(path[:-1], path[1:]))
                        L_ms += data['bw_min'] * model.x[m1 + 1, n1] * model.x[m2 + 1, n2] * path_cost

            for m in model.microservices:
                for n1 in model.edge_servers:
                    for n2 in model.edge_servers:
                        path_cost = 0
                        if n1 != n2:
                            path = nx.shortest_path(self.edge_network, source=n1 - 1, target=n2 - 1, weight='reverse_bw')
                            path_cost = sum(1 / self.edge_network[u][v]['bw'] for u, v in zip(path[:-1], path[1:]))
                        L_vol += model.x[m, n1] * model.y[m, n2] * path_cost

            return L_ms + L_vol

        model.objective = Objective(rule=cost, sense=minimize)

        # Constraints
        def placement_constraint(model, m):
            return sum(model.x[m, n] for n in model.edge_servers) == 1

        def resource_constraint_cpu(model, n):
            return sum(model.x[m, n] * self.microservice_graph.nodes[m - 1]['cpu'] for m in model.microservices) <= self.edge_network.nodes[n - 1]['available_cpu']

        def resource_constraint_memory(model, n):
            return sum(model.x[m, n] * self.microservice_graph.nodes[m - 1]['memory'] for m in model.microservices) <= self.edge_network.nodes[n - 1]['available_memory']

        def disk_constraint(model, n):
            return sum(model.y[m, n] for m in model.microservices) <= self.edge_network.nodes[n - 1]['available_disk']

        def disk_demand_constraint(model, m):
            return sum(model.y[m, n] for n in model.edge_servers) == self.microservice_graph.nodes[m - 1]['disk']

        model.placement_constraints = Constraint(model.microservices, rule=placement_constraint)
        model.cpu_constraints = Constraint(model.edge_servers, rule=resource_constraint_cpu)
        model.memory_constraints = Constraint(model.edge_servers, rule=resource_constraint_memory)
        model.disk_constraints = Constraint(model.edge_servers, rule=disk_constraint)
        model.disk_demand_constraints = Constraint(model.microservices, rule=disk_demand_constraint)

        # Solve the model
        solver = SolverFactory('gurobi')  # Use an appropriate solver
        result = solver.solve(model, tee=True)

        if result.solver.status == SolverStatus.ok and result.solver.termination_condition == TerminationCondition.optimal:
            print("Optimization successful!")
            placement = [[model.x[m, n].value for n in model.edge_servers] for m in model.microservices]
            disk_alloc = [[model.y[m, n].value for n in model.edge_servers] for m in model.microservices]
            return placement, disk_alloc
        else:
            print("Optimization failed:", result.solver.status)
            return None, None


    def search_with_pso(self, num_particles=50, max_iterations=100, w=0.5, c1=1.5, c2=1.5):
        particles = []
        velocities = []
        p_best = [] # best known poistion
        p_best_scores = []
        g_best = None # global best position
        g_best_score = float('inf')
        def fitness(x, y):
            # print(x.shape)
            x = (x >= np.max(x, axis=1).reshape(-1, 1).repeat(self.num_edge_servers, axis=1)).astype(int)
            # print(x)
            L_ms = 0
            L_vol = 0

            for m1, m2, data in self.microservice_graph.edges(data=True):
                for n1 in range(self.num_edge_servers):
                    for n2 in range(self.num_edge_servers):
                        path_cost = 0
                        if n1 != n2:
                            path = nx.shortest_path(self.edge_network, source=n1, target=n2, weight='reverse_bw')
                            path_cost = sum(1 / self.edge_network[u][v]['bw'] for u, v in zip(path[:-1], path[1:]))
                        L_ms += data['bw_min'] * x[m1, n1] * x[m2, n2] * path_cost

            for m in range(self.num_microservices):
                for n1 in range(self.num_edge_servers):
                    for n2 in range(self.num_edge_servers):
                        path_cost = 0
                        if n1 != n2:
                            path = nx.shortest_path(self.edge_network, source=n1, target=n2, weight='reverse_bw')
                            path_cost = sum(1 / self.edge_network[u][v]['bw'] for u, v in zip(path[:-1], path[1:]))
                        L_vol += x[m, n1] * y[m, n2] * path_cost


            constraint_violations = 0

            # Placement constraint: Each microservice must be assigned to one edge server
            for m in range(self.num_microservices):
                if np.sum(x[m, :]) != 1:
                    constraint_violations += 1

            # Resource constraints per edge server
            for n in range(self.num_edge_servers):
                cpu_usage = sum(x[m, n] * self.microservice_graph.nodes[m]['cpu'] for m in range(self.num_microservices))
                memory_usage = sum(x[m, n] * self.microservice_graph.nodes[m]['memory'] for m in range(self.num_microservices))
                disk_usage = sum(x[m, n] * self.microservice_graph.nodes[m]['disk'] for m in range(self.num_microservices))

                if cpu_usage > self.edge_network.nodes[n]['available_cpu']:
                    constraint_violations += 1
                if memory_usage > self.edge_network.nodes[n]['available_memory']:
                    constraint_violations += 1
                if disk_usage > self.edge_network.nodes[n]['available_disk']:
                    constraint_violations += 1

            # Disk demand constraint
            for m in range(self.num_microservices):
                if np.sum(y[m, :]) != self.microservice_graph.nodes[m]['disk']:
                    constraint_violations += 1

            penalty = 100 * constraint_violations

            return L_ms + L_vol + penalty

        for _ in range(num_particles):
            x = np.random.dirichlet(np.ones(self.num_edge_servers), size=self.num_microservices).T
            # x = np.random.randint(0, 2, (self.num_microservices, self.num_edge_servers))
            # What is the min and max allocatable disk ?
            min_vol = min(nx.get_node_attributes(microservice_graph, 'disk').values())
            max_vol = max(nx.get_node_attributes(microservice_graph, 'disk').values())
            y = np.random.uniform(min_vol, max_vol, (self.num_microservices, self.num_edge_servers))
            # print(y)
            
            particles.append([x, y])
            velocities.append([np.zeros_like(x), np.zeros_like(y)])
            # velocities.append(y)
            p_best.append([x, y])
            p_best_scores.append(fitness(x.T, y))

            if p_best_scores[-1] < g_best_score:
                g_best = [x, y]
                g_best_score = p_best_scores[-1]

        for iteration in range(max_iterations):
            for i in range(num_particles):
                # Update velocity
                r1 = np.random.rand(self.num_edge_servers, self.num_microservices)
                r2 = np.random.rand(self.num_edge_servers, self.num_microservices)
                velocities[i][0] = w * velocities[i][0] + c1 * r1 * (p_best[i][0] - particles[i][0]) + c2 * r2 * (g_best[0] - particles[i][0])


                velocities[i][1] = w * velocities[i][1] + c1 * r1.T * (p_best[i][1] - particles[i][1]) + c2 * r2.T * (g_best[1] - particles[i][1])


                # Update position
                particles[i][0] = particles[i][0] + velocities[i][0]
                particles[i][1] = particles[i][1] + velocities[i][1]
                
                # print(f'Iteration {iteration}, particle {i}, {particles[i][0].shape}')
                

                # # Ensure feasibility
                particles[i][0] = softmax(particles[i][0], axis=0)
                particles[i][1] = np.clip(particles[i][1], min_vol, max_vol)
                # print(particles[i].shape)

                # Evaluate fitness
                # print(f'Iteration {iteration}, particle {i}, {particles[i].shape}')
                score = fitness(particles[i][0].T, particles[i][1])
                if score < p_best_scores[i]:
                    p_best[i] = particles[i]
                    p_best_scores[i] = score

                    if score < g_best_score:
                        g_best = particles[i]
                        g_best_score = score

            print(f"Iteration {iteration+1}/{max_iterations}, Best Score: {g_best_score}")

            # Return best solution
        return g_best[0].T, g_best[1]

# edge_network = nx.Graph()

num_microservices = 6
num_edge_servers = 3

microservice_graph = nx.DiGraph()
edge_network = nx.erdos_renyi_graph(num_edge_servers, 0.5, seed=123, directed=False)

for m in range(num_microservices):
    microservice_graph.add_node(m, cpu=random.randint(100, 500), memory=random.randint(512, 1024), disk=random.randint(1, 100))
for _ in range(num_microservices * 2):
    m1, m2 = random.sample(range(num_microservices), 2)
    bandwidth = random.randint(1, 10)
    microservice_graph.add_edge(m1, m2, bw_min=bandwidth)
    
for n in range(num_edge_servers):
    # Update node attr
    edge_network.add_node(n, available_cpu=random.randint(1000, 4000), available_memory=random.randint(4096, 2 * 4096), available_disk=random.randint(100, 500))

for n1, n2 in edge_network.edges():
    # n1, n2 = random.sample(range(num_edge_servers), 2)
    # Update edge attr
    bandwidth = random.randint(10, 100)
    edge_network.add_edge(n1, n2, bw=bandwidth, reverse_bw=1/bandwidth)
    

# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# nx.draw(microservice_graph, with_labels=True, node_color='skyblue')
# plt.title("Microservice Graph")

# plt.subplot(1, 2, 2)
# pos = nx.spring_layout(edge_network)
# nx.draw(edge_network, pos, with_labels=True, node_color='green')
# edge_attr = nx.get_edge_attributes(edge_network, 'bw')
# nx.draw_networkx_edge_labels(edge_network, pos, edge_labels=edge_attr)
# plt.title("Edge Graph")

# plt.show()

sim = EdgeNetworkOptimization(microservice_graph, edge_network)
placement, disk_allocation = sim.search_with_pso(max_iterations=100)

placement = (placement >= np.max(placement, axis=1).reshape(-1, 1).repeat(num_edge_servers, axis=1)).astype(int)


placement = (placement >= np.max(placement, axis=1).reshape(-1, 1).repeat(num_edge_servers, axis=1)).astype(int)
if placement is not None:
    print("Placement Matrix:")
    print(placement)

    print("\nDisk Allocation Matrix:")
    print(disk_allocation)
    

# Exact method
placement, disk_alloc = sim.solve()

if placement is not None:
    print("Placement Matrix:")
    print(placement)

    print("\nDisk Allocation Matrix:")
    print(disk_allocation)