
import numpy as np
import networkx as nx
from pyomo.environ import *

class ExactSolver:
    def __init__(self, sim):
        self.sim = sim
        
        self.microservice_graph = self.sim.microservice_graph
        self.edge_network = self.sim.edge_network

        self.num_microservices = len(self.microservice_graph.nodes)
        self.num_edge_servers = len(self.edge_network.nodes)

    def solve(self, verbose=True):
        model = ConcreteModel()
        model.microservices = RangeSet(1, self.num_microservices)  # 1-based indexing
        model.edge_servers = RangeSet(1, self.num_edge_servers)  # 1-based indexing

        # Variables
        model.x = Var(model.microservices, model.edge_servers, domain=Binary)
        model.y = Var(model.microservices, model.edge_servers, domain=NonNegativeIntegers)
        

        def cost(model):
            L_ms = 0
            L_vol = 0
            L_energy = 0

            for m1, m2, data in self.microservice_graph.edges(data=True):
                cpu_demands_m1 = self.microservice_graph.nodes[m1]["requested_cpu"]
                cpu_demands_m2 = self.microservice_graph.nodes[m2]["requested_cpu"]
                min_bw = self.microservice_graph.edges[(m1, m2)]["bw_min"]
                for n1 in model.edge_servers:
                    for n2 in model.edge_servers:
                        path_cost = 0
                        # bw_uv = 0
                        if n1 != n2:
                            # Shortest paths can be precomputed in order to reduce time complexity
                            # Howver it would increase space complexity
                            try:
                                path = nx.shortest_path(self.edge_network, source=n1 - 1, target=n2 - 1, weight='reverse_bw')
                                # path_cost = sum(1 / self.edge_network[u][v]['bw'] for u, v in zip(path[:-1], path[1:]))
                                # path_cost = nx.shortest_path_length(self.edge_network, source=n1 - 1, target=n2 - 1, weight='reverse_bw')
                                for (u, v) in zip(path[:-1], path[1:]):
                                    # path_cost += min_bw / (self.edge_network.edges[(u, v)]['bw'] + 0.0001)
                                    # print(self.edge_network.edges[(u, v)]['bw'])
                                    path_cost += min_bw / self.edge_network.edges[(u, v)]['bw']
                                    # self.edge_network.edges[(u, v)]['bw'] = max(0.1, self.edge_network.edges[(u, v)]['bw'] - min_bw)
                                    # self.edge_network.edges[(u, v)]['reverse_bw'] = 1 / self.edge_network.edges[(u, v)]['bw']
                                    # print(self.edge_network.edges[(u, v)]['bw'])
                                    
                            except:
                                # path_cost += model.x[m1 + 1, n1] * model.x[m2 + 1, n2] * 1000
                                pass
                        
                        computation_cost = model.x[m1 + 1, n1] * cpu_demands_m1 / self.edge_network.nodes[n1-1]["cpu_frequency"] + model.x[m2 + 1, n2] * cpu_demands_m2 / self.edge_network.nodes[n2-1]["cpu_frequency"]
                        
                        energy_cost = model.x[m1 + 1, n1] * self.edge_network.nodes[n1-1]["energy_per_cpu_unit"] * cpu_demands_m1 + model.x[m2 + 1, n2] * self.edge_network.nodes[n2-1]["energy_per_cpu_unit"] * cpu_demands_m2
                        communication_cost = model.x[m1 + 1, n1] * model.x[m2 + 1, n2] * path_cost
                        L_ms +=  communication_cost + computation_cost
                        L_energy += energy_cost
                        
            for m in model.microservices:
                data_bw = self.microservice_graph.nodes[m - 1]["data_bw"]
                for n1 in model.edge_servers:
                    for n2 in model.edge_servers:
                        path_cost = 0
                        path = None
                        if n1 != n2:
                            try:
                                path = nx.shortest_path(self.edge_network, source=n1-1, target=n2-1, weight='reverse_bw')
                                for (u, v) in zip(path[:-1], path[1:]):
                                    # path_cost += min_bw / (self.edge_network.edges[(u, v)]['bw'] + 0.0001)
                                    path_cost += data_bw / self.edge_network.edges[(u, v)]['bw']
                                    # self.edge_network.edges[(u, v)]['bw'] = max(0.1, self.edge_network.edges[(u, v)]['bw'] - data_bw)
                                    # self.edge_network.edges[(u, v)]['reverse_bw'] = 1 / self.edge_network.edges[(u, v)]['bw']
                            except:
                                path_cost = 0
                            
                        L_vol += model.x[m, n1] * model.y[m, n2] * path_cost
            # print(L_ms+L_vol)
            return L_ms + L_vol + L_energy

        model.objective = Objective(rule=cost, sense=minimize)

        # Constraints
        def placement_constraint(model, m):
            return sum(model.x[m, n] for n in model.edge_servers) == 1

        def resource_constraint_cpu(model, n):
            return sum(model.x[m, n] * self.microservice_graph.nodes[m - 1]['requested_cpu'] for m in model.microservices) <= self.edge_network.nodes[n - 1]['available_cpu']

        def resource_constraint_memory(model, n):
            return sum(model.x[m, n] * self.microservice_graph.nodes[m - 1]['requested_memory'] for m in model.microservices) <= self.edge_network.nodes[n - 1]['available_memory']

        def disk_constraint(model, n):
            return sum(model.y[m, n] for m in model.microservices) <= self.edge_network.nodes[n - 1]['available_disk']

        def disk_demand_constraint(model, m):
            return sum(model.y[m, n] for n in model.edge_servers) == self.microservice_graph.nodes[m - 1]['requested_disk']
        
        # def non_negativity_constraint(model, m1, m2, n1, n2):
        #     return model.z[m1, m2, n1, n2] >= 1
        
        # def bandwidth_capacity_constraint(model, n1, n2):
        #     if n1 != n2:
        #         try:
        #             return sum(model.z[m1 + 1, m2 + 1, n1, n2] for m1, m2 in self.microservice_graph.edges()) <= self.edge_network.edges[(n1 - 1, n2 - 1)]['bw']
        #         except:
        #             return Constraint.Skip
                    
        #     else:
        #         return Constraint.Skip
        
        # def min_bandwidth_constraint(model, m1, m2):
        #     c = True
        #     if m1 != m2:
        #         for n1 in model.edge_servers:
        #             for n2 in model.edge_servers:
        #                 if n1 != n2:
        #                     try:
        #                         path = nx.shortest_path(self.edge_network, source=n1 - 1, target=n2 - 1, weight='reverse_bw')
        #                         for (u, v) in zip(path[:-1], path[1:]):
        #                             c = c and self.edge_network.edges[(u, v)]['bw'] >= self.microservice_graph.edges[(m1-1, m2-1)] and model.x[m1, n1] > 0 and model.x[m2, n1]
        #                     except:
        #                         pass
        #         return Constraint.Feasible if c else Constraint.Infeasible
        #     else:
        #         return Constraint.Skip


        model.placement_constraints = Constraint(model.microservices, rule=placement_constraint)
        model.cpu_constraints = Constraint(model.edge_servers, rule=resource_constraint_cpu)
        model.memory_constraints = Constraint(model.edge_servers, rule=resource_constraint_memory)
        model.disk_constraints = Constraint(model.edge_servers, rule=disk_constraint)
        model.disk_demand_constraints = Constraint(model.microservices, rule=disk_demand_constraint)
        
        # model.non_negativity_constraints = Constraint(model.microservices, model.microservices, model.edge_servers, model.edge_servers, rule=non_negativity_constraint)
        # model.bandwidth_capacity_constraints = Constraint(model.edge_servers, model.edge_servers, rule=bandwidth_capacity_constraint)
        # model.min_bandwidth_constraints = Constraint(model.microservices, model.microservices, rule=min_bandwidth_constraint)
        
        # Solve the model usnig Gurobi solver
        solver = SolverFactory('gurobi')
        result = solver.solve(model, tee=verbose)

        if result.solver.status == SolverStatus.ok and result.solver.termination_condition == TerminationCondition.optimal:
            print("Optimization successful!", value(model.objective))
            placement = [[model.x[m, n].value for n in model.edge_servers] for m in model.microservices]
            disk_alloc = [[model.y[m, n].value for n in model.edge_servers] for m in model.microservices]
            #bw_alloc = [[model.z[m1, m2, n1, n2].value for n1, n2 in zip(model.edge_servers, model.edge_servers)] for m1, m2 in zip(model.microservices, model.microservices)]
            # bw_alloc = np.zeros((self.num_microservices, self.num_microservices, self.num_edge_servers, self.num_edge_servers))
    
            # for m1 in model.microservices:
            #     for m2 in model.microservices:
            #         for n1 in model.edge_servers:
            #             for n2 in model.edge_servers:
            #                 bw_alloc[m1 - 1, m2 - 1, n1 - 1, n2 - 1] = model.z[m1, m2, n1, n2].value
            #                 print(model.z[m1, m2, n1, n2].value)
            return np.array(placement), np.array(disk_alloc)
        else:
            print("Optimization failed:", result.solver.status)
            return None, None