import numpy as np

class KnapsackPSO:
    def __init__(self, values, weights, capacity, num_particles=30, max_iter=100, w=0.5, c1=1.5, c2=1.5):
        self.values = np.array(values)
        self.weights = np.array(weights)
        self.capacity = capacity
        self.num_items = len(values)
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def fitness(self, particle):
        total_value = np.sum(self.values * particle)
        total_weight = np.sum(self.weights * particle)
        if total_weight > self.capacity:
            return -1  # Penalize infeasible solutions
        return total_value

    def solve(self):
        # Initialize particles and velocities
        particles = np.random.randint(2, size=(self.num_particles, self.num_items))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.num_items))
        p_best = particles.copy()
        p_best_scores = np.array([self.fitness(p) for p in particles])
        g_best = p_best[np.argmax(p_best_scores)]
        g_best_score = np.max(p_best_scores)

        # PSO iterations
        for iteration in range(self.max_iter):
            for i in range(self.num_particles):
                # Update velocity
                r1 = np.random.rand(self.num_items)
                r2 = np.random.rand(self.num_items)
                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * r1 * (p_best[i] - particles[i])
                    + self.c2 * r2 * (g_best - particles[i])
                )

                # Update particle position using sigmoid and threshold
                sigmoid = 1 / (1 + np.exp(-velocities[i]))
                particles[i] = np.where(np.random.rand(self.num_items) < sigmoid, 1, 0)

                # Evaluate fitness
                fitness_value = self.fitness(particles[i])
                if fitness_value > p_best_scores[i]:
                    p_best[i] = particles[i]
                    p_best_scores[i] = fitness_value

                if fitness_value > g_best_score:
                    g_best = particles[i]
                    g_best_score = fitness_value

            print(f"Iteration {iteration + 1}, Best Fitness: {g_best_score}")

        return g_best, g_best_score

# Example usage
values = [60, 100, 120]
weights = [10, 20, 30]
capacity = 50

knapsack = KnapsackPSO(values, weights, capacity, num_particles=50, max_iter=100)
solution, max_value = knapsack.solve()

print("Selected items (binary representation):", solution)
print("Maximum value:", max_value)
