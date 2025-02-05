import time
import numpy as np

# Define the RPO function
def RPO(search_space, objective_function, lb, ub, iterations):
    # Initialize the best solution found so far
    best_solution = None
    best_fitness = float('inf')

    convergence_curve = np.zeros((iterations, 1))
    # Initialize the population randomly within the search space
    population, dimensions = search_space.shape[0], search_space.shape[1]

    ct = time.time()
    for i in range(iterations):
        # Evaluate the fitness of each piranha
        fitness_values = [objective_function(individual) for individual in population]

        # Find the best piranha in the current population
        current_best_index = np.argmin(fitness_values)
        current_best_fitness = fitness_values[current_best_index]
        current_best_solution = population[current_best_index]

        # Update the global best solution if necessary
        if current_best_fitness < best_fitness:
            best_solution = current_best_solution
            best_fitness = current_best_fitness

        # Update the position of each piranha
        for j in range(len(population)):
            # Calculate the velocity vector
            velocity = np.random.uniform(-1, 1, (dimensions,))

            # Update the position of the piranha
            population[j] += velocity

            # Ensure the piranha stays within the search space
            population[j] = np.clip(population[j], search_space[0], search_space[1])
        convergence_curve[i] = population
    ct = time.time() - ct
    return best_solution, convergence_curve, best_fitness, ct

