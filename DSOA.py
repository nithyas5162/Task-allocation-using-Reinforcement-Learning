import time
import numpy as np


def sphere_function(x):
    return np.sum(x ** 2)


def DSOA(positions, objective_func, lb, ub, max_iter):
    num_doves, dim = positions.shape  # Number of doves (population size)

    # Evaluate initial positions
    fitness_values = np.apply_along_axis(objective_func, 1, positions)
    convergence_curve = np.zeros((max_iter, 1))
    # Find the best dove (global best)
    best_position = positions[np.argmin(fitness_values)]
    best_fitness = np.min(fitness_values)
    ct = time.time()
    for t in range(max_iter):
        # Update dove positions based on the flocking behavior
        # Compute the new positions of doves based on their flocking behavior
        new_positions = np.zeros_like(positions)

        for i in range(num_doves):
            # Update position of each dove
            rand_dove_index = np.random.randint(0, num_doves)
            new_positions[i] = positions[i] + np.random.uniform() * (best_position - positions[i]) \
                               + np.random.uniform() * (positions[rand_dove_index] - positions[i])

            # Ensure the new positions are within the search space bounds
            new_positions[i] = np.clip(new_positions[i], lb, ub)

        # Evaluate new positions
        new_fitness_values = np.apply_along_axis(objective_func, 1, new_positions)

        # Update the best dove (global best)
        min_index = np.argmin(new_fitness_values)
        if new_fitness_values[min_index] < best_fitness:
            best_fitness = new_fitness_values[min_index]
            best_position = new_positions[min_index]

        # Update the positions and fitness values for the next iteration
        positions = new_positions
        fitness_values = new_fitness_values
        convergence_curve[t] = new_positions
    ct = time.time() - ct
    return best_fitness, best_position, convergence_curve, ct



