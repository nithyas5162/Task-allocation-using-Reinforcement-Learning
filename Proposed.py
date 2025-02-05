import time
import numpy as np


# Proposed -> Water Strider Algorithm (WSA) + Billiards-Inspired Optimization (BIO) -> Update starts from line no 32

def Proposed(population, fobj, VRmin, VRmax, Max_iter):
    N, dim = population.shape[0], population.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]
    best_solution = None
    best_fitness = np.inf

    Convergence_curve = np.zeros((Max_iter, 1))

    t = 0
    ct = time.time()
    for t in range(Max_iter):
        # Evaluate fitness for each individual in the population
        fitness_values = np.apply_along_axis(fobj, 1, population)

        # Update the best solution
        min_fitness_index = np.argmin(fitness_values)
        if fitness_values[min_fitness_index] < best_fitness:
            best_fitness = fitness_values[min_fitness_index]
            best_solution = population[min_fitness_index]

        # Update the population using the water strider algorithm
        for i in range(N):
            for j in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()
                if t < Max_iter/2:
                    population[i, j] = population[i, j] + r1 * (best_solution[j] - population[i, j]) + r2 * (
                            population[min_fitness_index, j] - population[i, j])
                else:
                    Flag4ub = population[i, :] > ub
                    Flag4lb = population[i, :] < lb
                    population[i, :] = (population[i, :] * (~(Flag4ub + Flag4lb))) + ub * Flag4ub + lb * Flag4lb

        Convergence_curve[t] = best_solution
        t = t + 1
    best_solution = Convergence_curve[Max_iter - 1][0]
    ct = time.time() - ct

    return best_solution, Convergence_curve, best_fitness, ct
