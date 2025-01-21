from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import time
from part_1 import *

# Experiment pre jednu kombináciu
def run_experiment(func, alg, alg_name, dim, pop_size, f_evals, repeat):
    """
    Spustí jeden experiment pre danú kombináciu funkcie, algoritmu, dimenzie a počtu opakovaní.
    """
    fitness_values = []
    for _ in range(repeat):
        if alg_name == "PSO":
            best_solution, best_fitness = alg(func, dim=dim, num_particles=pop_size, max_iter=f_evals // pop_size)
        else:
            best_solution, best_fitness = alg(func, bounds=[BOUNDS] * dim, popsize=pop_size, its=f_evals // pop_size)
        fitness_values.append(best_fitness)
    return (func.__name__, alg_name, dim, np.mean(fitness_values), np.std(fitness_values))


# Paralelné spúšťanie experimentov
def parallel_experiment():
    """
    Paralelne spúšťa experimenty pre všetky kombinácie funkcií, dimenzií a algoritmov.
    """
    dim_sizes = [2, 10, 20]
    pop_sizes = {2: 10, 10: 20, 20: 40}
    num_repeats = 20
    f_evals_multiplier = 2000
    algorithms = {
        "DE_rand": de_rand,
        "DE_best": de_best,
        "PSO": pso,
    }
    testing_functions = [
        rastrigin_function,
        ackley_function,
        sphere_function,
        rosenbrock_function,
        griewank_function,
    ]

    results = []

    # Paralelizácia
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=12) as executor:  # 12 vlákien
        futures = []
        for func in testing_functions:
            for dim in dim_sizes:
                pop_size = pop_sizes[dim]
                f_evals = f_evals_multiplier * dim
                for alg_name, alg in algorithms.items():
                    futures.append(
                        executor.submit(run_experiment, func, alg, alg_name, dim, pop_size, f_evals, num_repeats)
                    )

        # Zber výsledkov
        for future in as_completed(futures):
            results.append(future.result())

    end_time = time.time()
    print(f"Experiment trval {end_time - start_time:.2f} sekúnd.")

    # Výpis výsledkov
    for result in results:
        func_name, alg_name, dim, mean_fitness, std_fitness = result
        print(f"Funkcia: {func_name}, Algoritmus: {alg_name}, Dimenzia: {dim}, "
              f"Priemerné fitness: {mean_fitness:.4f}, Štandardná odchýlka: {std_fitness:.4f}")

    return results

parallel_experiment()