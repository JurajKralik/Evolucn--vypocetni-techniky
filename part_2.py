from scipy.stats import friedmanchisquare
from part_1 import *

BOUNDS = [-100, 100]


def friedman_test(results):
    test_stat, p_value = friedmanchisquare(*results)
    print(f"Friedman test statistic: {test_stat}, p-value: {p_value}")
    if p_value < 0.05:
        print("Výsledky sú štatisticky významné.")
    else:
        print("Výsledky nie sú štatisticky významné.")

def main():
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

    results = {func.__name__: {alg: [] for alg in algorithms} for func in testing_functions}

    for func in testing_functions:
        for dim in dim_sizes:
            pop_size = pop_sizes[dim]
            f_evals = f_evals_multiplier * dim
            for alg_name, alg in algorithms.items():
                fitness_values = []
                for _ in range(num_repeats):
                    if alg_name == "PSO":
                        best_solution, best_fitness = alg(func, dim=dim, num_particles=pop_size, max_iter=f_evals)
                    else:
                        best_solution, best_fitness = alg(func, bounds=[BOUNDS] * dim, popsize=pop_size, its=f_evals)
                    fitness_values.append(best_fitness)
                results[func.__name__][alg_name] = fitness_values

    #Poradie
    for func_name, func_results in results.items():
        print(f"\nVyhodnotenie pre funkciu: {func_name}")
        data = [np.mean(func_results[alg]) for alg in algorithms]
        friedman_test(data)

main()
print("Done")