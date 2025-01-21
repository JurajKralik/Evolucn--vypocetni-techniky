import numpy as np
import matplotlib.pyplot as plt

BOUNDS = [-100, 100]

# ZDROJ: https://gist.github.com/pablormier/0caff10a5f76e87857b44f63757729b0
def de_rand(fobj, bounds= BOUNDS, mut=0.8, crossp=0.7, popsize=20, its=1000):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    for i in range(its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            f = fobj(trial_denorm)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        return best, fitness[best_idx]

# Modifikovaná DE/rand verzia na DE/best
def de_best(fobj, bounds, mut=0.8, crossp=0.7, popsize=20, its=1000):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    
    for i in range(its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            b, c = pop[np.random.choice(idxs, 2, replace=False)]
            mutant = np.clip(best + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            f = fobj(trial_denorm)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        return best, fitness[best_idx]

# ZDROJ: https://induraj2020.medium.com/implementing-particle-swarm-optimization-in-python-c59278bc5846
def pso(cost_func, dim=2, num_particles=30, max_iter=100, w=0.5, c1=1, c2=2):
    # Initialize particles and velocities
    particles = np.random.uniform(-5.12, 5.12, (num_particles, dim))
    velocities = np.zeros((num_particles, dim))

    # Initialize the best positions and fitness values
    best_positions = np.copy(particles)
    best_fitness = np.array([cost_func(p) for p in particles])
    swarm_best_position = best_positions[np.argmin(best_fitness)]
    swarm_best_fitness = np.min(best_fitness)

    # Iterate through the specified number of iterations, updating the velocity and position of each particle at each iteration
    for i in range(max_iter):
        # Update velocities
        r1 = np.random.uniform(0, 1, (num_particles, dim))
        r2 = np.random.uniform(0, 1, (num_particles, dim))
        velocities = w * velocities + c1 * r1 * (best_positions - particles) + c2 * r2 * (swarm_best_position - particles)

        # Update positions
        particles += velocities

        # Evaluate fitness of each particle
        fitness_values = np.array([cost_func(p) for p in particles])

        # Update best positions and fitness values
        improved_indices = np.where(fitness_values < best_fitness)
        best_positions[improved_indices] = particles[improved_indices]
        best_fitness[improved_indices] = fitness_values[improved_indices]
        if np.min(fitness_values) < swarm_best_fitness:
            swarm_best_position = particles[np.argmin(fitness_values)]
            swarm_best_fitness = np.min(fitness_values)

    # Return the best solution found by the PSO algorithm
    return swarm_best_position, swarm_best_fitness

# ZDROJ: https://induraj2020.medium.com/implementing-particle-swarm-optimization-in-python-c59278bc5846
def rastrigin_function(x):
    n = len(x)
    return 10*n + sum([xi**2 - 10*np.cos(2*np.pi*xi) for xi in x])

# ZDROJ: ChatGPT 4o - modifikované
def ackley_function(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    n = len(x)
    return -a * np.exp(-b * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + a + np.e

def sphere_function(x):
    return np.sum(x**2)

def rosenbrock_function(x):
    return np.sum([100 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2 for i in range(len(x) - 1)])

def griewank_function(x):
    sum_part = np.sum(x**2) / 4000
    prod_part = np.prod([np.cos(x[i] / np.sqrt(i + 1)) for i in range(len(x))])
    return sum_part - prod_part + 1

def schwefel_function(x):
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

def levy_function(x):
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    return term1 + term2 + term3

def zakharov_function(x):
    sum1 = np.sum(x**2)
    sum2 = np.sum(0.5 * (i + 1) * x[i] for i in range(len(x)))
    return sum1 + sum2**2 + sum2**4

def michalewicz_function(x, m=10):
    return -np.sum([np.sin(x[i]) * np.sin((i + 1) * x[i]**2 / np.pi)**(2 * m) for i in range(len(x))])

def beale_function(x):
    x1, x2 = x[0], x[1]
    return (1.5 - x1 + x1 * x2)**2 + (2.25 - x1 + x1 * x2**2)**2 + (2.625 - x1 + x1 * x2**3)**2

def booth_function(x):
    x1, x2 = x[0], x[1]
    return (x1 + 2 * x2 - 7)**2 + (2 * x1 + x2 - 5)**2

def easom_function(x):
    x1, x2 = x[0], x[1]
    return -np.cos(x1) * np.cos(x2) * np.exp(-((x1 - np.pi)**2 + (x2 - np.pi)**2))

def three_hump_camel_function(x):
    x1, x2 = x[0], x[1]
    return 2 * x1**2 - 1.05 * x1**4 + (x1**6) / 6 + x1 * x2 + x2**2

def xin_she_yang_function(x):
    return np.sum(np.abs(x) * np.exp(-np.sin(x**2)))

def sum_of_different_powers_function(x):
    return np.sum([np.abs(x[i])**(i + 2) for i in range(len(x))])

# Vlastné

def visualize(testing_function, best_solution, best_value):
    x = np.linspace(-100, 100, 400)
    y = np.linspace(-100, 100, 400)
    X, Y = np.meshgrid(x, y)
    Z = np.array([testing_function(np.array([xi, yi])) for xi, yi in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

    # 2D Contour Plot
    plt.figure(figsize=(10, 6))
    plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar()
    plt.scatter(best_solution[0], best_solution[1], color='red', marker='x', label='Best Solution')
    plt.title(f"2D Contour Plot: {testing_function.__name__}")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    # 3D Surface Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    ax.scatter(best_solution[0], best_solution[1], best_value, color='red', marker='x', s=100, label='Best Solution')
    ax.set_title(f"3D Surface Plot: {testing_function.__name__}")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    ax.legend()
    plt.show()

def run_testing_function(testing_function):
    print("Function: ",{testing_function.__name__})
    best_solution, best_value = de_rand(testing_function)
    print("Best Solution:", best_solution)
    print("Best Value:", best_value)
    visualize(testing_function, best_solution, best_value)

testing_functions = [
    sum_of_different_powers_function,
    ackley_function,
    rastrigin_function,
    rosenbrock_function,
    griewank_function,
    schwefel_function,
    sphere_function,
    levy_function,
    zakharov_function,
    michalewicz_function,
    beale_function,
    booth_function,
    easom_function,
    three_hump_camel_function,
    xin_she_yang_function,
]

def main():
    for testing_function in testing_functions:
        run_testing_function(testing_function)