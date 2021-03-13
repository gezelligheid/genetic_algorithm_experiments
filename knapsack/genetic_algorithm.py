import random
import math
import matplotlib.pyplot as plt
import numpy as np
import copy
import collections

random.seed(5)
np.random.seed(5)

# Define parameters
PROBLEM = "knapsack"
MUTATION_RATE = 0.2
CROSSOVER_RATE = 1
POPULATION_SIZE = 512
TEMPERATURE = 1
MAX_GENS = 20
if PROBLEM == "knapsack":
    N_ITEMS = 32
    WEIGHT_LIMIT = 64
    ITEM_REWARDS = list(range(1, N_ITEMS + 1))
    ITEM_WEIGHTS = [random.uniform(1, 10) for x in range(0, N_ITEMS)]
elif PROBLEM == "TSP":
    N_CITIES = 5
    # We build a TSP in which the cities to be visited all lie equidistantly on a unit circle
    unit_angle = (2 * np.pi) / N_CITIES
    DISTANCES = np.empty(shape=(N_CITIES, N_CITIES))
    for i in range(0, N_CITIES):
        for j in range(0, N_CITIES):
            angle_between_two_cities = unit_angle * np.abs(i - j)
            DISTANCES[i, j] = 2 * np.sin(angle_between_two_cities / 2)


def main():
    # Randomly initialize the population
    population = []
    for i in range(0, POPULATION_SIZE):
        population.append(initialize())

    current_gen = 0
    while current_gen <= MAX_GENS:
        # Statistic report of the current population
        pop_fitness = [calculate_fitness(individual) for individual in
                       population]
        best_index = np.argmax(pop_fitness)

        print(
            f"Best individual in generation {current_gen} : "
            f"{population[best_index]} with fitness {pop_fitness[best_index]}")
        if current_gen % 5 == 0:
            print("Population fitness histogram: ")
            print(np.histogram(pop_fitness))
            plt.hist(pop_fitness, bins=10)

            text1 = (
                f"problem: {PROBLEM}\n"
                f"generation is: {current_gen}\n"
                f"population_size: {POPULATION_SIZE}"
            )
            text2 = (
                f"temperature is: {TEMPERATURE}\n"
                f"c_prob: {CROSSOVER_RATE}\n"
                f"m_prob: {MUTATION_RATE}"
            )

            plt.gca().set_position((.1, .3, .8, .6))
            plt.figtext(.02, .02, text1)
            plt.figtext(.35, .02, text2)
            plt.show()
            print(
                '---------------------------------------------------------------------------------')

        mating_pool = []
        while len(mating_pool) < len(population):
            mating_pool.append(select(population, TEMPERATURE))

        population = []
        while len(mating_pool) != 0:
            parents = random.sample(mating_pool, 2)
            for parent in parents:
                mating_pool.remove(parent)
            first_parent = parents[0]
            second_parent = parents[1]

            if random.random() < CROSSOVER_RATE:
                first_child, second_child = crossover(first_parent,
                                                      second_parent)
            else:
                first_child = first_parent
                second_child = second_parent

            first_child = mutate(first_child, MUTATION_RATE)
            second_child = mutate(second_child, MUTATION_RATE)

            population.append(first_child)
            population.append(second_child)
        current_gen += 1


def are_same_individual(individual1, individual2):
    assert len(individual1) == len(individual2)
    for i in range(0, len(individual1)):
        if not individual1[i] == individual2[i]:
            return False
    return True


def initialize():
    if PROBLEM == "knapsack":
        while True:
            individual = [random.randint(0, 1) for _ in range(0, N_ITEMS)]
            if is_legal(individual):
                return individual
    elif PROBLEM == "TSP":
        individual = np.random.choice(N_CITIES, size=N_CITIES,
                                      replace=False).tolist()
        assert is_legal(individual)
        return individual
    else:
        raise RuntimeError("Problem not known.")


def mutate(individual, mutation_rate):
    if PROBLEM == "knapsack":
        while True:
            new_individual = copy.deepcopy(individual)
            for i in range(0, len(new_individual)):
                if random.random() < mutation_rate:
                    new_individual[i] = abs(new_individual[i] - 1)
            if is_legal(new_individual):
                return new_individual
    elif PROBLEM == "TSP":
        # To avoid illegal individual, we specify a special mutation operator:
        # For each city on the path, if it is mutated, we randomly choose another city on the path and switch them
        new_individual = copy.deepcopy(individual)
        for i in range(0, len(new_individual)):
            if random.random() < mutation_rate:
                # Randomly choose a spot on the path
                new_spot = np.random.choice(N_CITIES)
                # Switch the two cities
                tmp = new_individual[i]
                new_individual[i] = new_individual[new_spot]
                new_individual[new_spot] = tmp
        assert is_legal(new_individual)
        return new_individual
    else:
        raise RuntimeError("Problem not known.")


def crossover(first_parent, second_parent):
    first_parent_copy = copy.deepcopy(first_parent)
    second_parent_copy = copy.deepcopy(second_parent)
    if PROBLEM == "knapsack":
        first_child = first_parent_copy[
                      :math.floor(N_ITEMS / 2)] + second_parent_copy[
                                                  math.floor(N_ITEMS / 2):]
        second_child = second_parent_copy[
                       :math.floor(N_ITEMS / 2)] + first_parent_copy[
                                                   math.floor(N_ITEMS / 2):]
        if is_legal(first_child) and is_legal(second_child):
            return first_child, second_child
        else:
            return first_parent_copy, second_parent_copy
    elif PROBLEM == "TSP":
        # TODO: explain
        first_child_first_half = first_parent_copy[0: math.floor(N_CITIES / 2)]
        first_child_second_half = [x for x in second_parent_copy if
                                   x not in first_child_first_half]
        first_child = first_child_first_half + first_child_second_half

        second_child_first_half = second_parent_copy[
                                  0: math.floor(N_CITIES / 2)]
        second_child_second_half = [x for x in first_parent_copy if
                                    x not in second_child_first_half]
        second_child = second_child_first_half + second_child_second_half

        assert is_legal(first_child) and is_legal(second_child)
        return first_child, second_child
    else:
        raise RuntimeError("Problem not known.")


def find_index(individual, value):
    assert PROBLEM == 'TSP'
    for i in range(0, len(individual)):
        if math.isclose(individual[i], value):
            return i
    raise RuntimeError(f'Value {value} not found in individual {individual}')


def terminate():
    raise NotImplementedError


def select(population, temperature):
    roulette_wheel = []
    for individual in population:
        roulette_wheel.append(
            math.exp(calculate_fitness(individual) / temperature))
    total = sum(roulette_wheel)
    roulette_wheel = [x / total for x in roulette_wheel]

    cdf = [roulette_wheel[0]]
    for i in range(1, len(roulette_wheel)):
        cdf.append(cdf[i - 1] + roulette_wheel[i])
    cdf.insert(0, 0)
    assert math.isclose(cdf[-1], 1)
    assert math.isclose(cdf[0], 0)
    assert len(cdf) - 1 == len(population)

    random_numb = random.random()
    for i in range(0, len(population)):
        if cdf[i] <= random_numb < cdf[i + 1]:
            return population[i]


def is_legal(individual):
    if PROBLEM == "knapsack":
        weight_list = [w * i for (w, i) in zip(ITEM_WEIGHTS, individual)]
        total_weight = sum(weight_list)
        return total_weight <= WEIGHT_LIMIT
    elif PROBLEM == "TSP":
        # We are representing a solution as a path, implying that the last edge would be from the last city to the
        # first city. So if a city is visited more than once --> the solution no longer a path --> illegal
        for i in range(0, N_CITIES):
            if i not in individual:
                return False
        return len(set(individual)) == len(individual)
    else:
        raise RuntimeError("Problem not known.")


def calculate_fitness(individual):
    if PROBLEM == "knapsack":
        reward_list = [r * i for (r, i) in zip(ITEM_REWARDS, individual)]
        total_reward = sum(reward_list)
        return total_reward
    elif PROBLEM == "TSP":
        path_length = 0
        for i in range(0, len(individual)):
            if i == len(individual) - 1:
                path_length = path_length + DISTANCES[
                    int(individual[i]), int(individual[0])]
            else:
                path_length = path_length + DISTANCES[
                    int(individual[i]), int(individual[i + 1])]
        return -path_length
    else:
        raise RuntimeError("Problem not known.")


if __name__ == '__main__':
    main()
