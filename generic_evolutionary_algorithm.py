import math
from random import randrange
import random

import matplotlib.pyplot as plt
from typing import List

import constants
import numpy as np


# Define parameters


def main():
    random.seed(0)
    np.random.seed(0)

    fitness_history = []
    fittest_per_gen = []

    problem = "TSP"
    if problem == "knapsack":

        results = evolutionary_algorithm_knapsack(
            crossover_probability=constants.CROSSOVER_PROBABILITY_KNAPSACK,
            mutation_probability=constants.MUTATION_PROBABILITY_KNAPSACK,
            num_pairs=constants.NUM_PAIRS_KNAPSACK,
            elitism_proportion=constants.ELITISM_PROPORTION_KNAPSACK,
            num_generations=constants.MAXIMUM_NUMBER_OF_GENERATIONS_KNAPSACK,
            lightest_weight=constants.LIGHTEST_WEIGHT,
            heaviest_weight=constants.HEAVIEST_WEIGHT,
            number_of_items=constants.NUMBER_OF_ITEMS,
            knapsack_capacity=constants.KNAPSACK_CAPACITY,
            tournament_size=constants.TOURNAMENT_SIZE_KNAPSACK,
            reward_penalty_overweight=constants.REWARD_PENALTY_PER_WEIGHT_UNIT_OVERWEIGHT
        )
        (wt_arr, rw_arr, generations, fittest_per_gen) = results

        print(f"the knapsack capacity is {constants.KNAPSACK_CAPACITY}")
        print(f"The weight array is {wt_arr}")
        print(f"The reward_arr is {rw_arr}")
        print(f"generations look like: {generations}")
        print(f"the fittest per generation:\n {fittest_per_gen}")

    elif problem == "TSP":

        generations, fittest_per_gen = evolutionary_algorithm_tsp(
            crossover_probability=constants.CROSSOVER_PROBABILITY_TSP,
            mutation_probability=constants.MUTATION_PROBABILITY_TSP,
            elitism_mutation=constants.MUTATION_ELITISM_TSP,
            elitism_crossover=constants.CROSSOVER_ELITISM_TSP,
            num_generations=constants.GENERATIONS_TSP,
            number_of_cities=constants.NUMBER_OF_CITIES,
            num_solution_pairs=constants.NUM_PAIRS_TSP,
            tournament_size=constants.TOURNAMENT_SIZE_TSP,
            stopping_epsilon=constants.STOPPING_EPSILON

        )

        fitness_history = [ind[1] for ind in fittest_per_gen]
        print(f"plotting the fitness value over generations: ")
        text1 = (
            f"c_elite: {str(constants.CROSSOVER_ELITISM_TSP)}\n"
            f"m_elite: {str(constants.MUTATION_ELITISM_TSP)}\n"
            f"tournament_size: {constants.TOURNAMENT_SIZE_TSP}\n"
            f"population_size: {constants.NUM_PAIRS_TSP * 2}"
        )
        text2 = (
            f"problem = {problem}\n"
            f"n_cities: {constants.NUMBER_OF_CITIES}\n"
            f"c_prob: {constants.CROSSOVER_PROBABILITY_TSP}\n"
            f"m_prob: {constants.MUTATION_PROBABILITY_TSP}"
        )

        plt.plot(fitness_history)
        plt.gca().set_position((.1, .3, .8, .6))
        plt.figtext(.02, .02, text2)
        plt.figtext(.35, .02, text1)
        plt.ylabel("total distance of solution")
        plt.xlabel("generation")
    plt.show()


def evolutionary_algorithm_tsp(
        crossover_probability: float,
        mutation_probability: float,
        elitism_mutation: bool,
        elitism_crossover: bool,
        num_generations: int,
        number_of_cities: int,
        num_solution_pairs: int,
        tournament_size: int,
        stopping_epsilon: float
):
    # We build a TSP in which the cities to be visited all lie equidistantly
    # on a unit circle
    unit_angle = (2 * np.pi) / number_of_cities
    distances = np.empty(shape=(number_of_cities,
                                number_of_cities))
    for i in range(0, number_of_cities):
        for j in range(0, number_of_cities):
            angle_between_two_cities = unit_angle * np.abs(i - j)
            distances[i, j] = 2 * np.sin(angle_between_two_cities / 2)

    population: list = initialize_pop_tsp(num_pairs=num_solution_pairs,
                                          distances=distances)
    current_generation = 0

    # keep track of best so far, to check convergence
    previous_gen_fittest = None
    fittest = get_fittest_tsp(population=population)
    best_solutions_per_generation = [fittest]
    population_history = [population]

    while not terminate_tsp_circle(
            population=population,
            distances=distances,
            num_gens=num_generations,
            current_gen=current_generation,
            stopping_epsilon=stopping_epsilon
    ):

        population_mate = [
            tournament_select_tsp(
                population=population,
                tournament_size=tournament_size
            )
            for i in range(len(population))
        ]
        # generate new population
        population = []
        while len(population_mate) > 0:
            # drawing parents to mate without replacement
            # https://stackoverflow.com/questions/306400/
            # how-to-randomly-select-an-item-from-a-list
            random_index = randrange(len(population_mate))
            parent1 = population_mate.pop(random_index)
            random_index = randrange(len(population_mate))
            parent2 = population_mate.pop(random_index)
            children = list(
                crossover_order_tsp(
                    parent1=parent1,
                    parent2=parent2,
                    distances=distances,
                    crossover_probability=crossover_probability,
                    elitism=elitism_crossover
                )
            )
            for child in children:
                mutant = mutate_individual_inversion_tsp(
                    individual=child,
                    distances=distances,
                    mutation_probability=mutation_probability,
                    elitism=elitism_mutation
                )
                population.append(mutant)

        # evaluate the new population
        current_generation += 1
        previous_gen_fittest = fittest
        fittest = get_fittest_tsp(population=population)
        best_solutions_per_generation.append(fittest)
        population_history.append(population)

    return population_history, best_solutions_per_generation


def evolutionary_algorithm_knapsack(
        crossover_probability,
        mutation_probability,
        num_pairs,
        elitism_proportion,
        num_generations,
        lightest_weight,
        heaviest_weight,
        number_of_items,
        knapsack_capacity,
        tournament_size,
        reward_penalty_overweight
):
    """
    generic evolutionary algorithm developed by Holland and discussed
    by Goldberg, adapted to accommodate the knapsack and traveling salesperson
    problem.

    :param reward_penalty_overweight:
    :param tournament_size:
    :param knapsack_capacity:
    :param number_of_items:
    :param heaviest_weight:
    :param lightest_weight:
    :param crossover_probability:
    :param mutation_probability:
    :param num_pairs:
    :param elitism_proportion:
    :param num_generations:
    :return:
    """
    # set knapsack weights and rewards
    rewards = initialize_rewards_knapsack(number_of_items)
    weights = initialize_weights_knapsack(
        lightest_weight,
        heaviest_weight,
        number_of_items
    )
    rewards_array = np.array(rewards)
    weights_array = np.array(weights)

    current_generation = 0

    population = initialize_pop_knapsack(
        num_pairs=num_pairs,
        number_of_possible_items=number_of_items,
        rewards=rewards_array,
        weights=weights_array,
        reward_penalty_per_unit_overweight=reward_penalty_overweight,
        knapsack_capacity=knapsack_capacity
    )
    # keep track of best so far, to check convergence
    previous_gen_fittest = None
    fittest = get_fittest_knapsack(population=population)
    best_solutions_per_generation = [fittest]
    population_history = [population]
    print(
        f"fittest individual in generation{current_generation}"
        f"is {fittest}"
    )

    while not terminate_knapsack(
            population,
            num_generations,
            current_generation,
            previous_gen_fittest,
            fittest
    ):
        # selecting mating population
        population_mate = [
            tournament_select_knapsack(population, tournament_size)
            for i
            in range(len(population))
        ]
        # generate new population
        population = []
        while len(population_mate) > 0:
            # drawing parents to mate without replacement
            # https://stackoverflow.com/questions/306400/
            # how-to-randomly-select-an-item-from-a-list
            random_index = randrange(len(population_mate))
            parent1 = population_mate.pop(random_index)
            random_index = randrange(len(population_mate))
            parent2 = population_mate.pop(random_index)
            children = list(
                crossover_uniform_knapsack(
                    parent1=parent1,
                    parent2=parent2,
                    crossover_probability=crossover_probability,
                    rewards=rewards_array,
                    weights=weights_array,
                    reward_penalty_per_unit_overweight=reward_penalty_overweight,
                    knapsack_capacity=knapsack_capacity
                )
            )
            for child in children:
                mutant = mutate_individual_knapsack(
                    child,
                    mutation_probability,
                    rewards_array,
                    weights_array,
                    reward_penalty_overweight,
                    knapsack_capacity
                )
                population.append(mutant)

        # evaluate the new population
        current_generation += 1
        previous_gen_fittest = fittest
        fittest = get_fittest_knapsack(population=population)
        best_solutions_per_generation.append(fittest)
        population_history.append(population)
        print(
            f"fittest individual in generation {current_generation} "
            f"is {fittest}"
        )

    return (weights_array, rewards_array, population_history,
            best_solutions_per_generation)


def initialize_rewards_knapsack(number_of_items):
    return [i + 1 for i in range(number_of_items)]


def initialize_weights_knapsack(
        lightest_weight,
        heaviest_weight,
        number_of_items
):
    return [
        random.randint(lightest_weight, heaviest_weight)
        for i
        in range(number_of_items)
    ]


def initialize_individual_knapsack(number_of_possible_items,
                                   rewards,
                                   weights,
                                   reward_penalty_per_unit_overweight,
                                   knapsack_capacity
                                   ) -> tuple:
    genotype = np.array(
        [random.randint(0, 1) for i in range(number_of_possible_items)]
    )

    fitness_value = fitness_knapsack(
        genotype=genotype,
        rewards=rewards,
        weights=weights,
        reward_penalty_per_unit_overweight=reward_penalty_per_unit_overweight,
        knapsack_capacity=knapsack_capacity
    )
    return genotype, fitness_value


def initialize_pop_knapsack(num_pairs,
                            number_of_possible_items,
                            rewards,
                            weights,
                            reward_penalty_per_unit_overweight,
                            knapsack_capacity):
    population = [
        initialize_individual_knapsack(number_of_possible_items,
                                       rewards,
                                       weights,
                                       reward_penalty_per_unit_overweight,
                                       knapsack_capacity)
        for i in
        range(2 * num_pairs)
    ]
    return population


def fitness_knapsack(
        genotype,
        rewards,
        weights,
        reward_penalty_per_unit_overweight,
        knapsack_capacity
) -> int:
    """
    evaluates the individuals fitness for a solution to the 0-1 knapsack
    problem
    :param genotype: the genotype of the individual
    :param rewards: a numpy array of rewards belonging to items in the knapsack
    :param weights: a numpy array of weights belonging to items in the knapsack
    :param reward_penalty_per_unit_overweight: reduce the reward if a phenotype
    exceeds the capacity of the knapsack
    :param knapsack_capacity: weight units that can be feasibly carried
    :return: the fitness value of an individual
    """
    reward_value = np.dot(genotype, rewards)
    individual_weight = np.dot(genotype, weights)

    reward_penalty = max(
        0,
        (individual_weight - knapsack_capacity)
    ) * reward_penalty_per_unit_overweight

    fitness_value = reward_value - reward_penalty
    return fitness_value


def get_fittest_knapsack(population):
    return max(population, key=lambda individual: individual[1])


def tournament_select_knapsack(population, k):
    entrants = random.sample(population, k)
    return get_fittest_knapsack(entrants)


def crossover_uniform_knapsack(
        parent1,
        parent2,
        crossover_probability,
        rewards,
        weights,
        reward_penalty_per_unit_overweight,
        knapsack_capacity
):
    """
    uniform crossover 0-1 knapsack with elitism family selection
    :param reward_penalty_per_unit_overweight:
    :param knapsack_capacity:
    :param weights:
    :param rewards:
    :param parent1:
    :param parent2:
    :param crossover_probability:
    :return: two of four fittest family members
    """
    # coin toss whether this pair will crossover
    if random.random() < crossover_probability:
        cross1 = [random.randint(0, 1) for i in range(len(parent1[0]))]
        cross2 = [random.randint(0, 1) for i in range(len(parent1[0]))]
        genotype1 = [
            parent1[0][i] if cross1[i] else parent2[0][i]
            for i in range(len(cross1))
        ]
        genotype2 = [
            parent1[0][i] if cross2[i] else parent2[0][i]
            for i in range(len(cross1))
        ]
        candidate_child1 = (
            np.array(genotype1),
            fitness_knapsack(
                genotype1,
                rewards,
                weights,
                reward_penalty_per_unit_overweight,
                knapsack_capacity
            )
        )
        candidate_child2 = (
            np.array(genotype2),
            fitness_knapsack(
                genotype1,
                rewards,
                weights,
                reward_penalty_per_unit_overweight,
                knapsack_capacity
            )
        )
        family = [parent1, parent2, candidate_child1, candidate_child2]
        # out of four family members select two with highest fitness
        family.sort(key=lambda member: member[1], reverse=True)
        children = family[:2]
        child1 = children[0]
        child2 = children[1]
    else:
        child1 = parent1
        child2 = parent2

    return child1, child2


def mutate_individual_knapsack(individual,
                               mutation_probability,
                               rewards,
                               weights,
                               reward_penalty_per_unit_overweight,
                               knapsack_capacity
                               ):
    """
    executes a mutation on a all genes of an individual in the
    knapsack problem

    :param knapsack_capacity:
    :param reward_penalty_per_unit_overweight:
    :param weights:
    :param rewards:
    :param mutation_probability:
    :param individual: the individual which gene needs to be mutated
    :return: a possibly mutated individual
    """

    for gene in range(len(individual[0])):
        if random.random() < mutation_probability:
            if individual[0][gene] == 0:
                individual[0][gene] = 1
            else:
                individual[0][gene] = 0

    fitness_new = fitness_knapsack(
        individual[0],
        rewards,
        weights,
        reward_penalty_per_unit_overweight,
        knapsack_capacity
    )
    individual_mutant = (individual[0], fitness_new)
    return individual_mutant


def terminate_knapsack(
        population,
        num_generations,
        current_generation,
        previous_gen_fittest,
        fittest) -> bool:
    if current_generation == 0:
        return False
    # convergence check
    if previous_gen_fittest[1] >= fittest[1]:
        print(
            f"TERMINATED DUE TO CONVERGENCE IN GENERATION: {current_generation}"
        )
        return True
    return current_generation >= num_generations


def initialise_unit_circle_points(number_of_points):
    city_indices = list(range(number_of_points))
    city_angles = [
        (i * 2 * math.pi) / number_of_points for i in range(number_of_points)
    ]
    cities = {k: v for (k, v) in zip(city_indices, city_angles)}
    return cities


def get_unit_circle_distance(city1, city2, distances) -> float:
    return distances[city1, city2]


def initialize_individual_tsp(distances):
    """
     create a randomly ordered list of cities with starting city fixed 0
    """
    ordering = list(range(1, len(distances)))
    random.shuffle(ordering)
    genotype = [0]
    genotype.extend(ordering)
    # return to starting city
    genotype.append(0)
    fitness = get_fitness_tsp_circle(genotype, distances)

    return genotype, fitness


def get_fitness_tsp_circle(genotype, distances):
    """ The TSP minimizes distance. Lower is  fitter"""
    distance = 0
    for i in range(len(genotype) - 1):
        distance += get_unit_circle_distance(
            genotype[i],
            genotype[i + 1],
            distances
        )
    return distance


def initialize_pop_tsp(num_pairs, distances):
    population = [
        initialize_individual_tsp(distances=distances)
        for i in
        range(2 * num_pairs)
    ]
    return population


def tournament_select_tsp(population, tournament_size):
    entrants = random.sample(population, tournament_size)
    return get_fittest_tsp(entrants)


def get_fittest_tsp(population: list):
    return min(population, key=lambda ind: ind[1])


def crossover_order_tsp(parent1: tuple,
                        parent2: tuple,
                        crossover_probability: float,
                        distances,
                        elitism: bool):
    # 1. pick two cut points in the tour
    tour_length = len(distances)
    cut_points = random.sample(list(range(0, tour_length)), 2)
    cut_points.sort()
    # print(f"cut pts: {cut_points}")

    # 2. copy the fixed parts of each offspring

    # remove start and end
    tour_parent1: list = parent1[0][1:-1]
    tour_parent2: list = parent2[0][1:-1]

    # print(f"tour parent 1: {tour_parent1}")

    offspring1_fixed = tour_parent1[cut_points[0]:cut_points[1]]
    offspring2_fixed = tour_parent2[cut_points[0]:cut_points[1]]
    # print(f"offspring1 fixed part: {offspring1_fixed}")
    # print(f"offspring2 fixed part: {offspring2_fixed}")

    # 3. create the order of the remaining visits after the second cut point
    # for both offspring
    offspring1_remaining = tour_parent2
    offspring2_remaining = tour_parent1

    for i in offspring1_fixed:
        offspring1_remaining.remove(i)

    for i in offspring2_fixed:
        offspring2_remaining.remove(i)

    # print(f"remaining for offspring 1: {offspring1_remaining}")
    # print(f"remaining for offspring 2: {offspring2_remaining}")

    offspring1_post_cut = offspring1_remaining[
                          0:(tour_length - cut_points[1] - 1)
                          ]
    offspring2_post_cut = offspring2_remaining[
                          0:(tour_length - cut_points[1] - 1)
                          ]

    offspring1_pre_cut = offspring1_remaining[
                         (tour_length - cut_points[1] - 1):len(
                             offspring1_remaining)
                         ]

    offspring2_pre_cut = offspring2_remaining[
                         (tour_length - cut_points[1] - 1):len(
                             offspring2_remaining)
                         ]

    # 4. assemble the the tour for both offspring
    offspring1 = [
                     0] + offspring1_pre_cut + offspring1_fixed + offspring1_post_cut + [
                     0]
    offspring2 = [
                     0] + offspring2_pre_cut + offspring2_fixed + offspring2_post_cut + [
                     0]
    o1 = (offspring1, get_fitness_tsp_circle(offspring1, distances))
    o2 = (offspring2, get_fitness_tsp_circle(offspring2, distances))

    if elitism:
        family = [parent1, parent2, o1, o2]
        # out of four family members select two with highest fitness
        family.sort(key=lambda member: member[1], reverse=False)
        return family[0], family[1]

    return o1, o2


def mutate_individual_inversion_tsp(
        individual: tuple,
        distances,
        mutation_probability: float,
        elitism: bool):
    """
    executes a mutation on a specified  individual in the
    traveling salesman problem

    https://www.ijert.org/research/comparison-of-various-mutation-operators-of-genetic-algorithm-to-resolve-travelling-salesman-problem-IJERTV2IS60404.pdf

    :param individual: the individual which gene needs to be mutated
    :param cities: places to visit
    :return:
    """
    if random.random() > mutation_probability:
        return individual

    # randomly pick the indices to start and end the sequence to invert
    tour_length = len(distances)
    # exclude the start and end point indices from inversion
    cut_points = random.sample(list(range(1, tour_length)), 2)
    cut_points.sort()
    # print(f"cut pts: {cut_points}")

    # isolate the part to invert
    original_tour = individual[0]
    inversion_slice = original_tour[cut_points[0]:cut_points[1]]
    inversion_slice.reverse()
    mutated_tour = original_tour[:cut_points[0]] \
                   + inversion_slice + original_tour[cut_points[1]:]
    mutated_fitness = get_fitness_tsp_circle(mutated_tour, distances)

    mutated_individual = (mutated_tour, mutated_fitness)

    if elitism:
        candidates = [individual, mutated_individual]
        return get_fittest_tsp(candidates)

    return mutated_individual


def terminate_tsp_circle(
        population: list,
        distances,
        num_gens: int,
        current_gen: int,
        stopping_epsilon: float
) -> bool:
    # stopping based on known best solution
    best = [0]
    best_path = list(range(1, len(population[0][0]) - 1))
    best.extend(best_path)
    best.append(0)

    best_fitness = get_fitness_tsp_circle(best, distances)

    print(
        f"fittest individual in generation {current_gen} "
        f"is {get_fittest_tsp(population)}"
    )
    pop_fitness: List[float] = [individual[1] for individual in population]

    # print("Population fitness histogram: ")
    # print(np.histogram(pop_fitness))

    # plot every 10th generation
    if current_gen % 10 == 0:
        plt.hist(pop_fitness, bins=10)
        text1 = (
            f"problem: TSP\n"
            f"n_cities: {constants.NUMBER_OF_CITIES}\n"
            f"c_prob: {constants.CROSSOVER_PROBABILITY_TSP}\n"
            f"m_prob: {constants.MUTATION_PROBABILITY_TSP}"
        )
        text2 = (
            f"c_elite: {str(constants.CROSSOVER_ELITISM_TSP)}\n"
            f"m_elite: {str(constants.MUTATION_ELITISM_TSP)}\n"
            f"tournament_size: {constants.TOURNAMENT_SIZE_TSP}\n"
            f"population_size: {constants.NUM_PAIRS_TSP * 2}"
        )
        text3 = (f"generation is: {current_gen}")

        plt.gca().set_position((.1, .3, .8, .6))
        plt.figtext(.02, .02, text1)
        plt.figtext(.35, .02, text2)
        plt.figtext(.68, .02, text3)
        plt.xlabel("total distance of solution")
        plt.ylabel("frequency")

        plt.show()
        print(
            '-------------------------------------------------------------------')

    if get_fittest_tsp(population)[1] - best_fitness < stopping_epsilon:
        print(f"terminating as best solution was found in generation: "
              f"{current_gen}")
        print(f"known best solution {best} , {best_fitness}")
        print(f"found best solution {get_fittest_tsp(population)}")
        return True
    if current_gen >= num_gens:
        print(f"terminating as generation limit is reached")
        return True
    return False


if __name__ == "__main__":
    main()
