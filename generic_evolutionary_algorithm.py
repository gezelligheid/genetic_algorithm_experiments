import math
from random import randrange
import random
import constants
import numpy as np


def main():
    random.seed(0)
    rewards = initialize_rewards_knapsack(constants.NUMBER_OF_ITEMS)
    weights = initialize_weights_knapsack(constants.LIGHTEST_WEIGHT,
                                          constants.HEAVIEST_WEIGHT,
                                          constants.NUMBER_OF_ITEMS)

    rewards_array = np.array(rewards)
    weights_array = np.array(weights)

    # print(rewards_array)
    # print(weights_array)
    #
    # ind: tuple = initialize_individual_knapsack(
    #     constants.NUMBER_OF_ITEMS,
    #     rewards_array,
    #     weights_array,
    #     constants.REWARD_PENALTY_PER_WEIGHT_UNIT_OVERWEIGHT,
    #     constants.KNAPSACK_CAPACITY
    # )
    # print(ind)
    # print(type(ind))

    pop = initialize_pop_knapsack(
        4,
        constants.NUMBER_OF_ITEMS,
        rewards_array,
        weights_array,
        constants.REWARD_PENALTY_PER_WEIGHT_UNIT_OVERWEIGHT,
        constants.KNAPSACK_CAPACITY
    )
    print(f"population: {pop} ")

    best = tournament_select_knapsack(population=pop, k=4)
    print(f"fittest: {best}")
    print(len(best[0]))

    p1, p2 = pop[1], best
    c1, c2 = crossover_uniform_knapsack(
        p1, p2,
        constants.CROSSOVER_PROBABILITY,
        rewards_array,
        weights_array,
        constants.REWARD_PENALTY_PER_WEIGHT_UNIT_OVERWEIGHT,
        constants.KNAPSACK_CAPACITY
    )
    print(f"child1: {c1}")
    print(f"child2: {c2}")

    # population = initialize_pop_knapsack(constants.NUM_PAIRS,
    #                                      constants.NUMBER_OF_ITEMS)
    # print(population)
    # print(type(population))

    pass


def evolutionary_algorithm_knapsack(crossover_probability,
                                    mutation_probability,
                                    num_pairs,
                                    elitism_proportion,
                                    num_generations,
                                    lightest_weight,
                                    heaviest_weight,
                                    number_of_items,
                                    knapsack_capacity,
                                    tournament_size
                                    ):
    """
    generic evolutionary algorithm developed by Holland and discussed
    by Goldberg, adapted to accommodate the knapsack and traveling salesperson
    problem.

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

    population = initialize_pop_knapsack(num_pairs=num_pairs)

    while not terminate_knapsack(population, num_generations,
                                 current_generation):
        # selecting mating population
        population_mate = [
            tournament_select_knapsack(population, tournament_size)
            for i
            in range(len(population))
        ]
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
                    crossover_probability=crossover_probability
                )
            )
            for child in children:
                for gene in child:
                    mutate_individual_knapsack(child, gene)
                population.append(child)

            current_generation += 1


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
        range(2 * num_pairs)]
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


def tournament_select_knapsack(population, k):
    entrants = random.sample(population, k)
    return max(population, key=lambda item: item[1])


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
    :return:
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
    todo: the mutation
    executes a mutation on a all genes of an individual in the
    knapsack problem

    :param knapsack_capacity:
    :param reward_penalty_per_unit_overweight:
    :param weights:
    :param rewards:
    :param mutation_probability:
    :param individual: the individual which gene needs to be mutated
    :return:
    """

    for gene in range(individual[0]):
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


def terminate_knapsack(population, num_generations,
                       current_generation) -> bool:
    pass


def initialize_individual_tsp():
    pass


def initialize_pop_tsp(num_pairs):
    population_size = 2 * num_pairs
    pass


def select_tsp(population):
    pass


def crossover_tsp(parent1, parent2, crossover_probability):
    pass


def mutate_individual_tsp(individual, gene):
    """
    executes a mutation on a specified gene of an individual in the
    knapsack problem

    :param individual: the individual which gene needs to be mutated
    :param gene: the gene or locatation where the mutation takes place
    :return:
    """
    pass


def terminate_tsp(population) -> bool:
    pass


if __name__ == "__main__":
    main()
