import math
from random import randrange
import random
import constants
import numpy as np


def main():
    random.seed(0)
    weight_arr, reward_arr, solution, previous_sol = evolutionary_algorithm_knapsack(
        crossover_probability=constants.CROSSOVER_PROBABILITY,
        mutation_probability=constants.MUTATION_PROBABILITY,
        num_pairs=constants.NUM_PAIRS,
        elitism_proportion=constants.ELITISM_PROPORTION,
        num_generations=constants.MAXIMUM_NUMBER_OF_GENERATIONS,
        lightest_weight=constants.LIGHTEST_WEIGHT,
        heaviest_weight=constants.HEAVIEST_WEIGHT,
        number_of_items=constants.NUMBER_OF_ITEMS,
        knapsack_capacity=constants.KNAPSACK_CAPACITY,
        tournament_size=constants.TOURNAMENT_SIZE_KNAPSACK,
        reward_penalty_overweight=constants.REWARD_PENALTY_PER_WEIGHT_UNIT_OVERWEIGHT
    )
    print(f"the knapsack capacity is {constants.KNAPSACK_CAPACITY}")
    print(f"The weight array is {weight_arr}")
    print(f"The reward_arr is {reward_arr}")
    solution_weight = np.dot(solution[0], weight_arr)
    previous_sol_weight = np.dot(previous_sol[0], weight_arr)
    print(f"weight of solution: {solution_weight}")
    print(f"weight of previous solution: {previous_sol_weight}")

    # rewards = initialize_rewards_knapsack(constants.NUMBER_OF_ITEMS)
    # weights = initialize_weights_knapsack(constants.LIGHTEST_WEIGHT,
    #                                       constants.HEAVIEST_WEIGHT,
    #                                       constants.NUMBER_OF_ITEMS)
    #
    # rewards_array = np.array(rewards)
    # weights_array = np.array(weights)

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

    # pop = initialize_pop_knapsack(
    #     4,
    #     constants.NUMBER_OF_ITEMS,
    #     rewards_array,
    #     weights_array,
    #     constants.REWARD_PENALTY_PER_WEIGHT_UNIT_OVERWEIGHT,
    #     constants.KNAPSACK_CAPACITY
    # )
    # print(f"population: {pop} ")
    #
    # best = tournament_select_knapsack(population=pop, k=4)
    # print(f"fittest: {best}")
    # print(len(best[0]))
    #
    # p1, p2 = pop[1], best
    # c1, c2 = crossover_uniform_knapsack(
    #     p1, p2,
    #     constants.CROSSOVER_PROBABILITY,
    #     rewards_array,
    #     weights_array,
    #     constants.REWARD_PENALTY_PER_WEIGHT_UNIT_OVERWEIGHT,
    #     constants.KNAPSACK_CAPACITY
    # )
    # print(f"child1: {c1}")
    # print(f"child2: {c2}")
    #
    # mutant1 = mutate_individual_knapsack(
    #     c1,
    #     constants.MUTATION_PROBABILITY,
    #     rewards_array,
    #     weights_array,
    #     constants.REWARD_PENALTY_PER_WEIGHT_UNIT_OVERWEIGHT,
    #     constants.KNAPSACK_CAPACITY
    # )
    # mutant2 = mutate_individual_knapsack(
    #     c2,
    #     constants.MUTATION_PROBABILITY,
    #     rewards_array,
    #     weights_array,
    #     constants.REWARD_PENALTY_PER_WEIGHT_UNIT_OVERWEIGHT,
    #     constants.KNAPSACK_CAPACITY
    # )
    # print(f"mutant1: {mutant1}")
    # print(f"mutant2: {mutant2}")
    #
    # pass


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
        print(
            f"fittest individual in generation{current_generation}"
            f"is {fittest}"
        )

    return weights_array, rewards_array, fittest, previous_gen_fittest


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
    todo: the mutation
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
