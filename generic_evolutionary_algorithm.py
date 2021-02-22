from random import randrange


def main():
    pass


def evolutionary_algorithm(crossover_probability,
                           mutation_probability,
                           num_pairs,
                           elitism_proportion,
                           num_generations):
    """
    generic evolutionary algorithm developed by Holland and discussed
    by Goldberg, adapted to accommodate the knapsack and traveling salesperson
    problem.

    :param crossover_probability:
    :param mutation_probability:
    :param num_pairs:
    :param elitism_proportion:
    :param num_generations:
    :return:
    """
    population = initialize_pop_knapsack()

    while not terminate_knapsack(population):
        # selecting mating population
        population_mate = [select_knapsack(population) for i in
                           range(len(population))]
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
                crossover_knapsack(
                    parent1=parent1,
                    parent2=parent2,
                    crossover_probability=crossover_probability
                )
            )
            for child in children:
                for gene in child:
                    mutate_individual_knapsack(child, gene)
                population.append(child)


def initialize_individual_knapsack():

    return individual


def initialize_pop_knapsack(num_pairs):
    population = [initialize_individual_knapsack() for i in
                  range(2 * num_pairs)]
    return population


def select_knapsack(population):
    pass


def crossover_knapsack(parent1, parent2, crossover_probability):
    """
    todo
    :param parent1:
    :param parent2:
    :param crossover_probability:
    :return:
    """
    child1 = []
    child2 = []
    return child1, child2


def mutate_individual_knapsack(individual, gene):
    """
    todo: the mutation
    executes a mutation on a specified gene of an individual in the
    knapsack problem

    :param individual: the individual which gene needs to be mutated
    :param gene: the gene or locatation where the mutation takes place
    :return:
    """
    return individual


def terminate_knapsack(population) -> bool:
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
