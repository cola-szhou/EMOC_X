import abc
import random


class Sampling:
    def __init__(self) -> None:
        pass

    def __call__(self, population, pop_num, problem, **kwargs):
        for i in range(pop_num):
            self.sample_ind(population[i], problem, **kwargs)

    @abc.abstractmethod
    def sample_ind(self, ind, problem, **kwargs):
        pass


class Mutation:
    def __init__(self, mu_pro=None):
        self.pro = mu_pro

    def __call__(self, population, pop_num, problem):
        if self.pro == None:
            self.pro = 1 / problem.dec_num_

        if not isinstance(self.pro, float):
            raise Exception("Invalid type for mutation probability!")
        for i in range(pop_num):
            self.mutation_ind(population[i], problem)

    @abc.abstractmethod
    def mutation_ind(self, ind, problem):
        pass


class Crossover:
    def __init__(self, cross_pro=0.9):
        self.pro = cross_pro

    def __call__(
        self, parent_pop, offspring_pop, pop_num, problem, selection_operator, **kwargs
    ):
        index1 = list(range(pop_num))
        random.shuffle(index1)
        index2 = list(range(pop_num))
        random.shuffle(index2)
        for i in range(pop_num // 2):
            parent1 = selection_operator(
                parent_pop[index1[2 * i]], parent_pop[index1[2 * i + 1]]
            )
            parent2 = selection_operator(
                parent_pop[index2[2 * i]], parent_pop[index2[2 * i + 1]]
            )
            self.cross_ind(
                parent1,
                parent2,
                offspring_pop[2 * i],
                offspring_pop[2 * i + 1],
                problem,
            )

    @abc.abstractmethod
    def cross_ind(self, parent1, parent2, offspring1, offspring2, problem):
        pass
