from emoc import EMOC
import random

from emoc.core import Individual
from emoc.operator import TournamentByRank
from emoc.problem import Problem

class SBX:
    def __init__(self, cross_pro: float=1.0, eta_c: float=20.0) -> None:
        self.cross_param = EMOC.CrossoverParameter()
        self.cross_param.pro = cross_pro
        self.cross_param.index1 = eta_c

    def __call__(self, parent_pop, offspring_pop, pop_num, problem, selection_operator=TournamentByRank(), **kwargs):
        index1 = list(range(pop_num))
        random.shuffle(index1)
        index2 = list(range(pop_num))
        random.shuffle(index2)
        lower_bound = problem.lower_bound_
        upper_bound = problem.upper_bound_
        for i in range(pop_num // 2):
            parent1 = selection_operator(parent_pop[index1[2 * i]], parent_pop[index1[2 * i + 1]])
            parent2 = selection_operator(parent_pop[index2[2 * i]], parent_pop[index2[2 * i + 1]])
            EMOC.SBX(parent1, parent2, offspring_pop[2 * i], offspring_pop[2 * i +1], lower_bound, upper_bound, self.cross_param)
            
    def SBX_ind(self, parent1: Individual, parent2: Individual, offspring1: Individual, offspring2: Individual, problem: Problem):
        EMOC.SBX(parent1, parent2, offspring1, offspring2, problem.lower_bound_, problem.upper_bound_, self.cross_param)