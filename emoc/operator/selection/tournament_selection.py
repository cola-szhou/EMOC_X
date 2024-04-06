import EMOC
import random

# from emoc.operator import Operator
from emoc.core import Individual


class TournamentByRank:
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, ind1, ind2, **kwargs) -> Individual:
        if ind1.rank_ < ind2.rank_:
            return ind1
        elif ind2.rank_ < ind1.rank_:
            return ind2
        else:
            return ind1 if random.random() <= 0.5 else ind2


class TournamentByFitness:
    def __init__(self, greater_is_better: bool = False, **kwargs):
        super().__init__()
        self.greater_is_better = greater_is_better

    def do(self, ind1: Individual, ind2: Individual, **kwargs) -> Individual:
        if ind1.fitness_ < ind2.fitness_:
            return ind2 if self.greater_is_better else ind1
        elif ind2.fitness_ < ind1.fitness_:
            return ind1 if self.greater_is_better else ind2
        else:
            return ind1 if random.random() <= 0.5 else ind2


class TournamentByCustom:
    def __init__(self, comparison_function, **kwargs):
        super().__init__()
        self.comparison_function = comparison_function

    def do(self, ind1: Individual, ind2: Individual, **kwargs) -> Individual:
        if self.comparison_function(ind1, ind2):
            return ind1
        elif self.comparison_function(ind2, ind1):
            return ind2
        else:
            return ind1 if random.random() <= 0.5 else ind2
