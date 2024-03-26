from emoc import EMOC

# from emoc.operator import Operator
from emoc.problem import Problem


class PolynomialMutation:
    def __init__(self, mu_pro: float = -1.0, eta_m: float = 20.0, **kwargs):
        super().__init__()
        self.mu_param = EMOC.MutationParameter()
        self.mu_param.pro = mu_pro
        self.mu_param.index1 = eta_m

    def __call__(self, population, pop_num, problem: Problem, **kwargs):
        lower_bound = problem.lower_bound_
        upper_bound = problem.upper_bound_
        if self.mu_param.pro == -1.0:
            self.mu_param.pro = 1.0 / len(population[0].dec_)
        for i in range(pop_num):
            EMOC.PolynomialMutationIndividual(
                population[i], lower_bound, upper_bound, self.mu_param
            )
