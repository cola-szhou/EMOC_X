import EMOC


class NSGA2:
    def __init__(self, crossover_prob=1.0, eta_c=20.0, mutation_prob=None, eta_m=20.0):
        self.crossover_prob_ = crossover_prob
        self.eta_c_ = eta_c
        self.mutation_prob_ = mutation_prob
        self.eta_m_ = eta_m
        self.algorithm = None
        self.name = "NSGA-II"

    def Solve(self, problem, global_):
        if self.mutation_prob_ == None:
            self.mutation_prob_ = 1.0 / problem.dec_num_
        self.algorithm = EMOC.NSGA2(
            global_,
            problem,
            self.crossover_prob_,
            self.eta_c_,
            self.mutation_prob_,
            self.eta_m_,
        )
        self.algorithm.Solve()

    def PrintResult(self, i):
        self.algorithm.PrintResult()
