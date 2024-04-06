import os
from emoc.algorithm import Algorithm
import GPy
from emoc.utils import constants
import numpy as np
from emoc.core import Individual
from emoc.utils.utility import CalInverseChebycheff
from emoc.operator import UniformPoint
from GPyOpt.methods import BayesianOptimization
from GPyOpt.models.gpmodel import GPModel
import time


os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


class ParEGO(Algorithm):
    def __init__(self, restart_num_=1, window_size=25):
        super().__init__()
        print(
            "--warning: for ParEGO the recommended number of population size is 11 * dec_num - 1"
        )
        self.theta_ = None
        self.weight_ = None
        self.ideal_point_ = None
        self.weight_num_ = 0
        self.restart_num_ = restart_num_
        self.window_size_ = window_size
        self.bound_ = []

    def Solve(self, problem, global_):
        self.global_ = global_
        self.problem_ = problem

        self.Initialization(problem)

        while not self.IsTermination():
            # Randomly select a weight vector and preprocess the data
            lamda_ = self.weight_[np.random.randint(self.weight_num_)]

            # Calculate Chebyshev scalarization function value
            for i in range(self.real_popnum_):
                CalInverseChebycheff(
                    self.global_.parent_population_[i],
                    lamda_,
                    self.ideal_point_,
                    self.global_.obj_num_,
                )

            if self.real_popnum_ > 11 * self.global_.dec_num_ - 1 + self.window_size_:
                sorted_list = sorted(
                    self.global_.parent_population_,
                    key=lambda x: x.fitness_,
                    reverse=True,
                )[: 11 * self.global_.dec_num_ - 1 + self.window_size_]
                PDec = np.array([item.dec_ for item in sorted_list])
                PCheby = np.array([item.fitness_ for item in sorted_list])
            else:
                PDec = np.array([item.dec_ for item in self.global_.parent_population_])
                PCheby = np.array(
                    [item.fitness_ for item in self.global_.parent_population_]
                )

            # Eliminate the solutions having duplicated inputs or outputs
            PDec_rounded = np.round(PDec * 1e6) / 1e6
            PCheby_rounded = np.round(PCheby * 1e6) / 1e6

            # Find the indices of unique rows in PDec and PCheby
            _, distinct1 = np.unique(PDec_rounded, axis=0, return_index=True)
            _, distinct2 = np.unique(PCheby_rounded, return_index=True)

            # Find the intersection of the indices to get unique rows in both
            distinct = np.intersect1d(distinct1, distinct2)

            # Use the indices to filter the original PDec and PCheby
            PDec = PDec[distinct]
            PCheby = PCheby[distinct]
            # Surrogate-assisted prediction
            if self.optimizer is None:
                self.optimizer = BayesianOptimization(
                    f=None,
                    domain=self.bounds,
                    model=self.model,
                    acquisition_type="EI",
                    normalize_Y=True,
                    X=PDec,
                    Y=PCheby.reshape(-1, 1),
                )
            else:
                self.optimizer.X = PDec
                self.optimizer.Y = PCheby.reshape(-1, 1)
            newX = self.optimizer.suggest_next_locations()
            self.global_.parent_population_[self.real_popnum_].dec_ = newX[0]
            self.EvaluateInd(
                self.global_.parent_population_[self.real_popnum_], self.problem_
            )
            self.real_popnum_ += 1

            self.UpdateIdealpoint(
                self.global_.parent_population_,
                self.real_popnum_,
                self.global_.obj_num_,
            )

    def Initialization(self, problem):
        # initialize the weight vectors
        self.weight_, self.weight_num_ = UniformPoint(
            self.global_.dec_num_, self.global_.obj_num_
        )

        # initialize the surrogate model
        kernel = GPy.kern.RBF(input_dim=self.global_.dec_num_)
        self.model = GPModel(kernel=kernel, optimize_restarts=1)
        self.optimizer = None
        self.bounds = []
        for i in range(self.global_.dec_num_):
            self.bounds.append(
                {
                    "name": f"var_{i}",
                    "type": "continuous",
                    "domain": (
                        self.problem_.lower_bound_[i],
                        self.problem_.upper_bound_[i],
                    ),
                }
            )

        # initialize the population
        self.real_popnum_ = self.global_.population_num_
        self.global_.pop_num_ = self.global_.max_evaluation_
        self.global_.ResetPopulationSize(self.global_.max_evaluation_)
        self.global_.InitializePopulation(
            self.global_.parent_population_, self.real_popnum_, problem
        )
        self.EvaluatePop(self.global_.parent_population_, self.real_popnum_, problem)
        self.ideal_point_ = [constants.EMOC_INF for _ in range(self.global_.obj_num_)]
        self.UpdateIdealpoint(
            self.global_.parent_population_, self.real_popnum_, self.global_.obj_num_
        )
        self.theta_ = 10 * np.ones(self.global_.dec_num_)

        # self.optimizer = BayesianOptimization(f=None, domain=self.bound_, model=self.model,acquisition_type='EI', normalize_Y=True, X = , Y= )

    def UpdateIdealpoint(self, pop, pop_num, obj_num):
        for i in range(pop_num):
            for j in range(obj_num):
                if pop[i].obj_[j] < self.ideal_point_[j]:
                    self.ideal_point_[j] = pop[i].obj_[j]
