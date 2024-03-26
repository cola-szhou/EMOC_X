from emoc.algorithm import Algorithm
from emoc.operator import TournamentByRank, SBX, PolynomialMutation
from emoc.operator import FloatRandomSampling, NonDominatedSort
from emoc.core import constants


class DistanceInfo:
    def __init__(self, index, distance):
        self.index = index
        self.distance = distance


class NSGA2Framework(Algorithm):
    def __init__(
        self,
        sampling=FloatRandomSampling(),
        selection=TournamentByRank(),
        crossover=SBX(),
        mutation=PolynomialMutation(),
    ):
        super().__init__()
        self.sampling_ = sampling
        self.selection_ = selection
        self.crossover_ = crossover
        self.mutation_ = mutation

    def Solve(self, problem, global_):
        self.global_ = global_
        self.problem_ = problem
        self.Initialization()
        while not self.IsTermination():
            self.crossover_(
                self.global_.parent_population_,
                self.global_.offspring_population_,
                self.real_popnum_,
                self.problem_,
                self.selection_,
            )
            self.mutation_(
                self.global_.offspring_population_,
                2 * (self.real_popnum_ // 2),
                self.problem_,
            )
            self.EvaluatePop(
                self.global_.offspring_population_,
                2 * (self.real_popnum_ // 2),
                self.problem_,
            )
            self.MergePopulation(
                self.global_.parent_population_,
                self.global_.offspring_population_,
                self.global_.mixed_population_,
            )
            self.EnvironmentalSelection(
                self.global_.parent_population_, self.global_.mixed_population_
            )

    def Initialization(self):
        self.real_popnum_ = self.global_.population_num_
        # Initialize the parent population
        if self.global_.is_customized_init_pop_ == False:
            self.sampling_(
                self.global_.parent_population_, self.real_popnum_, self.problem_
            )
        # Evaluate the parent population
        self.EvaluatePop(
            self.global_.parent_population_, self.real_popnum_, self.problem_
        )

    def EnvironmentalSelection(self, parent_pop, mixed_pop):
        current_popnum = 0
        rank_index = 0
        mixed_popnum = self.global_.population_num_ + 2 * (
            self.global_.population_num_ // 2
        )

        NonDominatedSort(mixed_pop, mixed_popnum, self.global_.obj_num_)

        while True:
            temp_number = 0
            for i in range(mixed_popnum):
                if mixed_pop[i].rank_ == rank_index:
                    temp_number += 1
            if current_popnum + temp_number <= self.global_.population_num_:
                for i in range(mixed_popnum):
                    if mixed_pop[i].rank_ == rank_index:
                        self.CopyIndividual(mixed_pop[i], parent_pop[current_popnum])
                        current_popnum += 1
                rank_index += 1
            else:
                break

        # select individuals by crowding distance
        if current_popnum < self.global_.population_num_:
            pop_sort, sort_num = self.CrowdingDistance(
                mixed_pop, mixed_popnum, rank_index
            )
            while True:
                if current_popnum < self.global_.population_num_:
                    sort_num -= 1
                    index = pop_sort[sort_num]
                    self.CopyIndividual(
                        mixed_pop[pop_sort[sort_num]], parent_pop[current_popnum]
                    )
                    current_popnum += 1
                else:
                    break
        for i in range(self.global_.population_num_):
            parent_pop[i].fitness_ = 0
        return parent_pop

    def CrowdingDistance(self, mixed_pop, pop_num, rank_index):
        num_in_rank = 0
        sort_arr = []
        distanceinfo_vec = []

        for i in range(pop_num):
            mixed_pop[i].fitness_ = 0
            if mixed_pop[i].rank_ == rank_index:
                distanceinfo_vec.append(DistanceInfo(i, 0.0))
                sort_arr.append(i)
                num_in_rank += 1

        for i in range(self.global_.obj_num_):
            # sort the population with i-th obj
            sort_arr.sort(key=lambda index: mixed_pop[index].obj_[i])

            # set the first and last individual with INF fitness (crowding distance)
            mixed_pop[sort_arr[0]].fitness_ = constants.EMOC_INF
            self.SetDistanceInfo(distanceinfo_vec, sort_arr[0], constants.EMOC_INF)
            mixed_pop[sort_arr[num_in_rank - 1]].fitness_ = constants.EMOC_INF

            # calculate each solution's crowding distance
            for j in range(1, num_in_rank - 1):
                if constants.EMOC_INF != mixed_pop[sort_arr[j]].fitness_:
                    if (
                        mixed_pop[sort_arr[num_in_rank - 1]].obj_[i]
                        == mixed_pop[sort_arr[0]].obj_
                    ):
                        mixed_pop[sort_arr[j]].fitness_ += 0
                    else:
                        distance = (
                            mixed_pop[sort_arr[j + 1]].obj_[i]
                            - mixed_pop[sort_arr[j - 1]].obj_[i]
                        ) / (
                            mixed_pop[sort_arr[num_in_rank - 1]].obj_[i]
                            - mixed_pop[sort_arr[0]].obj_[i]
                        )
                        mixed_pop[sort_arr[j]].fitness_ += distance
                        self.SetDistanceInfo(distanceinfo_vec, sort_arr[j], distance)

        distanceinfo_vec = sorted(distanceinfo_vec, key=lambda x: x.distance)

        pop_sort = [0] * num_in_rank
        for i in range(num_in_rank):
            pop_sort[i] = distanceinfo_vec[i].index

        return pop_sort, num_in_rank

    def SetDistanceInfo(self, distanceinf_vec, target_index, distance):
        for i in range(len(distanceinf_vec)):
            if distanceinf_vec[i].index == target_index:
                distanceinf_vec[i].distance += distance
                break
