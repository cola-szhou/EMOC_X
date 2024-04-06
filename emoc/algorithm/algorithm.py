from emoc.core.global_information import Global
import time


class Algorithm:
    def __init__(self):
        self.global_ = Global()
        self.problem_ = None
        self.real_popnum_ = 0
        self.start_ = time.time()
        self.end_ = time.time()
        self.runtime_ = 0.0
        self.name = "Algorithm"

    def Solve(self, problem, global_):
        self.global_ = global_
        self.problem_ = problem

    def PrintPop(self):
        pass

    def IsTermination(self):
        # record runtime
        self.end_ = time.time()
        if self.global_.iteration_num_ >= 1:
            self.runtime_ += (float)(self.end_ - self.start_)

        # record the population every interval generations and the first and last genration
        is_terminate = self.global_.current_evaluation_ >= self.global_.max_evaluation_
        if (
            self.global_.iteration_num_ % self.global_.output_interval_ == 0
            or is_terminate
        ):
            self.TrackPopulation()

        if not is_terminate:
            self.global_.iteration_num_ += 1

        self.start_ = time.time()
        return is_terminate

    def EvaluatePop(self, pop, pop_num, problem):
        for i in range(pop_num):
            self.EvaluateInd(pop[i], problem)

    def PrintPop(self, pop, pop_num):
        for i in range(pop_num):
            print("ind {}:".format(i))
            print("fitness: {}".format(pop[i].fitness_))
            print("rank: {}".format(pop[i].rank_))
            print("dec: {}".format(pop[i].dec_))
            print("obj: {}".format(pop[i].obj_))
            print("con: {}".format(pop[i].con_))

    def EvaluateInd(self, ind, problem):
        problem.CalObj(ind)
        problem.CalCon(ind)
        self.global_.current_evaluation_ += 1

    def MergePopulation(self, pop_src1, pop_src2, pop_dest):
        for i in range(len(pop_src1)):
            self.CopyIndividual(pop_src1[i], pop_dest[i])
        for j in range(len(pop_src2)):
            self.CopyIndividual(pop_src2[j], pop_dest[len(pop_src1) + j])

    def CopyIndividual(self, ind_src, ind_dest):
        ind_dest.fitness_ = ind_src.fitness_
        ind_dest.rank_ = ind_src.rank_
        ind_dest.dec_ = ind_src.dec_
        ind_dest.obj_ = ind_src.obj_
        ind_dest.con_ = ind_src.con_

    def PrintResult(self, run):
        """ToDO: print the result of the algorithm, such as igd, hv, gd, etc."""
        obj_num = self.global_.obj_num_
        dec_num = self.global_.dec_num_
        if obj_num > 1:
            pf_size = 0
            # pf_data = EMOC.LoadPFData(pf_size, obj_num, self.problem_.problem_name_)
            print("******Finish Optimization Run {}******".format(run))
            print("time: {}".format(self.runtime_))
            # print("time: {}\t igd: {}\t hv: {}\t gd: {}".format(self.run_time_, igd, hv, gd))

    def TrackPopulation(self):
        self.global_.RecordPop(self.real_popnum_, self.runtime_)
