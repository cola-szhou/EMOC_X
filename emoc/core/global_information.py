from emoc import EMOC
import random

class Global(EMOC.Py_Global):
    def __init__(self):
        EMOC.Py_Global.__init__(self)
        
    def InitializePopulation(self, pop, pop_num, problem_):
        # When the initial population is set before algorithm starts, return directly. 
        if self.iteration_num_ == 0 and self.is_customized_init_pop_:
            return
        for i in range(pop_num):
            self.InitializeIndividual(pop[i], problem_)
    
    def InitializeIndividual(self, ind, problem_):
        if problem_.encoding_ == EMOC.EncodingType.REAL:
            ind.dec_ = [random.uniform(problem_.lower_bound_[i], problem_.upper_bound_[i]) for i in range(self.dec_num_)]
        elif problem_.encoding_ == EMOC.EncodingType.BINARY:
            ind.dec_ = [random.randint(0, 1) for _ in range(self.dec_num_)]
        elif problem_.encoding_ == EMOC.EncodingType.PERMUTATION:
            ind.dec_ = list(range(self.dec_num_))
            random.shuffle(ind.dec_)
            
    def Restart(self):
        self.iteration_num_ = 0
        self.current_evaluation_ = 0
    
    def SetCustomInitialPop(self, initial_pop):
        initial_pop_num = len(initial_pop)
        initial_dec_dim = len(initial_pop[0])
        if initial_pop_num != self.population_num_:
            raise RuntimeError("Initial population number is not equal to the set parameter!")
        if initial_dec_dim != self.dec_num_:
            raise RuntimeError("Initial population decision dimensions are not equal to the set parameter!")
        
        for i in range(initial_pop_num):
            for j in range(initial_dec_dim):
                self.parent_population_[i].dec_[j] = initial_pop[i][j]

        self.is_customized_init_pop_ = True
