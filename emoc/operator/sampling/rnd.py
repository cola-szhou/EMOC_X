import numpy as np
from emoc.core import Individual
from emoc.operator import Sampling


class FloatRandomSampling(Sampling):
    def __init__(self):
        pass

    def sample_ind(self, ind, problem, **kwargs):
        ind.dec_ = np.random.uniform(
            problem.lower_bound_, problem.upper_bound_, problem.dec_num_
        ).tolist()
