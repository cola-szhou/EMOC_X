from emoc.utils.utility import CheckDominance, CheckDominanceWithConstraint
import EMOC


def NonDominatedSort(pop, pop_num, obj_num, is_consider_cons=False):
    EMOC.NonDominatedSort(pop, pop_num, obj_num, is_consider_cons)
