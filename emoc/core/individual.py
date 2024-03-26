from emoc import EMOC
import numpy as np
# __list_attrs__ = ['con_', 'dec_', 'obj_']

class Individual(EMOC.Individual):
# class Individual:
    def __init__(self, dec_num, obj_num):
        EMOC.Individual.__init__(self, dec_num, obj_num)