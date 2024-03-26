from emoc.problem import Problem
from emoc import EMOC


class ZDT1(EMOC.ZDT1):
    def __init__(self, dec_num, obj_num):
        EMOC.ZDT1.__init__(self, dec_num, obj_num)
        self.dec_num_ = dec_num
        self.obj_num_ = obj_num
        self.lower_bound_ = [0.0] * dec_num
        self.upper_bound_ = [1.0] * dec_num
        self.name = "ZDT1"

    def CalObj(self, ind):
        self.cal_obj(ind)


class ZDT2(EMOC.ZDT2):
    def __init__(self, dec_num, obj_num):
        EMOC.ZDT2.__init__(self, dec_num, obj_num)
        self.dec_num_ = dec_num
        self.obj_num_ = obj_num
        self.lower_bound_ = [0.0] * dec_num
        self.upper_bound_ = [1.0] * dec_num
        self.name = "ZDT2"

    def CalObj(self, ind):
        self.cal_obj(ind)
