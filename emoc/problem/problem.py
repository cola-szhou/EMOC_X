import EMOC

from emoc.core.variable import EncodingType


class Problem(EMOC.Problem):
    def __init__(self, dec_num, obj_num):
        EMOC.Problem.__init__(self, dec_num, obj_num)
        self.name = "Problem"

    def CalObj(self, ind):
        pass

    def CalCon(self, ind):
        pass

    def GetType(self, type):
        if type in EncodingType:
            return EncodingType[type]
        else:
            print(
                "Error Type, please select from REAL, BINARY, INTEGER, CATEGORICAL, PERMUTATION or MIXED"
            )
