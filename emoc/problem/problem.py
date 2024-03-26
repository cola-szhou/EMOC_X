from emoc import EMOC


class Problem(EMOC.Problem):
    def __init__(self, dec_num, obj_num):
        EMOC.Problem.__init__(self, dec_num, obj_num)
        self.name = "Problem"

    def CalObj(self, ind):
        pass

    def CalCon(self, ind):
        pass

    def GetType(self, type):
        if type == "REAL":
            return EMOC.EncodingType.REAL
        elif type == "BINARY":
            return EMOC.EncodingType.BINARY
        elif type == "INTEGER":
            return EMOC.EncodingType.INTEGER
        elif type == "PERMUTATION":
            return EMOC.EncodingType.PERMUTATION
        elif type == "MIXED":
            return EMOC.EncodingType.MIXED
        else:
            print("Error Type, please select from REAL, BINARY and PERMUTATION")
