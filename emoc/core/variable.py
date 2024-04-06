import EMOC

EncodingType = {
    "REAL": EMOC.EncodingType.REAL,
    "BINARY": EMOC.EncodingType.BINARY,
    "INTEGER": EMOC.EncodingType.INTEGER,
    "CATEGORICAL": EMOC.EncodingType.CATEGORICAL,
    "PERMUTATION": EMOC.EncodingType.PERMUTATION,
    "MIXED": EMOC.EncodingType.MIXED,
}


class Variable(EMOC.Variable):
    def __init__(self, name=""):
        EMOC.Variable.__init__(self, name)
        self.name = name


class BinaryVariable(EMOC.BinaryVariable):
    def __init__(self, name=""):
        EMOC.BinaryVariable.__init__(self, name)
        self.name = name
        self.encoding_ = EncodingType["BINARY"]

    def Sample(self):
        return EMOC.BinaryVarible.Sample(self)


class RealVariable(EMOC.RealVariable):
    def __init__(self, lower_bound, upper_bound, name=""):
        EMOC.RealVariable.__init__(self, lower_bound, upper_bound, name)
        self.lower_bound_ = lower_bound
        self.upper_bound_ = upper_bound
        self.name = name
        self.encoding_ = EncodingType["REAL"]

    def Sample(self):
        return self.Sample(self)


class IntegerVariable(EMOC.IntegerVariable):
    def __init__(self, lower_bound, upper_bound, name=""):
        EMOC.IntegerVariable.__init__(self, lower_bound, upper_bound, name)
        self.lower_bound_ = lower_bound
        self.upper_bound_ = upper_bound
        self.name = name
        self.encoding_ = EncodingType["INTEGER"]

    def Sample(self):
        return self.Sample(self)


class CategoricalVariable(EMOC.CategoricalVariable):
    def __init__(self, categories, name=""):
        EMOC.CategoricalVariable.__init__(self, categories, name)
        self.categories_ = categories
        self.name = name
        self.encoding_ = EncodingType["CATEGORICAL"]

    def Sample(self):
        return self.Sample(self)


class PermutationVariable(EMOC.PermutationVariable):
    def __init__(self, size, name=""):
        EMOC.PermutationVariable.__init__(self, size, name)
        self.size_ = size
        self.name = name
        self.encoding_ = EncodingType["PERMUTATION"]

    def Sample(self):
        return EMOC.PermutationVariable.Sample(self)


class DecisionSpace(EMOC.DecisionSpace):
    def __init__(self):
        EMOC.DecisionSpace.__init__(self)

    def AddVariable(self, variable):
        EMOC.DecisionSpace.AddVariable(self, variable)

    def RemoveVariable(self, index):
        EMOC.DecisionSpace.RemoveVariable(self, index)

    def ModifyVariable(self, index, variable):
        EMOC.DecisionSpace.ModifyVariable(self, index, variable)

    def GetLowerBound(self, index):
        return EMOC.DecisionSpace.GetLowerBound(self, index)

    def GetUpperBound(self, index):
        return EMOC.DecisionSpace.GetUpperBound(self, index)

    def Sample(self, index):
        return EMOC.DecisionSpace.Sample(self, index)

    def GetCategoricalSpace(self, index):
        return EMOC.DecisionSpace.GetCategoricalSpace(self, index)
