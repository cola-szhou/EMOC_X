import EMOC


class CMOEAD:
    def __init__(self):
        self.algorithm = None

    def Solve(self, problem, global_):
        self.algorithm = EMOC.CMOEAD(global_, problem)
        self.algorithm.Solve()

    def PrintResult(self, r):
        self.algorithm.PrintResult()
