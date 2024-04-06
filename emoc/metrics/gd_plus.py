import EMOC


def CalculateGDPlus(pop, pf_data):
    if len(pop[0].obj_) != len(pf_data[0]):
        raise Exception(
            "Error: the dimension of objective is not equal to the dimension of pf_data"
        )
    return EMOC.CalculateGDPlus(pop, pf_data)
