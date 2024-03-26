from emoc import EMOC

def HV(pop, ref_point):
    """
    Hypervolume (HV) metric
    """
    emo = EMOC(pop, ref_point)
    return emo.hv()