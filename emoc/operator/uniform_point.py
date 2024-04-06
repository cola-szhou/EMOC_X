import EMOC


def UniformPoint(num, obj_num):
    weight_num = num
    weight = EMOC.UniformPoint(num, weight_num, obj_num)
    return weight, weight_num
