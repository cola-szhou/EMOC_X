from emoc.core import constants
from emoc import EMOC

def CheckDominance(ind1, ind2, obj_num):
    flag1 = 0
    flag2 = 0
    for i in range(obj_num):
        if ind1.obj_[i] < ind2.obj_[i]:
            flag1 = 1
        elif ind1.obj_[i] > ind2.obj_[i]:
            flag2 = 1
    
    if flag1 == 1 and flag2 == 0:
        return 1
    elif flag1 == 0 and flag2 == 1:
        return -1
    else:
        return 0

def CheckDominanceWithConstraint(ind1, ind2, obj_num):
    is_infeasible1 = False
    is_infeasible2 = False
    cons1 = 0.0
    cons2 = 0.0
    for i in range(len(ind1.con_)):
        cons1 += max(0.0, ind1.con_[i])
        cons2 += max(0.0, ind2.con_[i])
        if ind1.con_[i] > 0:
            is_infeasible1 = True
        if ind2.con_[i] > 0:
            is_infeasible2 = True
    
    if is_infeasible1 == False and is_infeasible2 == False:
        return CheckDominance(ind1, ind2, obj_num)
    elif is_infeasible1 == True and is_infeasible2 == False:
        return -1
    elif is_infeasible1 == False and is_infeasible2 == True:
        return 1
    else:
        if abs(cons1-cons2) < constants.EMOC_EPS:
            return 0
        elif cons1 < cons2:
            return 1
        else:
            return -1
    
def CalInverseChebycheff(ind, weight_vector, ideal_point, obj_num):
    fitness = EMOC.CalInverseChebycheff(ind, weight_vector, ideal_point, obj_num)
    return fitness

def LoadPFData(filepath):
    try:
        with open(filepath, 'r') as f:
            return [list(map(float, line.strip().split())) for line in f]
            
    except IOError:
        print(f'Error: File {filepath} not found or could not be opened!')
        return []