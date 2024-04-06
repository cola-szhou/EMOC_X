import importlib.resources as pkg_resources
from typing import Optional
import EMOC

from emoc.utils import constants


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
        if abs(cons1 - cons2) < constants.EMOC_EPS:
            return 0
        elif cons1 < cons2:
            return 1
        else:
            return -1


def CalInverseChebycheff(ind, weight_vector, ideal_point, obj_num):
    fitness = EMOC.CalInverseChebycheff(ind, weight_vector, ideal_point, obj_num)
    return fitness


def LoadPFData(
    pf_path: Optional[str] = None, prob_name: str = None, obj_num: Optional[int] = None
):
    if pf_path is not None:
        try:
            with open(pf_path, "r") as f:
                pf = [list(map(float, line.strip().split())) for line in f]
        except IOError:
            print(f"Error: File {pf_path} not found or could not be opened!")
            pf = []

    else:
        problem_name = prob_name
        count = 0
        for i in range(len(problem_name) - 1, -1, -1):
            if "0" <= problem_name[i] <= "9":
                count += 1
            else:
                break

        temp_problemname = problem_name[:-count] if count > 0 else problem_name

        temp_problemname = "".join(
            c.lower() if not (c.isdigit() or c == "_") else c for c in temp_problemname
        )
        problem_name = "".join(
            c.lower() if not (c.isdigit() or c == "_") else c for c in problem_name
        )
        pf_file_name = problem_name + "." + str(obj_num) + "D.pf"
        with pkg_resources.open_text(
            "emoc.pf_data." + temp_problemname, pf_file_name
        ) as f:
            pf = [list(map(float, line.strip().split())) for line in f]
    return pf
