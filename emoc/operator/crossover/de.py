import EMOC


def DE(
    parent1,
    parent2,
    parent3,
    offspring,
    lower_bound,
    upper_bound,
    cross_pro,
    cross_index,
):
    crossover_param = EMOC.CrossoverParameter()
    crossover_param.pro = cross_pro
    crossover_param.index1 = cross_index
    EMOC.DE(
        parent1, parent2, parent3, offspring, [0.0] * 10, [1.0] * 10, crossover_param
    )
