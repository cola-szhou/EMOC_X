from emoc import EMOC


def BitFlipMutation(ind, mu_pro):
    mutation_param = EMOC.MutationParameter()
    mutation_param.pro = mu_pro
    EMOC.BitFlipMutation(ind, mutation_param)
