from emoc import EMOC

def SwapMutation(ind, mu_pro):
    mutation_param = EMOC.MutationParameter()
    mutation_param.pro = mu_pro
    EMOC.SwapMutation(ind, mutation_param)