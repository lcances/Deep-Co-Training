import numpy as np
from ramps import sigmoid_rampup, sigmoid_rampdown

def uniform_rule(**kwargs):
    """ The probability to apply each augmentations is even during all the training."""
    sup_steps = [0.34 for _ in range(steps)]
    cot_steps = [0.33 for _ in range(steps)]
    diff_steps = [0.33 for _ in range(steps)]
    
    return sup_steps, cot_steps, diff_steps

def weighted_uniform_rule(lambda_sup_max: int = 1, lambda_cot_max: int = 1, lambda_diff_max: int = 1, **kwargs):
    """ The constant probability are weighted using this three lambda value.
    
    :param lamdda_sup_max: Weight of the Lsup loss.
    :param lambda_cot_max: Weight of the Lcot loss.
    :param lambda_dif_max: Weight of the Ldiff loss.
    """
    lsm, lcm, ldm = lambda_sup_max, lambda_cot_max, lambda_diff_max

    sup_steps = [lsm * 0.34 for _ in range(steps)]
    cot_steps = [lcm * 0.33 for _ in range(steps)]
    diff_steps = [ldm * 0.33 for _ in range(steps)]
    
    # normalize
    for i in range(steps):
        summed = sup_steps[i] + cot_steps[i] + diff_steps[i]
        sup_steps[i] /= summed
        cot_steps[i] /= summed
        diff_steps[i] /= summed
        
    return sup_steps, cot_steps, diff_steps

def linear_rule(nb_epoch: int, steps: int = 10, plsup_mini: float = 0.0, **kwargs):
    """ The probability to apply each augmentation linearly increate or decrease during the training.

    :param nb_epoch: The number of epoch the rule will be effective.
    :param steps: The number of time the probabilities will be updated.
    :param plsup_mini: The minimum probability for the supervised loss.
    """
    hop_length = np.linspace(0, nb_epoch, steps)
    
    s_start, s_end = 1, plsup_mini
    c_start, c_end = 0, 0.5 - (plsup_mini / 2)
    d_start, d_end = 0, 0.5 - (plsup_mini / 2)
    
    sup_steps = np.linspace(s_start, s_end, steps)
    cot_steps = np.linspace(d_start, c_end, steps)
    diff_steps = np.linspace(d_start, d_end, steps)

     # normalize
    for i in range(steps):
        summed = sup_steps[i] + cot_steps[i] + diff_steps[i]
        if summed != 1:
            sup_steps[i] /= summed
            cot_steps[i] /= summed
            diff_steps[i] /= summed
    
    return sup_steps, cot_steps, diff_steps


def weighted_linear_rule(
        nb_epoch: int, steps: int = 10, plsup_mini = 0.0,
        lambda_sup_max: int = 1, lambda_cot_max: int = 1, lambda_diff_max: int = 1,
        **kwargs):
    """ The probability to apply each augmentation linearly increase or decrease during the training.
    Those probabilities are weighted using three lambda value.
    
    :param nb_epoch: The number of epoch the rule will be effective.
    :param steps: The number of time the probabilities will be updated.
    :param plsup_mini: The minimum probability for the supervised loss.
    :param lamdda_sup_max: Weight of the Lsup loss.
    :param lambda_cot_max: Weight of the Lcot loss.
    :param lambda_dif_max: Weight of the Ldiff loss.
    """
    lsm, lcm, ldm = lambda_sup_max, lambda_cot_max, lambda_diff_max
    
    s_start, s_end = 1, lsm * (plsup_mini)
    c_start, c_end = 0, lcm * (0.5 - (plsup_mini / 2))
    d_start, d_end = 0, ldm * (0.5 - (plsup_mini / 2))
    
    # normalize
    total = s_end + c_end + d_end
    s_end /= total
    c_end /= total
    d_end /= total
    
    sup_steps = np.linspace(s_start, s_end, steps)
    cot_steps = np.linspace(d_start, c_end, steps)
    diff_steps = np.linspace(d_start, d_end, steps)
    
    return sup_steps, cot_steps, diff_steps

def cosine_rule(nb_epoch: int, steps: int = 10, cycle: int = 1, **kwargs):
    """ The probability to apply each augmentation follow a cosinus curve.

    :param nb_epoch: The number of epoch the rule will be effective.
    :param steps: The number of time the probabilities will be updated.
    :param cycle: The number of half period the cosinus curve will follow.
    """
    hop_length = np.linspace(0, nb_epoch, steps * cycle)
    
    sup_steps = 0.5 * (np.cos(np.pi * hop_length / (nb_epoch / cycle)) + 1)
    cot_steps = 1 - (0.5 * (np.cos(np.pi * hop_length / (nb_epoch / cycle)) + 1))
    diff_steps =  1 - (0.5 * (np.cos(np.pi * hop_length / (nb_epoch / cycle)) + 1))
    
    # normalize
    for i in range(steps * cycle):
        summed = sup_steps[i] + cot_steps[i] + diff_steps[i]
        sup_steps[i] /= summed
        cot_steps[i] /= summed
        diff_steps[i] /= summed
        
    return sup_steps, cot_steps, diff_steps

def weighted_cosine_rule(nb_epoch: int, steps: int = 10, cycle: int = 1, **kwargs):
    """ The probability to apply each augmentation follow a cosinus curve.
    Those probabilities are weighted using three lambda value.

    :param nb_epoch: The number of epoch the rule will be effective.
    :param steps: The number of time the probabilities will be updated.
    :param cycle: The number of half period the cosinus curve will follow.
    :param lamdda_sup_max: Weight of the Lsup loss.
    :param lambda_cot_max: Weight of the Lcot loss.
    :param lambda_dif_max: Weight of the Ldiff loss.
    """
    lsm, lcm, ldm = lambda_sup_max, lambda_cot_max, lambda_diff_max
    
    hop_length = np.linspace(0, nb_epoch, steps * cycle)
    
    sup_steps = 0.5 * (np.cos(np.pi * hop_length / (nb_epoch / cycle)) + 1)
    cot_steps = 1 - (0.5 * (np.cos(np.pi * hop_length / (nb_epoch / cycle)) + 1))
    diff_steps =  1 - (0.5 * (np.cos(np.pi * hop_length / (nb_epoch / cycle)) + 1))
    
    sup_steps *= lsm
    cot_steps *= lcm
    diff_steps *= ldm
    
    # normalize
    for i in range(steps * cycle):
        summed = sup_steps[i] + cot_steps[i] + diff_steps[i]
        sup_steps[i] /= summed
        cot_steps[i] /= summed
        diff_steps[i] /= summed
        
    return sup_steps, cot_steps, diff_steps

def annealing_cosine_rule(nb_epoch: int, steps: int = 10, cycle: int = 1, beta: float = 2, **kwargs):
    """ The probability to apply each augmentation follow a cosinus curve that decay with time.

    :param nb_epoch: The number of epoch the rule will be effective.
    :param steps: The number of time the probabilities will be updated.
    :param cycle: The number of half period the cosinus curve will follow.
    :param beta: The decay rate that follow the equation e^(-beta * t)
    """
    hop_length = np.linspace(0, nb_epoch, steps * cycle)
    
    # create original steps
    decaying =  np.exp(-beta * np.linspace(0, 1, len(hop_length)))
    
    sup_steps = 0.5 * (decaying * np.cos(np.pi * hop_length / (nb_epoch / cycle)) + 1)
    cot_steps = 1 - (0.5 * (decaying * np.cos(np.pi * hop_length / (nb_epoch / cycle)) + 1))
    diff_steps =  1 - (0.5 * (decaying * np.cos(np.pi * hop_length / (nb_epoch / cycle)) + 1))
    
    # normalize
    for i in range(steps * cycle):
        summed = sup_steps[i] + cot_steps[i] + diff_steps[i]
        sup_steps[i] /= summed
        cot_steps[i] /= summed
        diff_steps[i] /= summed
        
    return sup_steps, cot_steps, diff_steps

def weighted_annealing_cosine_rule(nb_epoch: int, steps: int = 10, cycle: int = 1, beta: float = 2, **kwargs):
    """ The probability to apply each augmentation follow a cosinus curve that decay with time.
    Those probabilities are weighted using three lambda value.

    :param nb_epoch: The number of epoch the rule will be effective.
    :param steps: The number of time the probabilities will be updated.
    :param cycle: The number of half period the cosinus curve will follow.
    :param beta: The decay rate that follow the equation e^(-beta * t)
    :param lamdda_sup_max: Weight of the Lsup loss.
    :param lambda_cot_max: Weight of the Lcot loss.
    :param lambda_dif_max: Weight of the Ldiff loss.
    """
    lcm = kwargs.get("lambda_cot_max", 1)
    ldm = kwargs.get("lambda_diff_max", 1)
    lsm = kwargs.get("lambda_sup_max", 1)
    
    hop_length = np.linspace(0, nb_epoch, steps * cycle)
    
    # create original steps
    decaying =  np.exp(-beta * np.linspace(0, 1, len(hop_length)))
    
    sup_steps = 0.5 * (decaying * np.cos(np.pi * hop_length / (nb_epoch / cycle)) + 1)
    cot_steps = 1 - (0.5 * (decaying * np.cos(np.pi * hop_length / (nb_epoch / cycle)) + 1))
    diff_steps =  1 - (0.5 * (decaying * np.cos(np.pi * hop_length / (nb_epoch / cycle)) + 1))
    
    sup_steps *= lsm
    cot_steps *= lcm
    diff_steps *= ldm
    
    # normalize
    for i in range(steps * cycle):
        summed = sup_steps[i] + cot_steps[i] + diff_steps[i]
        sup_steps[i] /= summed
        cot_steps[i] /= summed
        diff_steps[i] /= summed
        
    return sup_steps, cot_steps, diff_steps

def sigmoid_rule(nb_epoch: int, steps: int = 10, **kwargs):
    """ The probability to apply each augmentation increases or decreases following a sigmoid curve.

    :param nb_epoch: The number of epoch the rule will be effective.
    :param steps: The number of time the probabilities will be updated.
    """
    hop_length = np.linspace(0, nb_epoch, steps)
    
    sup_steps = np.asarray([sigmoid_rampdown(x, nb_epoch) for x in hop_length])
    cot_steps = np.asarray([sigmoid_rampup(x, nb_epoch) for x in hop_length])
    diff_steps = np.asarray([sigmoid_rampup(x, nb_epoch) for x in hop_length])
    
    # normalize
    for i in range(steps):
        summed = sup_steps[i] + cot_steps[i] + diff_steps[i]
        sup_steps[i] /= summed
        cot_steps[i] /= summed
        diff_steps[i] /= summed
        
    return sup_steps, cot_steps, diff_steps

def weighted_sigmoid_rule(nb_epoch: int, steps: int = 10, **kwargs):
    """ The probability to apply each augmentation increases or decreases following a sigmoid curve.
    Those probabilities are weighted using three lambda value.

    :param nb_epoch: The number of epoch the rule will be effective.
    :param steps: The number of time the probabilities will be updated.
    :param lamdda_sup_max: Weight of the Lsup loss.
    :param lambda_cot_max: Weight of the Lcot loss.
    :param lambda_dif_max: Weight of the Ldiff loss.
    """
    lcm = kwargs.get("lambda_cot_max", 1)
    ldm = kwargs.get("lambda_diff_max", 1)
    lsm = kwargs.get("lambda_sup_max", 1)

    hop_length = np.linspace(0, nb_epoch, steps)
    
    sup_steps = np.asarray([lsm * sigmoid_rampdown(x, nb_epoch) for x in hop_length])
    cot_steps = np.asarray([lcm * sigmoid_rampup(x, nb_epoch) for x in hop_length])
    diff_steps = np.asarray([ldm * sigmoid_rampup(x, nb_epoch) for x in hop_length])
    
    # normalize
    for i in range(steps):
        summed = sup_steps[i] + cot_steps[i] + diff_steps[i]
        sup_steps[i] /= summed
        cot_steps[i] /= summed
        diff_steps[i] /= summed
        
    return sup_steps, cot_steps, diff_steps


def rule_maker(rule_fn, nb_epoch: int, steps: int = 10, **kwargs):
    """ Build the schedulers steps for all three loss using the array outputed by the rule functions.
    Can take any argument from any rule function.
    """
    p_lsup, p_lcot, p_ldiff = rule_fn(steps, **kwargs)
    
    hop_length = np.linspace(0, nb_epoch, steps)
    
    rules = dict()
    for i, epoch in enumerate(hop_length):
        rules[epoch] = {"lsup": p_lsup[i], "lcot": p_lcot[i], "ldiff": p_ldiff[i]}
        
    return rules


def loss_chooser(epoch):
    """ Simply return one loss randomly following the probabilities of the current step.
    """
    for k in reversed(rules.keys()):
        if epoch >= k:
            chance = list(rules[k].values())
            break
    
    loss_function = ["sup", "cot", "diff"]
    return np.random.choice(loss_function, p=chance)


def rule_chooser(rule_str):
    """ To manually choose which rule will be applied."""
    mapper = {
        "uniform": uniform_rule,
        "weighted-uniform": weighted_uniform_rule,
        "linear": linear_rule,
        "weighted-linear": weighted_linear_rule,
        "sigmoid": sigmoid_rule,
        "weighted-sigmoid": weighted_sigmoid_rule,
        "cosine": cosine_rule,
        "weighted-cosine": weighted_cosine_rule,
        "annealing-cosine": annealing_cosine_rule,
        "weighted-annealing-cosine": weighted_annealing_cosine_rule,
    }
    
    if rule_str not in mapper:
        raise ValueError("Loss scheduler must be from [%s]" % (", ".join(mapper.keys())))
        
    return mapper[rule_str]
