import random
import torch

import numpy as np


def set_random_seed(seed):
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def renorm(ps):
    norm_p = [p/sum(ps) for p in ps]
    return norm_p
