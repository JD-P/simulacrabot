# Zippy's seed_all function
from numpy.random import MT19937, RandomState, SeedSequence
import random, numpy as np, time
import torch

def seed_all(seed_value=None):
    new_state = RandomState(MT19937(SeedSequence(int(time.time_ns()) % (2**32 - 1))))
    np.random.set_state(new_state.get_state())
    if seed_value is None:
        seed_value = np.random.randint(1, 2**32 - 1)
    random.seed(seed_value)  # Python
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False
    return seed_value
