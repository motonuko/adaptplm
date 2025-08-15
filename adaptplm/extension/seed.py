import os
import random

import numpy as np
import torch


# NOTE: Seeding may not work on MPS devices.
def set_random_seed(seed: int):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.benchmark = False  default: False
