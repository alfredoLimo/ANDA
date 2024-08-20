import numpy as np
import torch
from .utils import *

# For reproducibility only
def set_seed(
    RANDOM_SEED: int = 42
):
    '''
    Set the random seed for reproducibility.
    
    Args:
        RANDOM_SEED (int): The random seed to set.
    '''
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

def split_trDA_teDR_Px(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    client_number: int = 10,
    dataset_scaling: float = 1.0,
    rotation_bank: int = 1,
    color_bank: int = 1,
    epoch_locker_num: int = 10,
    random_locker: bool = False,
    px_scaling_low: float = 0.5,
    px_scaling_high: float = 0.5,
    verbose: bool = True
) -> list:
    pass