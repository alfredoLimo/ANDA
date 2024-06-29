import numpy as np
from collections import Counter
from scipy.stats import truncnorm

import torch
import torch.nn.functional as F

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

def split_trND_teDR_Px(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    client_number: int = 10,
    rotation_bank: int = 1,
    color_bank: int = 1,
    scaling_low: float = 0.5,
    scaling_high: float = 0.5,
    reverse_test: bool = False,
    verbose: bool = True
) -> list:
    '''
    Split the dataset into distributions as:
        Training: A (large in size)
        Testing: B (unseen)
    with distribution difference in P(x)
    for A SINGLE CLIENT. (overall skew among clients exists)

    Args:
        train_features (torch.Tensor): The training features.
        train_labels (torch.Tensor): The training labels.
        test_features (torch.Tensor): The testing features.
        test_labels (torch.Tensor): The testing labels.
        client_number (int): The number of clients.
        rotation_bank (int): The number of rotation patterns. 1 as no rotation.
        color_bank (int): The number of color patterns. 1 as no color.
        scaling_low (float): The lower bound of scaling.
        scaling_high (float): The upper bound of scaling.
        reverse_test (bool): Testing and training use reverse patterns. (used to create strong unseen level)
        verbose (bool): Whether to print the distribution information.

    Description:
        A pattern bank will be created based on #rotation and #color.
        Each DATAPOINT applies one of the patterns in the bank.
        Each sub-dataset choose any distribution of the patterns.
        
    Warning:
        SAMPLE, NOT CHOICES: original datapoints are not repeated.
        Recommended bank size is:
            rotation 2 * color 2 = 4, or
            rotation 2 * color 3 = 6, or
            rotation 4 * color 2 = 8.

        Balance the "unseen" level with scaling.    

    Returns:
        list: A list of dictionaries where each dictionary contains the features and labels for each client.
                Both train and test.
    '''
    assert len(train_features) == len(train_labels), "The number of samples in features and labels must be the same."
    assert len(test_features) == len(test_labels), "The number of samples in features and labels must be the same."
    assert rotation_bank > 0, "The number of rotation patterns must be greater than 0."
    assert color_bank > 0, "The number of color patterns must be greater than 0."
    assert scaling_high >= scaling_low, "The upper bound of scaling must be greater than the lower bound."

    # generate basic split
    basic_split_data_train = split_basic(train_features, train_labels, client_number)
    basic_split_data_test = split_basic(test_features, test_labels, client_number)
    
    # generate pattern bank
    angles = [i * 360 / rotation_bank for i in range(rotation_bank)] if rotation_bank > 1 else [0.0]

    if color_bank == 1:
        colors = ['gray']
    elif color_bank == 2:
        colors = ['red', 'blue']
    elif color_bank == 3:
        colors = ['red', 'blue', 'green']
    else:
        raise ValueError("The number of color patterns must be 1, 2, or 3.")

    pattern_bank = [[angle, color] for angle in angles for color in colors]
    # assign patterns to each client
    client_Count = 0
    print("Showing patterns for each client..") if verbose else None

    for client_data_train, client_data_test in zip(basic_split_data_train, basic_split_data_test):
        print(f"Client: {client_Count}") if verbose else None

        # for training test
        train_pattern = np.random.permutation(pattern_bank).tolist()
        train_pattern = [(float(angle), color) for angle, color in train_pattern]

        scaled_values = np.arange(len(pattern_bank), 0, -1) * np.random.uniform(scaling_low,scaling_high)
        exp_values = np.exp(scaled_values)
        train_prob = exp_values / np.sum(exp_values)

        print("Train bank: ", train_pattern) if verbose else None
        print("Assigned probability: ", train_prob) if verbose else None

        indices = np.arange(len(train_pattern))
        sampled_indices = np.random.choice(indices, size=len(client_data_train['labels']), p=train_prob)
        sampled_pattern = [train_pattern[i] for i in sampled_indices]

        angles_assigned = [item[0] for item in sampled_pattern]
        colors_assigned = [item[1] for item in sampled_pattern]
        client_data_train['features'] = rotate_dataset(client_data_train['features'], angles_assigned)
        client_data_train['features'] = color_dataset(client_data_train['features'], colors_assigned)


        # for testing test
        test_pattern = np.random.permutation(pattern_bank).tolist()
        test_pattern = list(reversed(train_pattern)) if reverse_test else test_pattern
        test_pattern = [(float(angle), color) for angle, color in test_pattern]

        scaled_values = np.arange(len(pattern_bank), 0, -1) * np.random.uniform(scaling_low,scaling_high)
        exp_values = np.exp(scaled_values)
        test_prob = exp_values / np.sum(exp_values)

        print("Test bank: ", test_pattern) if verbose else None
        print("Assigned probability: ", test_prob) if verbose else None

        indices = np.arange(len(test_pattern))
        sampled_indices = np.random.choice(indices, size=len(client_data_test['labels']), p=test_prob)
        sampled_pattern = [test_pattern[i] for i in sampled_indices]

        angles_assigned = [item[0] for item in sampled_pattern]
        colors_assigned = [item[1] for item in sampled_pattern]
        client_data_test['features'] = rotate_dataset(client_data_test['features'], angles_assigned)
        client_data_test['features'] = color_dataset(client_data_test['features'], colors_assigned)

        client_Count += 1

    rearranged_data = []
    # Iterate through the indices of the lists
    for i in range(client_number):
        # Create a new dictionary for each client
        client_data = {
            'train_features': basic_split_data_train[i]['features'],
            'train_labels': basic_split_data_train[i]['labels'],
            'test_features': basic_split_data_test[i]['features'],
            'test_labels': basic_split_data_test[i]['labels']
        }
        # Append the new dictionary to the list
        rearranged_data.append(client_data)
            
    return rearranged_data

def split_trND_teDR_Py(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    client_number: int = 10,
    rotation_bank: int = 1,
    color_bank: int = 1,
    scaling_low: float = 0.5,
    scaling_high: float = 0.5,
    reverse_test: bool = False,
    verbose: bool = True
) -> list:
    '''
    Split the dataset into distributions as:
        Training: A (large in size)
        Testing: B (unseen)
    with distribution difference in P(y)
    for A SINGLE CLIENT. (overall skew among clients exists)

    Args:

    Description:
        
    Warning:

        Balance the "unseen" level with scaling.    

    Returns:
        list: A list of dictionaries where each dictionary contains the features and labels for each client.
                Both train and test.
    '''
    pass