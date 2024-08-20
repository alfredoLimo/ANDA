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
    rotation_bank: int = 1,
    color_bank: int = 1,
    DA_dataset_scaling: float = 1.0,
    DA_epoch_locker_num: int = 10,
    DA_random_locker: bool = False,
    DA_max_dist: int = 2,
    DA_continual_divergence: bool = False,
    px_scaling_low: float = 0.5,
    px_scaling_high: float = 0.5,
    verbose: bool = True
) -> list:
    '''
    Split the dataset into distributions as:
        Training: A-A-AB-ABB-ABB-ABBB-ABBBC-ABBBC (accumulative)
        Testing: D (unseen)
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
        [DA_parameters]: Details in Description.
        px_scaling_low (float): The lower bound of the scaling factor for P(x).
        px_scaling_high (float): The upper bound of the scaling factor for P(x).
        verbose (bool): Whether to print the distribution information.

    Description:
        A pattern bank will be created based on #rotation and #color.
        Each pattern is considered as a type of P(x) distribution (A/B/C ...).
        Each subset taking a certain pattern for all its datapoints.
        
        DA_dataset_scaling (float): The scaling factor for the training dataset.
            (size = original size * dataset_scaling)
        DA_epoch_locker_num (int): The number of dataset growth stages during overall training.
            E.g., 2 means the dataset grows once (two stages), and an indicator float=0.5 will be labeled
            to the subdataset to let known when to grow.
            Accordingly, 3 gives float=0.33,0.67.
        DA_random_locker (bool): Whether clients are randomly assigned to epoch lockers.
            When true, the locker float is randomly generated.
            E.g., when DA_epoch_locker_num=3 with DA_random_locker=True, the locker floats could be 0.20,0.80.
        DA_max_dist (int): The maximum dist types during overall training.
            E.g., when DA_epoch_locker_num=5 with DA_max_dist=3,
            drifting as [A]-[AB]-[ABB]-[ABBB]-[ABBBC] is VALID. 3=(A,B,C)
            drifting as [A]-[AB]-[ABC]-[ABCC]-[ABCCD] is INVALID. 4=(A,B,C,D)
        DA_continual_divergence (bool): Whether the distribution drifts continually.
            E.g. when True,
            drifting as [A]-[AB]-[ABC]-[ABCD]-[ABCDE] is VALID. (continual)
            drifting as [A]-[AB]-[ABC]-[ABCD]-[ABCDA] is INVALID. (back to dist A)
        
    Warning:
        EXTENSION: YES. Datapoints replicated overall. (by dataset_scaling)
        SAMPLE: YES. Datapoints not ever repeated for different clients.
        CHOICES: NO. (contrast to SAMPLE)

        VERBOSE is always recommended for sanity check.

        #TODO to support manual distribution assignment.

    Returns:
        A list of dictionaries contains:
            train set,
            test set,
            client number,
            epoch locker indicator
    '''
    assert len(train_features) == len(train_labels), "The number of samples in features and labels must be the same."
    assert len(test_features) == len(test_labels), "The number of samples in features and labels must be the same."
    assert 0 < rotation_bank, "The number of rotation patterns must be greater than 0."
    assert 0 < color_bank, "The number of color patterns must be greater than 0."
    assert DA_dataset_scaling >= 1, "Invalid downscaling."
    assert DA_epoch_locker_num > 0, "The number of epoch lockers must be greater than 0."
    assert 1 < DA_max_dist < rotation_bank * color_bank, "Distribution assignment out of range."
    assert px_scaling_low <= px_scaling_high, "Invalid scaling range."

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

    pattern_bank = {i + 1: [angle, color] for i, (angle, color)
                    in enumerate([(angle, color) for angle in angles for color in colors])}
    
    print("Pattern bank:") if verbose else None
    print('\n'.join(f"{key}: {value}" for key, value in pattern_bank.items())) if verbose else None

    # generate basic split
    basic_split_data_train = split_basic(train_features, train_labels, client_number)
    basic_split_data_test = split_basic(test_features, test_labels, client_number)
    rearranged_data = []

    client_Count = 0


    for client_data_train, client_data_test in zip(basic_split_data_train, basic_split_data_test):
        print(f"Client: {client_Count}") if verbose else None

        # generate drifting
        dist_bank = list(range(1, rotation_bank * color_bank + 1))
        test_dist = np.random.choice(dist_bank)
        dist_bank.remove(test_dist)
        train_dist = generate_DA_dist(dist_bank,
                                      DA_epoch_locker_num,DA_max_dist,DA_continual_divergence)

        print("Train distribution: ", train_dist) if verbose else None
        print("Test distribution: ", test_dist) if verbose else None


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


        # # for testing test
        # test_pattern = list(reversed(train_pattern)) if reverse_test else np.random.permutation(pattern_bank).tolist()
        # test_pattern = [(float(angle), color) for angle, color in test_pattern]

        # scaled_values = np.arange(len(pattern_bank), 0, -1) * np.random.uniform(scaling_low,scaling_high)
        # exp_values = np.exp(scaled_values)
        # test_prob = exp_values / np.sum(exp_values)

        # print("Test bank: ", test_pattern) if verbose else None
        # print("Assigned probability: ", test_prob,"\n") if verbose else None

        # indices = np.arange(len(test_pattern))
        # sampled_indices = np.random.choice(indices, size=len(client_data_test['labels']), p=test_prob)
        # sampled_pattern = [test_pattern[i] for i in sampled_indices]

        # angles_assigned, colors_assigned = map(list, zip(*sampled_pattern))
        # client_data_test['features'] = rotate_dataset(client_data_test['features'], angles_assigned)
        # client_data_test['features'] = color_dataset(client_data_test['features'], colors_assigned)

        client_Count += 1