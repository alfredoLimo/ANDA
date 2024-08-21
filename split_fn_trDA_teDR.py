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
    DA_dataset_scaling: float = 1.5,
    DA_epoch_locker_num: int = 10,
    DA_random_locker: bool = False,
    DA_max_dist: int = 2,
    DA_continual_divergence: bool = False,
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
            train, (bool) whether it is training set
            dataset,
            client number,
            epoch locker indicator, (float) to tell the growth time
            epoch locker order: (int) to tell the order of current subset
    '''
    assert len(train_features) == len(train_labels), "The number of samples in features and labels must be the same."
    assert len(test_features) == len(test_labels), "The number of samples in features and labels must be the same."
    assert 0 < rotation_bank, "The number of rotation patterns must be greater than 0."
    assert 0 < color_bank, "The number of color patterns must be greater than 0."
    assert DA_dataset_scaling >= 1, "Invalid downscaling."
    assert DA_epoch_locker_num > 0, "The number of epoch lockers must be greater than 0."
    assert 1 < DA_max_dist < rotation_bank * color_bank, "Distribution assignment out of range."

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
    
    print("Pattern bank:\n", '\n'.join(f"{key}: {value}" for key, value in pattern_bank.items())) if verbose else None

    # generate basic split
    basic_split_data_train = split_basic(train_features, train_labels, client_number)
    basic_split_data_test = split_basic(test_features, test_labels, client_number)

    rearranged_data = []
    client_Count = 0

    for client_data_train, client_data_test in zip(basic_split_data_train, basic_split_data_test):
        print(f"Client: {client_Count}") if verbose else None
        # training dataset scaling
        original_train_feature = client_data_train['features']
        original_train_label = client_data_train['labels']
        indices = torch.randint(0, original_train_label.shape[0],
                                (int(original_train_label.shape[0] * (DA_dataset_scaling - 1)),))
        sampled_data = original_train_feature[indices]
        sampled_label = original_train_label[indices]

        cur_train_feature = torch.cat((original_train_feature, sampled_data), dim=0)
        cur_train_label = torch.cat((original_train_label, sampled_label), dim=0)
        permuted_indices = torch.randperm(cur_train_label.shape[0])
        cur_train_feature = cur_train_feature[permuted_indices]
        cur_train_label = cur_train_label[permuted_indices]

        cur_test_feature = client_data_test['features']
        cur_test_label = client_data_test['labels']

        # generate drifting
        dist_bank = list(range(1, rotation_bank * color_bank + 1))
        test_dist = np.random.choice(dist_bank)
        dist_bank.remove(test_dist)
        train_dist = generate_DA_dist(dist_bank,
                                      DA_epoch_locker_num,DA_max_dist,DA_continual_divergence)
        
        lockers = sorted(torch.rand(DA_epoch_locker_num - 1).tolist() + [0.0]) if DA_random_locker \
                else torch.linspace(0, 1, steps=DA_epoch_locker_num + 1)[:-1].tolist()

        print("Train distribution: ", train_dist,
              "\nTest distribution: ", test_dist,
              "\nEpoch lockers: ", lockers,
              "\n") if verbose else None

        # training test
        feature_chunks = torch.chunk(cur_train_feature, DA_epoch_locker_num, dim=0)
        label_chunks = torch.chunk(cur_train_label, DA_epoch_locker_num, dim=0)

        # Initialize cumulative feature and label tensors
        cumulative_features = None
        cumulative_labels = None

        # Loop through feature chunks and label chunks
        for i, (feature_chunk, label_chunk) in enumerate(zip(feature_chunks, label_chunks)):
            # Get angle and color from the pattern bank based on the train_dist
            angle, color = pattern_bank[train_dist[i]]

            # Apply rotation and color transformations
            feature_chunk = rotate_dataset(feature_chunk, [float(angle)] * feature_chunk.shape[0])
            feature_chunk = color_dataset(feature_chunk, [color] * feature_chunk.shape[0])

            # Concatenate cumulatively with previous chunks
            if cumulative_features is None:
                cumulative_features = feature_chunk
                cumulative_labels = label_chunk
            else:
                cumulative_features = torch.cat((cumulative_features, feature_chunk), dim=0)
                cumulative_labels = torch.cat((cumulative_labels, label_chunk), dim=0)

            permuted_indices = torch.randperm(cumulative_labels.shape[0])
            cumulative_features = cumulative_features[permuted_indices]
            cumulative_labels = cumulative_labels[permuted_indices]

            # Append the cumulative data to rearranged_data
            rearranged_data.append({
                'train': True,
                'features': cumulative_features,
                'labels': cumulative_labels,
                'client_number': client_Count,
                'epoch_locker_indicator': lockers[i],
                'epoch_locker_order': i
            })

        # testing set
        angle, color = pattern_bank[test_dist]
        cur_test_feature = rotate_dataset(cur_test_feature, [float(angle)] * cur_test_feature.shape[0])
        cur_test_feature = color_dataset(cur_test_feature, [color] * cur_test_feature.shape[0])

        rearranged_data.append({
            'train': False,
            'features': cur_test_feature,
            'labels': cur_test_label,
            'client_number': client_Count,
            'epoch_locker_indicator': -1.0,
            'epoch_locker_order': -1
        })

        client_Count += 1

    return rearranged_data