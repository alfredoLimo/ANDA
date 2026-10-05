from . import split_fn
from . import split_fn_trDA_teDR
from . import split_fn_trND_teDR
from . import split_fn_trDA_teND
from . import split_fn_trDR_teDR
from . import split_fn_trDR_teND
from . import utils
from .split_fn import *
from .split_fn_trDA_teDR import *
from .split_fn_trND_teDR import *
from .split_fn_trDA_teND import *
from .split_fn_trDR_teDR import *
from .split_fn_trDR_teND import *
from .utils import *

# Arguments used in auto mode, for each non_iid_type and non_iid_level.
AUTO_MODE_PRESETS = {
    "feature_skew": {
        "low": dict(set_rotation=True, rotations=2, scaling_rotation_low=0.0, scaling_rotation_high=0.4, set_color=False, colors=2, scaling_color_low=0.0, scaling_color_high=0.4, random_order=True),
        "medium": dict(set_rotation=True, rotations=2, scaling_rotation_low=0.3, scaling_rotation_high=0.7, set_color=True, colors=2, scaling_color_low=0.3, scaling_color_high=0.7, random_order=True),
        "high": dict(set_rotation=True, rotations=4, scaling_rotation_low=0.6, scaling_rotation_high=1.0, set_color=True, colors=3, scaling_color_low=0.6, scaling_color_high=1.0, random_order=True),
    },
    "label_skew": {
        "low": dict(scaling_label_low=0.0, scaling_label_high=0.5),
        "medium": dict(scaling_label_low=0.5, scaling_label_high=1.0),
        "high": dict(scaling_label_low=1.0, scaling_label_high=3.0),
    },
    "feature_label_skew": {
        "low": dict(scaling_label_low=0.0, scaling_label_high=0.5, set_rotation=True, rotations=2, scaling_rotation_low=0.0, scaling_rotation_high=0.4, set_color=False, colors=2, scaling_color_low=0.0, scaling_color_high=0.4, random_order=True),
        "medium": dict(scaling_label_low=0.5, scaling_label_high=1.0, set_rotation=True, rotations=2, scaling_rotation_low=0.3, scaling_rotation_high=0.7, set_color=True, colors=2, scaling_color_low=0.3, scaling_color_high=0.7, random_order=True),
        "high": dict(scaling_label_low=1.0, scaling_label_high=3.0, set_rotation=True, rotations=4, scaling_rotation_low=0.6, scaling_rotation_high=1.0, set_color=True, colors=3, scaling_color_low=0.6, scaling_color_high=1.0, random_order=True),
    },
    "feature_skew_unbalanced": {
        "low": dict(set_rotation=True, rotations=2, scaling_rotation_low=0.0, scaling_rotation_high=0.4, set_color=False, colors=2, scaling_color_low=0.0, scaling_color_high=0.4, std_dev=0.3, permute=True),
        "medium": dict(set_rotation=True, rotations=2, scaling_rotation_low=0.3, scaling_rotation_high=0.7, set_color=True, colors=2, scaling_color_low=0.3, scaling_color_high=0.7, std_dev=1.0, permute=True),
        "high": dict(set_rotation=True, rotations=4, scaling_rotation_low=0.6, scaling_rotation_high=1.0, set_color=True, colors=3, scaling_color_low=0.6, scaling_color_high=1.0, std_dev=2.0, permute=True),
    },
    "label_skew_unbalanced": {
        "low": dict(scaling_label_low=0.0, scaling_label_high=0.5, std_dev=0.3),
        "medium": dict(scaling_label_low=0.5, scaling_label_high=1.0, std_dev=1.0),
        "high": dict(scaling_label_low=1.0, scaling_label_high=3.0, std_dev=2.0),
    },
    "label_condition_skew": {
        "low": dict(random_mode=True, mixing_label_number=2, scaling_label_low=0.0, scaling_label_high=0.4),
        "medium": dict(random_mode=True, mixing_label_number=3, scaling_label_low=0.3, scaling_label_high=0.7),
        "high": dict(random_mode=True, mixing_label_number=5, scaling_label_low=0.6, scaling_label_high=1.0),
    },
    "label_condition_skew_unbalanced": {
        "low": dict(random_mode=True, mixing_label_number=2, scaling_label_low=0.0, scaling_label_high=0.4, std_dev=0.3, permute=True),
        "medium": dict(random_mode=True, mixing_label_number=3, scaling_label_low=0.3, scaling_label_high=0.7, std_dev=1.0, permute=True),
        "high": dict(random_mode=True, mixing_label_number=5, scaling_label_low=0.6, scaling_label_high=1.0, std_dev=2.0, permute=True),
    },
    "feature_condition_skew": {
        "low": dict(set_rotation=True, rotations=2, set_color=False, colors=2, random_mode=True, rotated_label_number=2, colored_label_number=2),
        "medium": dict(set_rotation=True, rotations=2, set_color=True, colors=2, random_mode=True, rotated_label_number=3, colored_label_number=3),
        "high": dict(set_rotation=True, rotations=4, set_color=True, colors=3, random_mode=True, rotated_label_number=5, colored_label_number=5),
    },
    "feature_condition_skew_unbalanced": {
        "low": dict(set_rotation=True, rotations=2, set_color=False, colors=2, random_mode=True, rotated_label_number=2, colored_label_number=2, std_dev=0.3, permute=True),
        "medium": dict(set_rotation=True, rotations=2, set_color=True, colors=2, random_mode=True, rotated_label_number=3, colored_label_number=3, std_dev=1.0, permute=True),
        "high": dict(set_rotation=True, rotations=4, set_color=True, colors=3, random_mode=True, rotated_label_number=5, colored_label_number=5, std_dev=2.0, permute=True),
    },
    "label_condition_skew_with_label_skew": {
        "low": dict(scaling_label_low=0.0, scaling_label_high=0.5, random_mode=True, mixing_label_number=2, scaling_swapping_low=0.0, scaling_swapping_high=0.4),
        "medium": dict(scaling_label_low=0.5, scaling_label_high=1.0, random_mode=True, mixing_label_number=3, scaling_swapping_low=0.3, scaling_swapping_high=0.7),
        "high": dict(scaling_label_low=1.0, scaling_label_high=3.0, random_mode=True, mixing_label_number=5, scaling_swapping_low=0.6, scaling_swapping_high=1.0),
    },
    "feature_condition_skew_with_label_skew": {
        "low": dict(scaling_label_low=0.0, scaling_label_high=0.5, set_rotation=True, rotations=2, set_color=False, colors=2, random_mode=True, rotated_label_number=2, colored_label_number=2),
        "medium": dict(scaling_label_low=0.5, scaling_label_high=1.0, set_rotation=True, rotations=2, set_color=True, colors=2, random_mode=True, rotated_label_number=3, colored_label_number=3),
        "high": dict(scaling_label_low=1.0, scaling_label_high=3.0, set_rotation=True, rotations=4, set_color=True, colors=3, random_mode=True, rotated_label_number=5, colored_label_number=5),
    },
}

def set_seed(
    RANDOM_SEED: int = 42
):
    '''
    Set the random seed for reproducibility.
    
    Args:
        RANDOM_SEED (int): The random seed to set.
    '''
    utils.set_seed(RANDOM_SEED)

def load_split_datasets(
    dataset_name: str = "MNIST",
    client_number: int = 10,
    non_iid_type: str = "feature_skew",
    mode: str = "auto",
    non_iid_level: str = "medium",
    verbose: bool = True,
    count_labels: bool = True,
    plot_clients: bool = False,
    random_seed: int = 42,
    **kwargs: dict
) -> list:
    """
    Load the split datasets for the federated learning.

    Refer to
    https://github.com/alfredoLimo/ANDA 
    for a quick start.

    Args:
        dataset_name (str): The name of the dataset to load.
        client_number (int): The number of clients to split the dataset.
        non_iid_type (str): The type of non-iid data distribution.
        mode (str): "auto" or "manual".
        non_iid_level (str): The level of non-iid data distribution. (in auto mode)
        verbose (bool): Show verbose information during generating.
        count_labels (bool): Show the label distribution.
        plot_clients (bool): Plot and save images of each client.
        random_seed (int): The random seed for reproducibility.
        **kwargs (dict): The additional arguments for manual mode.
    
    Returns:
        list: The list of length client_number, each element is a dictionary containing the split dataset.
    """
    set_seed(random_seed)
    train_features, train_labels, test_features, test_labels = load_full_datasets(dataset_name)

    if mode == "auto":
        if non_iid_level not in ["low", "medium", "high"]:
            raise ValueError("non_iid_level must be 'low', 'medium', or 'high'")
        if non_iid_type not in AUTO_MODE_PRESETS:
            raise ValueError("Not supported non_iid_type.")
        rearranged_data = globals()[f"split_{non_iid_type}"](
            train_features, train_labels, test_features, test_labels, client_number,
            verbose = verbose, **AUTO_MODE_PRESETS[non_iid_type][non_iid_level]
        )

    elif mode == "manual":
        fn = f"split_{non_iid_type}"
        if fn in globals():
            rearranged_data = globals()[fn](
                train_features, train_labels, test_features, test_labels, \
                client_number, verbose = verbose, \
                **kwargs, 
            )
        else:
            raise ValueError(f"Function {fn} does not exist. Check non_iid_type.")

    else:
        raise ValueError("mode must be 'auto' or 'manual'")

    if count_labels:
        print("Count labels...")
        count_labels_static(rearranged_data)

    if plot_clients:
        print("Plotting and saving images...")
        plot_static(rearranged_data, file_name=f"{dataset_name}_{client_number}_{non_iid_type}")

    return rearranged_data

def load_split_datasets_dynamic(
    dataset_name: str = "MNIST",
    client_number: int = 10,
    non_iid_type: str = "Px",
    drfting_type: str = "trND_teDR",
    verbose: bool = True,
    count_labels: bool = True,
    plot_clients: bool = False,
    random_seed: int = 42,
    **kwargs: dict
) -> list:
    """
    Load the dynamic split datasets for the federated learning.

    Refer to
    https://github.com/alfredoLimo/ANDA 
    for a quick start.

    Args:
        dataset_name (str): The name of the dataset to load.
        client_number (int): The number of clients to split the dataset.
        non_iid_type (str): The type of non-iid data distribution.
        drfting_type (str): The type of drifting data distribution.
        verbose (bool): Whether to show the feature distribution.
        count_labels (bool): Whether to show the label distribution.
        random_seed (int): The random seed for reproducibility.
        **kwargs (dict): The additional arguments for manual mode.
    
    Returns:
        list: The list of length client_number, each element is a dictionary containing the split dataset.
    """
    assert drfting_type in ["trDA_teDR", "trND_teDR", "trDA_teND", "trDR_teDR", "trDR_teND"], "drfting type not supported"
    assert non_iid_type in ["Px","Py","Px_y","Py_x"], "non_iid type not supported"
    
    set_seed(random_seed)
    train_features, train_labels, test_features, test_labels = load_full_datasets(dataset_name)

    fn = f"split_{drfting_type}_{non_iid_type}"
    if fn in globals():
        rearranged_data = globals()[fn](
            train_features, train_labels, test_features, test_labels,
            client_number, verbose = verbose,
            **kwargs,
        )
    else:
        raise ValueError(f"Function {fn} does not exist.")

    if count_labels:
        print("Count labels...")
        if drfting_type == "trND_teDR":
            count_labels_static(rearranged_data)
        else:
            count_labels_dynamic(rearranged_data)         

    if plot_clients:
        print("Plotting and saving images...")
        if drfting_type == "trND_teDR":
            plot_static(rearranged_data, file_name=f"{dataset_name}_{client_number}_{non_iid_type}")
        else:
            plot_dynamic(rearranged_data, file_name=f"{dataset_name}_{client_number}_{drfting_type}_{non_iid_type}")

    return rearranged_data
