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
    split_fn.set_seed(RANDOM_SEED)
    split_fn_trDA_teDR.set_seed(RANDOM_SEED)
    split_fn_trND_teDR.set_seed(RANDOM_SEED)
    split_fn_trDA_teND.set_seed(RANDOM_SEED)
    split_fn_trDR_teDR.set_seed(RANDOM_SEED)
    split_fn_trDR_teND.set_seed(RANDOM_SEED)
    utils.set_seed(RANDOM_SEED)

def load_split_datasets(
    dataset_name: str = "MNIST",
    client_number: int = 10,
    non_iid_type: str = "feature_skew",
    mode: str = "auto",
    non_iid_level: str = "medium",
    show_features: bool = False,
    show_labels: bool = False,
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
        show_features (bool): Whether to show the feature distribution.
        show_labels (bool): Whether to show the label distribution.
        random_seed (int): The random seed for reproducibility.
        **kwargs (dict): The additional arguments for manual mode.
    
    Returns:
        list: The list of length client_number, each element is a dictionary containing the split dataset.
    """
    set_seed(random_seed)
    train_features, train_labels, test_features, test_labels = load_full_datasets(dataset_name)

    if mode == "auto":
        if non_iid_level == "low":
            if non_iid_type == "feature_skew":
                rearranged_data =  split_feature_skew(
                    train_features, train_labels, test_features, test_labels, client_number,
                    set_rotation = True, rotations = 2, scaling_rotation_low = 0.0, scaling_rotation_high = 0.4,
                    set_color = False, colors = 2, scaling_color_low = 0.0, scaling_color_high = 0.4,
                    random_order = True, show_distribution = show_features
                )
            elif non_iid_type == "label_skew":
                rearranged_data = split_label_skew(
                    train_features, train_labels, test_features, test_labels, client_number,
                    scaling_label_low = 0.0, scaling_label_high = 0.5,
                )
            elif non_iid_type == "feature_label_skew":
                rearranged_data =  split_feature_label_skew(
                    train_features, train_labels, test_features, test_labels, client_number,
                    scaling_label_low = 0.0, scaling_label_high = 0.5,
                    set_rotation = True, rotations = 2, scaling_rotation_low = 0.0, scaling_rotation_high = 0.4,
                    set_color = False, colors = 2, scaling_color_low = 0.0, scaling_color_high = 0.4,
                    random_order = True, show_distribution = show_features
                )       
            elif non_iid_type == "feature_skew_unbalanced":
                rearranged_data =  split_feature_skew_unbalanced(
                    train_features, train_labels, test_features, test_labels, client_number,
                    set_rotation = True, rotations = 2, scaling_rotation_low = 0.0, scaling_rotation_high = 0.4,
                    set_color = False, colors = 2, scaling_color_low = 0.0, scaling_color_high = 0.4,
                    std_dev = 0.3, permute = True, show_distribution = show_features
                )
            elif non_iid_type == "label_skew_unbalanced":
                rearranged_data = split_label_skew_unbalanced(
                    train_features, train_labels, test_features, test_labels, client_number,
                    scaling_label_low = 0.0, scaling_label_high = 0.5,
                    std_dev = 0.3, verbose = show_features
                )
            elif non_iid_type == "feature_condition_skew":
                rearranged_data = split_feature_condition_skew(
                    train_features, train_labels, test_features, test_labels, client_number,
                    random_mode = True, mixing_label_number = 2,
                    scaling_label_low = 0.0, scaling_label_high = 0.4, verbose = show_features
                )
            elif non_iid_type == "feature_condition_skew_unbalanced":
                rearranged_data = split_feature_condition_skew_unbalanced(
                    train_features, train_labels, test_features, test_labels, client_number,
                    random_mode = True, mixing_label_number = 2,
                    scaling_label_low = 0.0, scaling_label_high = 0.4,
                    std_dev = 0.3, permute = True, verbose = show_features
                )
            elif non_iid_type == "label_condition_skew":
                rearranged_data = split_label_condition_skew(
                    train_features, train_labels, test_features, test_labels, client_number,
                    set_rotation = True, rotations = 2, set_color = False, colors = 2,
                    random_mode = True, rotated_label_number = 2, colored_label_number = 2,
                    verbose = show_features
                )
            elif non_iid_type == "label_condition_skew_unbalanced":
                rearranged_data = split_label_condition_skew_unbalanced(
                    train_features, train_labels, test_features, test_labels, client_number,
                    set_rotation = True, rotations = 2, set_color = False, colors = 2,
                    random_mode = True, rotated_label_number = 2, colored_label_number = 2,
                    std_dev = 0.3, permute = True, verbose = show_features
                )
            elif non_iid_type == "feature_condition_skew_with_label_skew":
                rearranged_data = split_feature_condition_skew_with_label_skew(
                    train_features, train_labels, test_features, test_labels, client_number,
                    scaling_label_low = 0.0, scaling_label_high = 0.5,
                    random_mode = True, mixing_label_number = 2,
                    scaling_swapping_low = 0.0, scaling_swapping_high = 0.4,
                    verbose = show_features
                )
            elif non_iid_type == "label_condition_skew_with_label_skew":
                rearranged_data = split_label_condition_skew_with_label_skew(
                    train_features, train_labels, test_features, test_labels, client_number,
                    scaling_label_low = 0.0, scaling_label_high = 0.5,
                    set_rotation = True, rotations = 2, set_color = False, colors = 2,
                    random_mode = True, rotated_label_number = 2, colored_label_number = 2,
                    verbose = show_features
                )
            else:
                raise ValueError("Not supoorted non_iid_type.")

        elif non_iid_level == "medium":
            if non_iid_type == "feature_skew":
                rearranged_data =  split_feature_skew(
                    train_features, train_labels, test_features, test_labels, client_number,
                    set_rotation = True, rotations = 2, scaling_rotation_low = 0.3, scaling_rotation_high = 0.7,
                    set_color = True, colors = 2, scaling_color_low = 0.3, scaling_color_high = 0.7,
                    random_order = True, show_distribution = show_features
                )
            elif non_iid_type == "label_skew":
                rearranged_data = split_label_skew(
                    train_features, train_labels, test_features, test_labels, client_number,
                    scaling_label_low = 0.5, scaling_label_high = 1.0,
                )
            elif non_iid_type == "feature_label_skew":
                rearranged_data =  split_feature_label_skew(
                    train_features, train_labels, test_features, test_labels, client_number,
                    scaling_label_low = 0.5, scaling_label_high = 1.0,
                    set_rotation = True, rotations = 2, scaling_rotation_low = 0.3, scaling_rotation_high = 0.7,
                    set_color = True, colors = 2, scaling_color_low = 0.3, scaling_color_high = 0.7,
                    random_order = True, show_distribution = show_features
                )
            elif non_iid_type == "feature_skew_unbalanced":
                rearranged_data =  split_feature_skew_unbalanced(
                    train_features, train_labels, test_features, test_labels, client_number,
                    set_rotation = True, rotations = 2, scaling_rotation_low = 0.3, scaling_rotation_high = 0.7,
                    set_color = True, colors = 2, scaling_color_low = 0.3, scaling_color_high = 0.7,
                    std_dev = 1.0, permute = True, show_distribution = show_features
                )
            elif non_iid_type == "label_skew_unbalanced":
                rearranged_data = split_label_skew_unbalanced(
                    train_features, train_labels, test_features, test_labels, client_number,
                    scaling_label_low = 0.5, scaling_label_high = 1.0,
                    std_dev = 1.0, verbose = show_features
                )
            elif non_iid_type == "feature_condition_skew":
                rearranged_data = split_feature_condition_skew(
                    train_features, train_labels, test_features, test_labels, client_number,
                    random_mode = True, mixing_label_number = 3,
                    scaling_label_low = 0.3, scaling_label_high = 0.7, verbose = show_features
                )
            elif non_iid_type == "feature_condition_skew_unbalanced":
                rearranged_data = split_feature_condition_skew_unbalanced(
                    train_features, train_labels, test_features, test_labels, client_number,
                    random_mode = True, mixing_label_number = 3,
                    scaling_label_low = 0.3, scaling_label_high = 0.7,
                    std_dev = 1.0, permute = True, verbose = show_features
                )
            elif non_iid_type == "label_condition_skew":
                rearranged_data = split_label_condition_skew(
                    train_features, train_labels, test_features, test_labels, client_number,
                    set_rotation = True, rotations = 2, set_color = True, colors = 2,
                    random_mode = True, rotated_label_number = 3, colored_label_number = 3,
                    verbose = show_features
                )
            elif non_iid_type == "label_condition_skew_unbalanced":
                rearranged_data = split_label_condition_skew_unbalanced(
                    train_features, train_labels, test_features, test_labels, client_number,
                    set_rotation = True, rotations = 2, set_color = True, colors = 2,
                    random_mode = True, rotated_label_number = 3, colored_label_number = 3,
                    std_dev = 1.0, permute = True, verbose = show_features
                )
            elif non_iid_type == "feature_condition_skew_with_label_skew":
                rearranged_data = split_feature_condition_skew_with_label_skew(
                    train_features, train_labels, test_features, test_labels, client_number,
                    scaling_label_low = 0.5, scaling_label_high = 1.0,
                    random_mode = True, mixing_label_number = 3,
                    scaling_swapping_low = 0.3, scaling_swapping_high = 0.7,
                    verbose = show_features
                )
            elif non_iid_type == "label_condition_skew_with_label_skew":
                rearranged_data = split_label_condition_skew_with_label_skew(
                    train_features, train_labels, test_features, test_labels, client_number,
                    scaling_label_low = 0.5, scaling_label_high = 1.0,
                    set_rotation = True, rotations = 2, set_color = True, colors = 2,
                    random_mode = True, rotated_label_number = 3, colored_label_number = 3,
                    verbose = show_features
                )
            else:
                raise ValueError("Not supoorted non_iid_type.")
        elif non_iid_level == "high":
            if non_iid_type == "feature_skew":
                rearranged_data =  split_feature_skew(
                    train_features, train_labels, test_features, test_labels, client_number,
                    set_rotation = True, rotations = 4, scaling_rotation_low = 0.6, scaling_rotation_high = 1.0,
                    set_color = True, colors = 3, scaling_color_low = 0.6, scaling_color_high = 1.0,
                    random_order = True, show_distribution = show_features
                )
            elif non_iid_type == "label_skew":
                rearranged_data = split_label_skew(
                    train_features, train_labels, test_features, test_labels, client_number,
                    scaling_label_low = 1.0, scaling_label_high = 3.0,
                )
            elif non_iid_type == "feature_label_skew":
                rearranged_data =  split_feature_label_skew(
                    train_features, train_labels, test_features, test_labels, client_number,
                    scaling_label_low = 1.0, scaling_label_high = 3.0,
                    set_rotation = True, rotations = 4, scaling_rotation_low = 0.6, scaling_rotation_high = 1.0,
                    set_color = True, colors = 3, scaling_color_low = 0.6, scaling_color_high = 1.0,
                    random_order = True, show_distribution = show_features
                )
            elif non_iid_type == "feature_skew_unbalanced":
                rearranged_data =  split_feature_skew_unbalanced(
                    train_features, train_labels, test_features, test_labels, client_number,
                    set_rotation = True, rotations = 4, scaling_rotation_low = 0.6, scaling_rotation_high = 1.0,
                    set_color = True, colors = 3, scaling_color_low = 0.6, scaling_color_high = 1.0,
                    std_dev = 2.0, permute = True, show_distribution = show_features
                )
            elif non_iid_type == "label_skew_unbalanced":
                rearranged_data = split_label_skew_unbalanced(
                    train_features, train_labels, test_features, test_labels, client_number,
                    scaling_label_low = 1.0, scaling_label_high = 3.0,
                    std_dev = 2.0, verbose = show_features
                )
            elif non_iid_type == "feature_condition_skew":
                rearranged_data = split_feature_condition_skew(
                    train_features, train_labels, test_features, test_labels, client_number,
                    random_mode = True, mixing_label_number = 5,
                    scaling_label_low = 0.6, scaling_label_high = 1.0, verbose = show_features
                )
            elif non_iid_type == "feature_condition_skew_unbalanced":
                rearranged_data = split_feature_condition_skew_unbalanced(
                    train_features, train_labels, test_features, test_labels, client_number,
                    random_mode = True, mixing_label_number = 5,
                    scaling_label_low = 0.6, scaling_label_high = 1.0,
                    std_dev = 2.0, permute = True, verbose = show_features
                )
            elif non_iid_type == "label_condition_skew":
                rearranged_data = split_label_condition_skew(
                    train_features, train_labels, test_features, test_labels, client_number,
                    set_rotation = True, rotations = 4, set_color = True, colors = 3,
                    random_mode = True, rotated_label_number = 5, colored_label_number = 5,
                    verbose = show_features
                )
            elif non_iid_type == "label_condition_skew_unbalanced":
                rearranged_data = split_label_condition_skew_unbalanced(
                    train_features, train_labels, test_features, test_labels, client_number,
                    set_rotation = True, rotations = 4, set_color = True, colors = 3,
                    random_mode = True, rotated_label_number = 5, colored_label_number = 5,
                    std_dev = 2.0, permute = True, verbose = show_features
                )
            elif non_iid_type == "feature_condition_skew_with_label_skew":
                rearranged_data = split_feature_condition_skew_with_label_skew(
                    train_features, train_labels, test_features, test_labels, client_number,
                    scaling_label_low = 1.0, scaling_label_high = 3.0,
                    random_mode = True, mixing_label_number = 5,
                    scaling_swapping_low = 0.6, scaling_swapping_high = 1.0,
                    verbose = show_features
                )
            elif non_iid_type == "label_condition_skew_with_label_skew":
                rearranged_data = split_label_condition_skew_with_label_skew(
                    train_features, train_labels, test_features, test_labels, client_number,
                    scaling_label_low = 1.0, scaling_label_high = 3.0,
                    set_rotation = True, rotations = 4, set_color = True, colors = 3,
                    random_mode = True, rotated_label_number = 5, colored_label_number = 5,
                    verbose = show_features
                )
            else:
                raise ValueError("Not supoorted non_iid_type.")
        else:
            raise ValueError("non_iid_level must be 'low', 'medium', or 'high'")

    elif mode == "manual":
        fn = f"split_{non_iid_type}"
        if fn in globals():
            rearranged_data = globals()[fn](
                train_features, train_labels, test_features, test_labels, client_number,
                **kwargs,
            )
        else:
            print(f"Function {fn} does not exist")

    else:
        raise ValueError("mode must be 'auto' or 'manual'")   

    if show_labels:
        print("Count labels...")
        print("Plotting and saving first 100 images each for datasets of first four clients...")
        print("This is hardcoded, please refer to the last part of load_split_datasets if any error/ modification is needed...")
        draw_split_statistic(rearranged_data, plot_indices=[0,1,2,3],save=True,
                             file_name=f"{dataset_name}_{client_number}_{non_iid_type}")

    return rearranged_data

def load_split_datasets_dynamic(
    dataset_name: str = "MNIST",
    client_number: int = 10,
    non_iid_type: str = "Px",
    drfting_type: str = "trND_teDR",
    show_features: bool = True,
    show_labels: bool = True,
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
        show_features (bool): Whether to show the feature distribution.
        show_labels (bool): Whether to show the label distribution.
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
            client_number, verbose = show_features,
            **kwargs,
        )
    else:
        print(f"Function {fn} does not exist")

    if show_labels:
        if drfting_type == "trND_teDR":
            print("Count labels and plot...")   
            print("Plotting and saving first 100 images each for datasets of first four clients...")
            print("This is hardcoded, please refer to the last part of load_split_datasets if any error/ modification is needed...")
            draw_split_statistic(rearranged_data, plot_indices=[0,1,2,3],save=True,
                                file_name=f"{dataset_name}_{client_number}_{non_iid_type}")

        else:
            print("Count labels and plot...")
            print("This is hardcoded, please refer to the last part of load_split_datasets if any error/ modification is needed...")
            draw_split_statistic_dy(rearranged_data,count=True,save=True,
                                    file_name=f"{dataset_name}_{client_number}_{drfting_type}_{non_iid_type}",
                                    locker_indices=[0,1,2,-1])

    return rearranged_data
