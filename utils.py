import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import truncnorm
import os
import random

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

def set_seed(
    RANDOM_SEED: int = 42
):
    '''
    Set the random seed for reproducibility.
    
    Args:
        RANDOM_SEED (int): The random seed to set.
    '''
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

def merge_data(
    data: list
) -> list:
    '''
    Merges the data from multiple clients into a single dataset.

    Args:
        data (list): A list of dictionaries where each dictionary contains the features and labels for each client (output of previous functions).
    
    Returns:
        list: A list of four torch.Tensors containing the training features, training labels, testing features, and testing labels.
    
    '''
    # Concatenate all the data (outputs of split functions are numpy arrays)
    train_features = torch.cat([torch.as_tensor(client_data['train_features']) for client_data in data], dim=0)
    train_labels = torch.cat([torch.as_tensor(client_data['train_labels']) for client_data in data], dim=0)
    test_features = torch.cat([torch.as_tensor(client_data['test_features']) for client_data in data], dim=0)
    test_labels = torch.cat([torch.as_tensor(client_data['test_labels']) for client_data in data], dim=0)

    return [train_features, train_labels, test_features, test_labels]

def load_full_datasets(
    dataset_name: str = "MNIST",
) -> list:
    '''
    Load datasets into four separate parts: train labels, train images, test labels, test images.

    Args:
        dataset_name (str): Name of the dataset to load. Options are "MNIST", "FMNIST", "EMNIST", "CIFAR10", "CIFAR100".

    TODO: EMNIST IS NOT WELL.

    Returns:
        list: [4] of torch.Tensor. [train_images, train_labels, test_images, test_labels]
    '''
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    if dataset_name == "MNIST":
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == "FMNIST":
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == "EMNIST": # not auto-downloaded successfully
        train_dataset = datasets.EMNIST(root='./data', split='letters', train=True, download=True, 
                                            transform = transforms.Compose([ 
                                            lambda img: transforms.functional.rotate(img, -90), 
                                            lambda img: transforms.functional.hflip(img), 
                                            transforms.ToTensor()
                                            ])
                                        )               
        test_dataset = datasets.EMNIST(root='./data', split='letters', train=False, download=True,
                                            transform = transforms.Compose([ 
                                            lambda img: transforms.functional.rotate(img, -90), 
                                            lambda img: transforms.functional.hflip(img), 
                                            transforms.ToTensor()
                                            ])
                                        )         
    elif dataset_name == "CIFAR10":
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == "CIFAR100":
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    # Extracting train and test images and labels.
    # Read the raw uint8 arrays directly instead of iterating through PIL images one by one;
    # the result is identical to ToTensor() (uint8 / 255) but much faster.
    train_images = _raw_images_to_tensor(train_dataset.data, dataset_name)
    test_images = _raw_images_to_tensor(test_dataset.data, dataset_name)

    if dataset_name in ["CIFAR10", "CIFAR100"]:
        train_labels = torch.tensor(train_dataset.targets).clone().detach()
        test_labels = torch.tensor(test_dataset.targets).clone().detach()
    else:
        train_labels = train_dataset.targets.clone().detach()
        test_labels = test_dataset.targets.clone().detach()

    return [train_images, train_labels, test_images, test_labels]

def _raw_images_to_tensor(
    data,
    dataset_name: str
) -> torch.Tensor:
    '''
    Converts the raw uint8 image array stored in a torchvision dataset into a float tensor,
    exactly as ToTensor() would do image by image.
    '''
    images = torch.as_tensor(data)
    if dataset_name in ["CIFAR10", "CIFAR100"]:
        images = images.permute(0, 3, 1, 2) # (N, H, W, 3) -> (N, 3, H, W)
    elif dataset_name == "EMNIST":
        images = images.transpose(1, 2) # same as rotating by -90 degrees and flipping horizontally
    return images.contiguous().float().div(255)

def _pil_rotate(
    img_tensor: torch.Tensor,
    degree: float
) -> torch.Tensor:
    '''
    Rotates one image through PIL (used for angles that are not a multiple of 90 degrees).
    '''
    img = transforms.ToPILImage()(img_tensor)
    return transforms.ToTensor()(img.rotate(degree)).squeeze(0)

def rotate_dataset(
    dataset: torch.Tensor,
    degrees: list
) -> torch.Tensor:
    '''
    Rotates all images in the dataset by a specified degree.

    Args:
        dataset (torch.Tensor): Input dataset, a tensor of shape (N, ) where N is the number of images.
        degrees (list) : List of degrees to rotate each image.

    Returns:
        torch.Tensor: The rotated dataset, a tensor of the same shape (N, ) as the input.
    '''

    if len(dataset) != len(degrees):
        raise ValueError("The length of degrees list must be equal to the number of images in the dataset.")

    if len(dataset) == 0:
        return dataset.clone()

    # Images used to go through PIL (as 8-bit images) one by one. Reproduce the same 8-bit
    # rounding here, then rotate whole groups of images at once.
    quantized = dataset.mul(255).byte().float().div(255)
    rotated_dataset = torch.empty_like(quantized)

    degrees = np.asarray(degrees, dtype=float) % 360.0
    square = dataset.shape[-1] == dataset.shape[-2]

    for degree in np.unique(degrees):
        idx = torch.from_numpy(np.nonzero(degrees == degree)[0])
        if degree % 180 == 0 or (degree % 90 == 0 and square):
            # PIL rotates counter-clockwise by transposing pixels for these angles, same as rot90
            rotated_dataset[idx] = torch.rot90(quantized[idx], int(degree // 90), dims=(-2, -1))
        else:
            rotated_dataset[idx] = torch.stack([_pil_rotate(dataset[i], degree) for i in idx.tolist()])

    return rotated_dataset

def color_dataset(
    dataset: torch.Tensor,
    colors: list
) -> torch.Tensor:
    '''
    Colors all images in the dataset by a specified color.

    Args:
        dataset (torch.Tensor): Input dataset, a tensor of shape (N, H, W) or (N, 3, H, W)
                                where N is the number of images.
        colors (list) : List of 'red', 'green', 'blue', 'gray'.

    Warning:
        MNIST, FMNIST, EMNIST are 1-channel. CIFAR10, CIFAR100 are 3-channel.

    Returns:
        torch.Tensor: The colored dataset, a tensor of the shape (N, 3, H, W) with 3 channels.
    '''

    if len(dataset) != len(colors):
        raise ValueError("The length of colors list must be equal to the number of images in the dataset.")

    if dataset.dim() == 3:
        # Handle 1-channel dataset
        colored_dataset = dataset.unsqueeze(1).repeat(1, 3, 1, 1) # Shape becomes (N, 3, H, W)
    elif dataset.dim() == 4 and dataset.size(1) == 3:
        colored_dataset = dataset.clone()
    else:
        raise ValueError("This function only supports 1-channel (N, H, W) or 3-channel (N, 3, H, W) datasets.")

    colors = np.asarray(colors, dtype=str)
    if not np.isin(colors, ['red', 'green', 'blue', 'gray']).all():
        raise ValueError("Color must be 'red', 'green', or 'blue'")

    # Map the grayscale values to the specified color by setting that channel to 1
    for channel, color in enumerate(['red', 'green', 'blue']):
        idx = torch.from_numpy(np.nonzero(colors == color)[0])
        colored_dataset[idx, channel, :, :] = 1

    return colored_dataset

def split_basic(
    features: torch.Tensor,
    labels: torch.Tensor,
    client_number: int = 10,
    permute: bool = True
) -> list:
    """
    Splits a dataset into a specified number of clusters (clients).
    
    Args:
        features (torch.Tensor): The dataset features.
        labels (torch.Tensor): The dataset labels.
        client_number (int): The number of clients to split the data into.
        permute (bool): Whether to shuffle the data before splitting.
        
    Returns:
        list: A list of dictionaries where each dictionary contains the features and labels for each client.
    """

    # Ensure the features and labels have the same number of samples
    assert len(features) == len(labels), "The number of samples in features and labels must be the same."

    # Randomly shuffle the dataset while maintaining correspondence between features and labels
    if permute:
        indices = torch.randperm(len(features))
        features, labels = features[indices], labels[indices]
    
    # Calculate the number of samples per client
    samples_per_client = len(features) // client_number
    
    # List to hold the data for each client
    client_data = []
    
    for i in range(client_number):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        
        # Handle the last client which may take the remaining samples
        if i == client_number - 1:
            end_idx = len(features)
        
        client_features = features[start_idx:end_idx]
        client_labels = labels[start_idx:end_idx]
        
        client_data.append({
            'features': client_features,
            'labels': client_labels
        })
    
    return client_data

def split_unbalanced(
    features: torch.Tensor,
    labels: torch.Tensor,
    client_number: int = 10,
    std_dev: float = 0.1,
    permute: bool = True
) -> list:
    """
    Splits a dataset into a specified number of clusters unbalanced (clients).
    
    Args:
        features (torch.Tensor): The dataset features.
        labels (torch.Tensor): The dataset labels.
        client_number (int): The number of clients to split the data into.
        std_dev (float): standard deviation of the normal distribution for the number of samples per client.
        permute (bool): Whether to shuffle the data before splitting.
        
    Returns:
        list: A list of dictionaries where each dictionary contains the features and labels for each client.
    """

    # Ensure the features and labels have the same number of samples
    assert len(features) == len(labels), "The number of samples in features and labels must be the same."
    assert std_dev > 0, "Standard deviation must be larger than 0."

    # Generate random percentage from a truncated normal distribution
    percentage = truncnorm.rvs(-0.5/std_dev, 0.5/std_dev, loc=0.5, scale=std_dev, size=client_number)
    normalized_percentage = percentage / np.sum(percentage)

    # Randomly shuffle the dataset while maintaining correspondence between features and labels
    if permute:
        indices = torch.randperm(len(features))
        features = features[indices]
        labels = labels[indices]

    # Calculate the number of samples per client based on the normalized samples
    total_samples = len(features)
    samples_per_client = (normalized_percentage * total_samples).astype(int)

    # Adjust to ensure the sum of samples_per_client equals the total_samples
    difference = total_samples - samples_per_client.sum()
    for i in range(abs(difference)):
        samples_per_client[i % client_number] += np.sign(difference)
    
    # List to hold the data for each client
    client_data = []
    start_idx = 0
    
    for i in range(client_number):
        end_idx = start_idx + samples_per_client[i]
        
        client_features = features[start_idx:end_idx]
        client_labels = labels[start_idx:end_idx]
        
        client_data.append({
            'features': client_features,
            'labels': client_labels
        })
        
        start_idx = end_idx
    
    return client_data

def assigning_rotation_features(
    datapoint_number: int,
    rotations: int = 4,
    scaling: float = 0.1,
    random_order: bool = True
) -> list:
    '''
    Assigns a rotation to each datapoint based on a softmax distribution.

    Args:
        datapoint_number (int): The number of datapoints to assign rotations to.
        rotations (int): The number of possible rotations. Recommended to be [2,4].
        scaling (float): The scaling factor for the softmax distribution. 0: Uniform distribution.
        random_order (bool): Whether to shuffle the order of the rotations.
    
    Returns:
        list: A list of rotations assigned to the datapoints.
    '''
    assert 0 <= scaling <= 1, "k must be between 0 and 1."
    assert rotations > 1, "Must have at least 2 rotations."

    # Scale the values based on k
    values = np.arange(rotations, 0, -1)  # From N to 1
    scaled_values = values * scaling
    
    # Apply softmax to get the probabilities
    exp_values = np.exp(scaled_values)
    probabilities = exp_values / np.sum(exp_values)

    angles = [i * 360 / rotations for i in range(rotations)]
    if random_order:
        np.random.shuffle(angles)

    angles_assigned = np.random.choice(angles, size=datapoint_number, p=probabilities)

    return angles_assigned

def assigning_color_features(
    datapoint_number: int,
    colors: int = 3,
    scaling: float = 0.1,
    random_order: bool = True
) -> list:
    '''
    Assigns colors to the datapoints based on the softmax probabilities.

    Args:
        datapoint_number (int): Number of datapoints to assign colors to.
        colors (int): Number of colors to assign. Must be 2 or 3.
        scaling (float): Scaling factor for the softmax probabilities. 0: Uniform distribution.
        random_order (bool): Whether to shuffle the order of the colors.

    Returns:
        list: A list of colors assigned to the datapoints.
    '''

    assert 0 <= scaling <= 1, "k must be between 0 and 1."
    assert colors == 2 or colors == 3, "Color must be 2 or 3."
    
    # Scale the values based on k
    values = np.arange(colors, 0, -1)  # From N to 1
    scaled_values = values * scaling
    
    # Apply softmax to get the probabilities
    exp_values = np.exp(scaled_values)
    probabilities = exp_values / np.sum(exp_values)

    if colors == 2:
        letters = ['red', 'blue']
    else:
        letters = ['red', 'blue', 'green']

    if random_order:
        np.random.shuffle(letters)

    colors_assigned = np.random.choice(letters, size=datapoint_number, p=probabilities)

    # unique, counts = np.unique(colors_assigned, return_counts=True)
    # for letter, count in zip(unique, counts):
    #     print(f'{letter}: {count}')

    return colors_assigned

def assigning_gray_color_features(
    datapoint_number: int,
    colors: int = 3,
    scaling: float = 0.1,
    random_order: bool = True
) -> list:
    '''
    Assigns colors to the datapoints based on the softmax probabilities.

    Args:
        datapoint_number (int): Number of datapoints to assign colors to.
        colors (int): Number of colors to assign. Must be 2 or 3.
        scaling (float): Scaling factor for the softmax probabilities. 0: Uniform distribution.
        random_order (bool): Whether to shuffle the order of the colors.

    Returns:
        list: A list of colors assigned to the datapoints.
    '''

    return assigning_color_features(datapoint_number, colors, scaling, random_order)

def calculate_probabilities(
    labels,
    scaling
):
    # Count the occurrences of each label
    label_counts = torch.bincount(labels, minlength=10).float()
    scaled_counts = label_counts ** scaling
    
    # Apply softmax to get probabilities
    probabilities = F.softmax(scaled_counts, dim=0)
    
    return probabilities

def _select_by_probability(
    point_probabilities: np.ndarray,
    labels: np.ndarray,
    num_points: int,
    draw,
    get_state,
    set_state
) -> np.ndarray:
    '''
    Walks over the datapoints and keeps each one with its probability, one pass after another,
    until num_points different datapoints are kept. Returns the kept indices in the order they were kept.

    Args:
        point_probabilities (np.ndarray): The probability of keeping each datapoint.
        labels (np.ndarray): The label of each datapoint.
        num_points (int): The number of datapoints to keep (at most all of them).
        draw, get_state, set_state: Draw n uniform numbers / save / restore the random generator.

    The random numbers of a pass are drawn in one call. The generator is then rewound so that exactly
    as many numbers are consumed as a point-by-point loop would consume.
    A datapoint is never kept twice. When no datapoint left can be drawn any more (e.g. the preferred
    labels are used up), the most likely datapoints left are taken.
    '''
    num_points = min(int(num_points), len(point_probabilities))
    taken = np.zeros(len(point_probabilities), dtype=bool)
    selected_chunks = []
    selected_number = 0
    while selected_number < num_points:
        rng_state = get_state()
        hits = np.nonzero((draw(len(point_probabilities)) < point_probabilities) & ~taken)[0]
        missing = num_points - selected_number
        if len(hits) >= missing:
            hits = hits[:missing]
            set_state(rng_state)
            draw(int(hits[-1]) + 1)
        elif len(hits) == 0:
            left = np.nonzero(~taken)[0]
            label_counts = np.bincount(labels[left])[labels[left]]
            order = np.lexsort((left, -label_counts, -point_probabilities[left]))
            hits = left[order[:missing]]
        taken[hits] = True
        selected_chunks.append(hits)
        selected_number += len(hits)

    return np.concatenate(selected_chunks) if selected_chunks else np.empty(0, dtype=np.int64)

def create_sub_dataset(
        features, 
        labels, 
        probabilities, 
        num_points
):
    # Keep each datapoint with the probability of its label until num_points are selected
    selected_indices = torch.from_numpy(_select_by_probability(
        probabilities[labels].numpy(), labels.numpy(), num_points,
        lambda n: torch.rand(n).numpy(), torch.get_rng_state, torch.set_rng_state))

    sub_features = features[selected_indices]
    sub_labels = labels[selected_indices]
    remaining_indices = torch.ones(len(labels), dtype=torch.bool)
    remaining_indices[selected_indices] = 0
    remaining_features = features[remaining_indices]
    remaining_labels = labels[remaining_indices]

    return sub_features, sub_labels, remaining_features, remaining_labels

def _sample_by_label_probability(
    labels: torch.Tensor,
    label_order: list,
    probabilities: np.ndarray,
    num_points: int
) -> torch.Tensor:
    '''
    Keeps datapoints with probability probabilities[label_order.index(label)] until num_points are kept,
    using the numpy random generator. Returns the kept indices in the order they were kept.
    '''
    label_probabilities = np.zeros(max(label_order) + 1)
    label_probabilities[label_order] = probabilities
    labels = labels.numpy()

    return torch.from_numpy(_select_by_probability(
        label_probabilities[labels], labels, num_points,
        np.random.rand, np.random.get_state, np.random.set_state))

def generate_DA_dist(
    dist_bank: list,
    DA_epoch_locker_num: int,
    DA_max_dist: int,
    DA_continual_divergence: bool
) -> list:
    lst = []
    while len(lst) < DA_epoch_locker_num:
        # reaching DA_max_dist
        if len(set(lst)) == DA_max_dist:
            lst.append(lst[-1]) if DA_continual_divergence else lst.append(np.random.choice(lst))
        else:
            # update dist_bank 
            if len(lst) > 0 and DA_continual_divergence:
                dist_bank = [x for x in dist_bank if x not in lst or x == lst[-1]]
            lst.append(np.random.choice(dist_bank))
    
    return lst

def _label_counts(
    labels
) -> torch.Tensor:
    '''
    Counts the occurrences of each class (at least 10 classes are shown).
    '''
    labels = np.asarray(labels).astype(np.int64).ravel()
    return torch.from_numpy(np.bincount(labels, minlength=10))

# ---------------------------------------------------------------------------
# Building blocks shared by the drifting (dynamic) split functions
# ---------------------------------------------------------------------------

def _rotation_angles(
    rotation_bank: int
) -> list:
    '''
    Returns the rotation angles of a rotation bank. 1 as no rotation.
    '''
    return [i * 360 / rotation_bank for i in range(rotation_bank)] if rotation_bank > 1 else [0.0]

def _color_names(
    color_bank: int
) -> list:
    '''
    Returns the colors of a color bank. 1 as no color.
    '''
    if color_bank == 1:
        return ['gray']
    elif color_bank == 2:
        return ['red', 'blue']
    elif color_bank == 3:
        return ['red', 'blue', 'green']
    raise ValueError("The number of color patterns must be 1, 2, or 3.")

def _extend_dataset(
    features: torch.Tensor,
    labels: torch.Tensor,
    dataset_scaling: float
) -> tuple:
    '''
    Extends a dataset to dataset_scaling times its size with randomly repeated datapoints, then shuffles it.
    '''
    indices = torch.randint(0, labels.shape[0], (int(labels.shape[0] * (dataset_scaling - 1)),))
    features = torch.cat((features, features[indices]), dim=0)
    labels = torch.cat((labels, labels[indices]), dim=0)
    permuted_indices = torch.randperm(labels.shape[0])
    return features[permuted_indices], labels[permuted_indices]

def _epoch_lockers(
    epoch_locker_num: int,
    random_locker: bool
) -> list:
    '''
    Returns the epoch locker indicators (when each subset starts during training), starting with 0.0.
    '''
    if random_locker:
        return sorted(torch.rand(epoch_locker_num - 1).tolist() + [0.0])
    return torch.linspace(0, 1, steps=epoch_locker_num + 1)[:-1].tolist()

def _swap_labels(
    labels: torch.Tensor,
    label_remapping: dict
) -> torch.Tensor:
    '''
    Returns a copy of labels where each original label is replaced by label_remapping[original label].
    '''
    remapped_labels = torch.clone(labels)
    for original_label, new_label in label_remapping.items():
        remapped_labels[labels == original_label] = new_label
    return remapped_labels

def _targeted_px_pattern(
    labels: torch.Tensor,
    targeted_classes: list,
    px_pattern: list
) -> tuple:
    '''
    Returns the angle and color of each datapoint: px_pattern for the targeted classes, no change otherwise.
    '''
    angle, color = px_pattern
    targeted = [label in targeted_classes for label in labels.tolist()]
    angles = [float(angle) if t else 0.0 for t in targeted]
    colors = [color if t else 'gray' for t in targeted]
    return angles, colors

def count_labels_static(
    data_list: list
) -> None:
    '''
    Print label counts for each client in the data list.
    
    Args:
        data_list (list): A list of dictionaries where each dictionary contains the features and labels for each client.
                          * Output of split_fns
    '''
    # Print label counts for each dictionary
    for i, data in enumerate(data_list):
        print(f"Client {i}:")
        print("Training label counts:", _label_counts(data['train_labels']))
        print("Test label counts:", _label_counts(data['test_labels']))
        print("\n")
    
    return

def count_labels_dynamic(
    data_list: list
) -> None:
    '''
    Print label counts for each client in the data list. (for drifting and dynamic datasets)
    
    Args:
        data_list (list): A list of dictionaries where each dictionary contains the features and labels for each client.
                          * Output of split_fns
    '''
    # Print label counts for each dictionary
    for data in data_list:
        print(
            f"Client {data['client_number']} | {'Train' if data['train'] else 'Test'} | "
            f"Epoch Locker Order: {data['epoch_locker_order']} | "
            f"Label Counts: {_label_counts(data['labels']).tolist()}"
        )
    
    return

def _plot_images(
    features,
    labels,
    title: str,
    save_path: str
) -> None:
    '''
    Plots the first 100 images in a 10x10 grid, saves the figure and shows it.
    '''
    features = np.asarray(features)
    labels = np.asarray(labels)

    num_images = min(100, features.shape[0])
    fig, axes = plt.subplots(10, 10, figsize=(15, 15))
    fig.suptitle(title.format(num_images=num_images), fontsize=16)

    for i, ax in enumerate(axes.flat):
        ax.axis('off')
        if i >= num_images:
            continue

        image = features[i]
        if image.ndim == 3 and image.shape[0] == 3:
            # For colored or CIFAR images (3, H, W) -> (H, W, 3)
            image = image.transpose(1, 2, 0)
        else:
            # For MNIST (1, H, W) or (H, W) -> (H, W)
            image = image.squeeze()

        ax.imshow(image, cmap='gray' if image.ndim == 2 else None)
        ax.set_title(labels[i].item())

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    # Save before showing: showing the figure may clear it in notebooks
    fig.savefig(save_path)
    print(f"Saved images to {save_path}")
    plt.show()
    plt.close(fig)

def plot_static(
    data_list: list,
    plot_indices: list = [0,1,2,3],
    save_dir: str = './anda_plot',
    file_name: str = None
) -> None:
    '''
    Plot and save the first 100 training and testing images of some clients.
    
    Args:
        data_list (list): A list of dictionaries where each dictionary contains the features and labels for each client.
                          * Output of split_fns
        plot_indices (list): A list of indices to plot the first 100 images for each client.
        save_dir (str): The directory to save the images.
        file_name (str): The prefix of the saved image files.
    '''

    os.makedirs(save_dir, exist_ok=True)

    for idx in plot_indices:
        if idx < len(data_list):
            data = data_list[idx]

            _plot_images(
                data['train_features'], data['train_labels'],
                f'Dictionary {idx} - First {{num_images}} Training Images',
                os.path.join(save_dir, f'{file_name}_client_{idx}_train_data_plot.png')
            )
            _plot_images(
                data['test_features'], data['test_labels'],
                f'Dictionary {idx} - First {{num_images}} Testing Images',
                os.path.join(save_dir, f'{file_name}_client_{idx}_test_data_plot.png')
            )

def plot_dynamic(
    data_list: list,
    client: int = 0,
    locker_indices: list = [0,1,2,-1],
    save_dir: str = './anda_plot',
    file_name: str = None
) -> None:
    '''
    Plot and save the first 100 images of some subsets of one client. (for drifting and dynamic datasets)
    
    Args:
        data_list (list): A list of dictionaries where each dictionary contains the features and labels for each client.
                          * Output of split_fns
        client (int): The client index to plot the images.
        locker_indices (list): The epoch locker orders to plot. -1 is the testing set.
        save_dir (str): The directory to save the images.
        file_name (str): The prefix of the saved image files.
    '''

    os.makedirs(save_dir, exist_ok=True)

    for data in data_list:
        # Check if the current client matches and if the epoch_locker_order is in locker_indices
        if data['client_number'] == client and data['epoch_locker_order'] in locker_indices:

            # Determine whether we are dealing with training or testing data based on 'train'
            data_type = 'Training' if data['train'] else 'Testing'

            _plot_images(
                data['features'], data['labels'],
                f'Client {data["client_number"]} | {data_type} | Epoch Locker Order: {data["epoch_locker_order"]} | First {{num_images}} Images',
                os.path.join(save_dir, f'{file_name}_client_{data["client_number"]}_epoch_{data["epoch_locker_order"]}_{data_type}_data_plot.png')
            )
