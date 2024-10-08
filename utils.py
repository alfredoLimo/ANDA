import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import truncnorm
import os

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
    train_features = []
    train_labels = []
    test_features = []
    test_labels = []
    for client_data in data:
        train_features.append(client_data['train_features'])
        train_labels.append(client_data['train_labels'])
        test_features.append(client_data['test_features'])
        test_labels.append(client_data['test_labels'])

    # Concatenate all the data
    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    test_features = torch.cat(test_features, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

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

    # Extracting train and test images and labels
    train_images = torch.stack([data[0] for data in train_dataset]).squeeze(1)
    test_images = torch.stack([data[0] for data in test_dataset]).squeeze(1)
    
    if dataset_name in ["CIFAR10", "CIFAR100"]:
        train_labels = torch.tensor(train_dataset.targets).clone().detach()
        test_labels = torch.tensor(test_dataset.targets).clone().detach()
    else:
        train_labels = train_dataset.targets.clone().detach()
        test_labels = test_dataset.targets.clone().detach()

    return [train_images, train_labels, test_images, test_labels]

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
    
    rotated_images = []
    
    for img_tensor, degree in zip(dataset, degrees):
        # Convert the tensor to a PIL image
        img = transforms.ToPILImage()(img_tensor)
        # Rotate the image
        rotated_img = img.rotate(degree)
        
        # Convert the PIL image back to a tensor
        rotated_img_tensor = transforms.ToTensor()(rotated_img).squeeze(0)
        
        rotated_images.append(rotated_img_tensor)
    
    # Stack all tensors into a single tensor
    rotated_dataset = torch.stack(rotated_images)
    
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

    for i, color in enumerate(colors):
        # Map the grayscale values to the specified color
        if color == 'red':
            colored_dataset[i, 0, :, :] = 1  # Set the red channel for the image
        elif color == 'green':
            colored_dataset[i, 1, :, :] = 1  # Set the green channel for the image
        elif color == 'blue':
            colored_dataset[i, 2, :, :] = 1  # Set the blue channel for the image
        elif color == "gray":
            pass
        else:
            raise ValueError("Color must be 'red', 'green', or 'blue'")

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

def create_sub_dataset(
        features, 
        labels, 
        probabilities, 
        num_points
):
    selected_indices = []
    while len(selected_indices) < num_points:
        for i in range(len(labels)):
            if torch.rand(1).item() < probabilities[labels[i]].item():
                selected_indices.append(i)
            if len(selected_indices) >= num_points:
                break
    
    selected_indices = torch.tensor(selected_indices)
    sub_features = features[selected_indices]
    sub_labels = labels[selected_indices]
    remaining_indices = torch.ones(len(labels), dtype=torch.bool)
    remaining_indices[selected_indices] = 0
    remaining_features = features[remaining_indices]
    remaining_labels = labels[remaining_indices]

    return sub_features, sub_labels, remaining_features, remaining_labels

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
        train_labels = data['train_labels']
        test_labels = data['test_labels']
        
        train_label_counts = torch.tensor([train_labels.tolist().count(x) for x in range(10)])
        test_label_counts = torch.tensor([test_labels.tolist().count(x) for x in range(10)])
        
        print(f"Client {i}:")
        print("Training label counts:", train_label_counts)
        print("Test label counts:", test_label_counts)
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
    for _, data in enumerate(data_list):
        # Count the label occurrences for each class (assuming 10 classes)
        label_counts = torch.tensor([data['labels'].tolist().count(x) for x in range(10)])

        print(
            f"Client {data['client_number']} | {'Train' if data['train'] else 'Test'} | "
            f"Epoch Locker Order: {data['epoch_locker_order']} | "
            f"Label Counts: {label_counts.tolist()}"
        )
    
    return

def plot_static(
    data_list: list,
    plot_indices: list = [0,1,2,3],
    save_dir: str = './anda_plot',
    file_name: str = None
) -> None:
    '''
    Print label counts and plot images.
    
    Args:
        data_list (list): A list of dictionaries where each dictionary contains the features and labels for each client.
                          * Output of split_fns
        plot_indices (list): A list of indices to plot the first 100 images for each client.
        save_dir (str): The directory to save the images.

    Warning:
        Working for only 10 classes dataset. (EMNIST e CIFAR100 NOT SUPPORTED)
    '''

    os.makedirs(save_dir, exist_ok=True)

    for idx in plot_indices:
        if idx < len(data_list):
            data = data_list[idx]

            # training data plot
            train_features = data['train_features']
            train_labels = data['train_labels']
            
            num_images = min(100, train_features.shape[0])
            fig, axes = plt.subplots(10, 10, figsize=(15, 15))
            fig.suptitle(f'Dictionary {idx} - First {num_images} Training Images', fontsize=16)
            
            for i in range(num_images):
                ax = axes[i // 10, i % 10]
                image = train_features[i]
                
                if image.shape[0] == 3:
                    # For CIFAR (3, H, W) -> (H, W, 3)
                    image = image.permute(1, 2, 0).numpy()
                else:
                    # For MNIST (1, H, W) -> (H, W)
                    image = image.squeeze().numpy()
                
                ax.imshow(image, cmap='gray' if image.ndim == 2 else None)
                ax.set_title(train_labels[i].item())
                ax.axis('off')
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

            save_path = os.path.join(save_dir, f'{file_name}_client_{idx}_train_data_plot.png')
            plt.savefig(save_path)
            print(f"Saved images to {save_path}")

            # testing data plot
            test_features = data['test_features']
            test_labels = data['test_labels']
            
            num_images = min(100, test_features.shape[0])
            fig, axes = plt.subplots(10, 10, figsize=(15, 15))
            fig.suptitle(f'Dictionary {idx} - First {num_images} Testing Images', fontsize=16)
            
            for i in range(num_images):
                ax = axes[i // 10, i % 10]
                image = test_features[i]
                
                if image.shape[0] == 3:
                    # For CIFAR (3, H, W) -> (H, W, 3)
                    image = image.permute(1, 2, 0).numpy()
                else:
                    # For MNIST (1, H, W) -> (H, W)
                    image = image.squeeze().numpy()
                
                ax.imshow(image, cmap='gray' if image.ndim == 2 else None)
                ax.set_title(test_labels[i].item())
                ax.axis('off')
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

            save_path = os.path.join(save_dir, f'{file_name}_client_{idx}_test_data_plot.png')
            plt.savefig(save_path)
            print(f"Saved images to {save_path}")

def plot_dynamic(
    data_list: list,
    client: int = 0,
    locker_indices: list = [0,1,2,-1],
    save_dir: str = './anda_plot',
    file_name: str = None
) -> None:
    '''
    Print label counts and plot images. (for drifting and dynamic datasets)
    
    Args:
        data_list (list): A list of dictionaries where each dictionary contains the features and labels for each client.
                          * Output of split_fns
        client (int): The client index to plot the images.
        locker_indices (list): A list of indices to plot the images.
        count (bool): If True, print the label counts.
        save_dir (str): The directory to save the images.
        file_name (str): The name of the file to save the images.

    Warning:
        Work with 10 classes dataset. (EMNIST e CIFAR100 NOT #TODO SUPPORTED)
    '''

    os.makedirs(save_dir, exist_ok=True)

    for _ , data in enumerate(data_list):
        # Check if the current client matches and if the epoch_locker_order is in locker_indices
        if data['client_number'] == client and data['epoch_locker_order'] in locker_indices:

            # Features and labels are based on the current data dictionary
            features, labels = data['features'], data['labels']
            
            # Determine whether we are dealing with training or testing data based on 'train'
            data_type = 'Training' if data['train'] else 'Testing'

            num_images = min(100, features.shape[0])
            fig, axes = plt.subplots(10, 10, figsize=(15, 15))
            fig.suptitle(f'Client {data["client_number"]} | {data_type} | Epoch Locker Order: {data["epoch_locker_order"]} | First {num_images} Images', fontsize=16)

            for i in range(num_images):
                ax = axes[i // 10, i % 10]
                image = features[i]
                
                if image.shape[0] == 3:
                    # For CIFAR (3, H, W) -> (H, W, 3)
                    image = image.permute(1, 2, 0).numpy()
                else:
                    # For MNIST (1, H, W) -> (H, W)
                    image = image.squeeze().numpy()

                ax.imshow(image, cmap='gray' if image.ndim == 2 else None)
                ax.set_title(labels[i].item())
                ax.axis('off')

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()
            
            save_path = os.path.join(save_dir, f'{file_name}_client_{data["client_number"]}_epoch_{data["epoch_locker_order"]}_{data_type}_data_plot.png')
            plt.savefig(save_path)
            print(f"Saved images to {save_path}")
            
