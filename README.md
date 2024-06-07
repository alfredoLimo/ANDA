<p align="center">
  <img src="https://github.com/alfredoLimo/ANDA/assets/68495667/593c8ca0-fce0-4fa7-ba73-9d900ce95559" alt="Picture1" width="700"/>
</p>

# ABOUT ANDA
**A** **N**on-IID **D**ata generator supporting **A**ny kind. Generate your non-IID datasets with one line.

# FEATURES
- Repeat your Federated Learning (FL) experiments with non-IID datasets and without saving it!
- Supporting five public datasets: **MNIST**, **EMNIST**, **FMNIST**, **CIFAR10**, and **CIFAR100**
- Supporting five types of non-IID-ness and their mixtures:
  - **feature distribution skew**: P(x)
  - **label distribution skew:** P(y)
  - **concept drift: feature condition skew:** P(x|y)
  - **concept drift: label condition skew:** P(y|x)
  - **quantity skew**
- Supporting two modes: **AUTO** and **MANUAL**.
---
## USAGE WITH ONE LINE
- Clone ANDA repo to your working repo
- Create the default non-IID dataset with one line using `load_split_datasets` as follows:
```python
from ANDA import anda

new_dataset = anda.load_split_datasets()
```

## QUICK START WITH AUTO MODE

`load_split_datasets` **parameters**

- **dataset_name**: str = "MNIST", "EMNIST", "FMNIST", "CIFAR10", or "CIFAR100"
- **client_number**: int = 10, number of clients/sub-datasets
- **non_iid_type**: str = "feature_skew", types of non-IID-ness. [More details](appendix.md).
- **mode**: str = "auto", using AUTO mode.
- **non_iid_level**: str = "medium"
  - "low":
  - "medium":
  - "high":
- **show_features**: bool = False, show generated feature details if any
- **show_labels**: bool = False, show generated label details if any (also save the imgs)
- **random_seed**: int = 42, a random seed to repeat your results

```python
from ANDA import anda

new_dataset = anda.load_split_datasets()
```

## CUSTOMIZE WITH MANUAL MODE

`load_split_datasets` **parameters**

- **dataset_name**: str = "MNIST", "EMNIST", "FMNIST", "CIFAR10", or "CIFAR100"
- **client_number**: int = 10, number of clients/sub-datasets
- **non_iid_type**: str = "feature_skew", types of non-IID-ness. [More details](appendix.md).
- **mode**: str = "manual", using MANUAL mode.
- **show_features**: bool = False, show generated feature details if any
- **show_labels**: bool = False, show generated label details if any (also save the imgs)
- **random_seed**: int = 42, a random seed to repeat your results
- **\*\*kwargs**: customized parameters for chosen non-IID type. [More details](appendix.md).

```python
new_dataset = anda.load_split_datasets(
    dataset_name = "MNIST",
    client_number = 10,
    non_iid_type = "feature_skew",
    mode = "auto",
    non_iid_level = "high",
    show_labels = True,
    random_seed = 42
)
```
Results: (showing )


## MORE ON NON-IID

## FUTURE WORK
