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
 
| **Non-IID type** | **P(x)**     | **P(y)**           | **P(x\|y)**                            | **P(y\|x)**                          | **Quantity**                      |
|------------------|--------------|--------------------|----------------------------------------|--------------------------------------|-----------------------------------|
| **P(x)**         | `feature_skew` | `feature_label_skew` | /                        | /                        | `feature_skew_unbalanced`           |
| **P(y)**         | /            | `label_skew`         | `feature_condition_skew_with_label_skew` | `label_condition_skew_with_label_skew` | `label_skew_unbalanced`             |
| **P(x\|y)**      | /            | /                  | `feature_condition_skew`                 | /                      | `feature_condition_skew_unbalanced` |
| **P(y\|x)**      | /            | /                  | /                                      | `label_condition_skew`                 | `label_condition_skew_unbalanced`   |
| **Quantity**     | /            | /                  | /                                      | /                                    | `split_unbalanced`                  |
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
- **non_iid_type**: str = "feature_skew", types of non-IID-ness. [More details](appendix.md)
- **mode**: str = "auto", using AUTO mode
- **non_iid_level**: str = "medium", auto setting a level for non-IID, "low","medium", or "high"
- **show_features**: bool = False, show generated feature details if any
- **show_labels**: bool = False, show generated label details if any (also save the imgs)
- **random_seed**: int = 42, a random seed to repeat your results

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
Results: (showing data from first four clients, try to repeat it with the same seed)

<img src="https://github.com/alfredoLimo/ANDA/assets/68495667/2cbd40db-0848-4f2f-b564-f686d8e1a4e7" alt="Client 0" width="400"/>
<img src="https://github.com/alfredoLimo/ANDA/assets/68495667/d1fc756d-8c9f-4e29-9b26-58b315619971" alt="Client 1" width="400"/>
<img src="https://github.com/alfredoLimo/ANDA/assets/68495667/42edd1ad-4838-4f53-873c-4bb6f5a54b87" alt="Client 2" width="400"/>
<img src="https://github.com/alfredoLimo/ANDA/assets/68495667/1ae8f2e5-82c9-4868-8493-3fc817b08192" alt="Client 3" width="400"/>

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
    non_iid_type = "label_condition_skew",
    mode = "manual",
    show_labels = True,
    random_seed = 42,
    set_color = True,
    colors = 3,
    random_mode = True,
    colored_label_number = 4,
)
```
Results: (showing data from first four clients, try to repeat it with the same seed)

<img src="https://github.com/alfredoLimo/ANDA/assets/68495667/c5aa40c7-e078-4564-b786-8ee3a901e6fa" alt="Client 0" width="400"/>
<img src="https://github.com/alfredoLimo/ANDA/assets/68495667/708526dc-e7c1-4828-840e-851a1d7e0ab3" alt="Client 1" width="400"/>
<img src="https://github.com/alfredoLimo/ANDA/assets/68495667/fe9ee7a1-4be5-47bd-b242-bc0a16c96239" alt="Client 2" width="400"/>
<img src="https://github.com/alfredoLimo/ANDA/assets/68495667/a9b75621-5ad5-4d09-be36-b96e38e7def2" alt="Client 3" width="400"/>



## MORE ON NON-IID
[Independent and identically distributed (IID) random variables](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables)

[Federated learning on non-IID data: A survey](https://www.sciencedirect.com/science/article/pii/S0925231221013254)

## UNDER CONSTRUCTION
- Dynamic/drifting non-IID datasets.

<table>
  <tr>
    <th>Training</th>
    <th>Testing</th>
    <th>Supporting</th>
    <th>Module Name</th>
  </tr>
  <tr>
    <td rowspan="2">No drifting (with a relatively large in size dataset)</td>
    <td>Drifting</td>
    <th>Yes</th>
    <th>trND_teDR</th>
    
  </tr>
  <tr>
    <td>N/A (No drifting for both is static dataset)</td>
    <th>No</th>
    <th>\</th>
    
  </tr>

  
  <tr>
    <td rowspan="2">Drifting with accumulation</td>
    <td>Drifting</td>
    <th>Yes</th>
    <th>trDA_teDR</th>
    
  </tr>
  <tr>
    <td>No drifting</td>
    <th>Yes</th>
    <th>trDA_teND</th>
    
  </tr>

  <tr>
    <td rowspan="2">Drifting without accumulation</td>
    <td>Drifting</td>
    <th>Yes</th>
    <th>trDR_teDR</th>
    
    
  </tr>
  <tr>
    <td>No drifting</td>
    <th>Yes</th>
    <th>trDR_teND</th>
    
  </tr>
*accumulation: drifting data are <b>appended to</b>, not replacing old datasets.
</table>
