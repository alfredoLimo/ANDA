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

## CUSTOMIZE WITH MANUAL MODE








## MORE ON NON-IID

## FUTURE WORK
