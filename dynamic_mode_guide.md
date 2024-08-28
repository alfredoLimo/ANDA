# Dynamic mode supported types of non-IID-ness

## Mode **`trND_teDR`**

Resembles static (\\) mode as the training set is not changing during all epochs. More details about the color/rotation can be found in the [static mode guide](static_mode_guide.md).

### non_iid_type = `Px`
> Creating feature-skewed (Px) sub-datasets with (image) rotation and coloring.
> A pattern bank will be created based on #rotation and #color. Each DATAPOINT applies one of the patterns in the bank. Each sub-dataset choose one and only distribution of the patterns.
- `rotation_bank: int` The number of rotation patterns. **1** as no rotation.
- `color_bank: int` The number of color patterns. **1** as no color.
- `scaling_low, scaling_high: float` A scaling factor range (randomly uniformly chosen between) for non-IID-ness with a softmax function.
- `reverse_test: bool` Reverse the patterns from the training set for the testing set. (creating strong unseen level)
- `verbose: bool` Whether to print the distribution information.
  
### non_iid_type = `Py`
> Creating label-skewed (Py) sub-datasets.
> Two probability distributions of the labels will be created for each client, for both training and testing.
- `scaling_low, scaling_high: float` A scaling factor range (randomly uniformly chosen between) for non-IID-ness with a softmax function.
- `reverse_test: bool` Reverse the patterns from the training set for the testing set. (creating strong unseen level)
- `verbose: bool` Whether to print the distribution information.

### non_iid_type = `Px_y`
> Creating feature condition skewed (P(x|y)) sub-datasets.
> For each client, train and test datasets are assigned with diff random label-swapping partterns.
- `mixing_num: int` The number of mixed labels. A list of classes (#len = mixing_num) will be generated as the only swapping pool.
- `scaling_low, scaling_high: float` A scaling factor range (randomly uniformly chosen between) for non-IID-ness with a softmax function.
- `verbose: bool` Whether to print the distribution information.

### non_iid_type = `Py_x`
> Creating feature condition skewed (P(y|x)) sub-datasets.
> Randomly chosen #rotated_label_number and #colored_label_number of classes will be rotated/colored. And the pattern is randomly generated based on #rotation_bank and #color_bank.
- `rotation_bank: int` The number of rotation patterns. **1** as no rotation.
- `color_bank: int` The number of color patterns. **1** as no color.
- `rotated_label_number: int` The number of labels (classes) to rotate.
- `colored_label_number: int` The number of labels (classes) to color.
- `verbose: bool` Whether to print the distribution information.