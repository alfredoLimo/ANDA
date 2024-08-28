# Supported types of non-IID-ness

### **`non_iid_type`**

### feature_skew
- Creating feature-skewed sub-datasets with (image) rotation and coloring.

- `set_rotation: bool` Whether assigning rotations

- `rotations: int > 1` The number of possible rotations. Recommended to be {**2**,**4**}, as of **[0°,180°]** and **[0°,90°,180°,270°]**.

- `scaling_rotation_low, scaling_rotation_high: float [0,1]` A scaling factor for non-IID-ness with a softmax function. **0** as uniformly distributed, and **1** as highest possible skewed.

- `set_color: bool` Whether assigning colors

- `colors: int` The number of possible colors. {2,3}.

- `scaling_color_low, scaling_color_high: float [0,1]` A scaling factor for non-IID-ness with a softmax function. **0** as uniformly distributed, and **1** as highest possible skewed.

- `random_order: bool` Shuffling the order of probabilities of feature assignment for each client. When **False**, all clients will have a similar ratio of labels (e.g. red is the most possible color for anyone)

- `show_distribution: bool` verbose for printing out the distributions

### label_skew
- Creating label-skewed sub-datasets.

-  `scaling_label_low, scaling_label_high: float` A scaling factor for non-IID-ness with a softmax function. **0** as uniformly distributed, (fine-tune a higher scaling factor for optimal results).

### feature_label_skew
- Creating feature-skewed sub-datasets with (image) rotation and coloring, while labels are also skewed.
Not suggested being used for creating concept drift datasets.

- `scaling_label_low, scaling_label_high: float` A scaling factor for non-IID-ness with a softmax function. **0** as uniformly distributed, (fine-tune a higher scaling factor for optimal results).

- `set_rotation: bool` Whether assigning rotations

- `rotations: int > 1` The number of possible rotations. Recommended to be {**2**,**4**}, as of **[0°,180°]** and **[0°,90°,180°,270°]**.

- `scaling_rotation_low, scaling_rotation_high: float [0,1]` A scaling factor for non-IID-ness with a softmax function. **0** as uniformly distributed, and **1** as highest possible skewed.

- `set_color: bool` Whether assigning colors

- `colors: int` The number of possible colors. {2,3}.

- `scaling_color_low, scaling_color_high: float [0,1]` A scaling factor for non-IID-ness with a softmax function. **0** as uniformly distributed, and **1** as highest possible skewed.

- `random_order: bool` Shuffling the order of probabilities of feature assignment for each client. When **False**, all clients will have a similar ratio of labels (e.g. red is the most possible color for anyone)

- `show_distribution: bool` verbose for printing out the distributions

### feature_skew_unbalanced
- Creating feature-skewed sub-datasets with (image) rotation and coloring, while the quantity is unbalanced.

- `set_rotation: bool` Whether assigning rotations

- `rotations: int > 1` The number of possible rotations. Recommended to be {**2**,**4**}, as of **[0°,180°]** and **[0°,90°,180°,270°]**.

- `scaling_rotation_low, scaling_rotation_high: float [0,1]` A scaling factor for non-IID-ness with a softmax function. **0** as uniformly distributed, and **1** as highest possible skewed.

- `set_color: bool` Whether assigning colors

- `colors: int` The number of possible colors. {2,3}.

- `scaling_color_low, scaling_color_high: float [0,1]` A scaling factor for non-IID-ness with a softmax function. **0** as uniformly distributed, and **1** as highest possible skewed.

- `random_order: bool` Shuffling the order of probabilities of feature assignment for each client. When **False**, all clients will have a similar ratio of labels (e.g. red is the most possible color for anyone)

- `std_dev: float` Standard deviation of the normal distribution for the data partition. A higher number indicates more unbalanced.

- `permute: bool` Shuffle the data before splitting.

- `show_distribution: bool` verbose for printing out the distributions

### label_skew_unbalanced
- Creating label-skewed sub-datasets, while the quantity is unbalanced.

- `scaling_label_low, scaling_label_high: float` A scaling factor for non-IID-ness with a softmax function. **0** as uniformly distributed, (fine-tune a higher scaling factor for optimal results).

- `std_dev: float` Standard deviation of the normal distribution for the data partition. A higher number indicates more unbalanced.

- `verbose: bool` Print the number of samples for each client.

### feature_condition_skew
- Creating datasets that P(x|y) differs across clients by label swapping. A label-swapping pool is created (e.g. {1,4,7}) and data points of those labels are re-assigned to one label in the pool.

- `random_mode: bool` Randomly choose which labels are in the swapping pool.

- `mixing_label_number: int ` The number of types of labels in the pool if `random_mode`.

- `mixing_label_list: list ` If not `random_mode`, provide a list of types of labels for the swapping pool.

- `scaling_label_low, scaling_label_high: float [0,1]` The probability if a label will be swapped, **0** as of no swapping, **1** as of always swapping.

- `verbose: bool` Verbose. Print clients' swapping maps.

### feature_condition_skew_unbalanced
- Creating unbalanced datasets that P(x|y) differs across clients by label swapping. A label-swapping pool is created (e.g. {1,4,7}) and data points of those labels are re-assigned to one label in the pool.

- `random_mode: bool` Randomly choose which labels are in the swapping pool.

- `mixing_label_number: int ` The number of types of labels in the pool if `random_mode`.

- `mixing_label_list: list ` If not `random_mode`, provide a list of types of labels for the swapping pool.

- `scaling_label_low, scaling_label_high: float [0,1]` The probability if a label will be swapped, **0** as of no swapping, **1** as of always swapping.

- `std_dev: float` Standard deviation of the normal distribution for the data partition. A higher number indicates more unbalanced.

- `permute: bool` Shuffle the data before splitting.

- `verbose: bool` Verbose. Print clients' swapping maps.

### label_condition_skew
- Creating datasets that P(y|x) differs across clients by targeted rotation/coloring.

- `set_rotation: bool` Whether assigning rotations

- `rotations: int > 1` The number of possible rotations. Recommended to be {**2**,**4**}, as of **[0°,180°]** and **[0°,90°,180°,270°]**.

- `set_color: bool` Whether assigning colors

- `colors: int` The number of possible colors. {1,2,3}.

- `random_mode: bool` Randomly choose which (images of) labels will be rotated/colored.

- `rotated_label_number: int` The number of labels to be rotated if `random_mode`.

- `colored_label_number: int` The numbers of labels to be colored if `random_mode`.

- `rotated_label_list: list ` If not `random_mode`, provide a list of types of labels to be rotated.

- `colored_label_list: list ` If not `random_mode`, provide a list of types of labels to be colored.

- `verbose: bool` Verbose. Print clients' swapping maps.

### label_condition_skew_unbalanced
- Creating unbalanced datasets that P(y|x) differs across clients by targeted rotation/coloring.

- `set_rotation: bool` Whether assigning rotations

- `rotations: int > 1` The number of possible rotations. Recommended to be {**2**,**4**}, as of **[0°,180°]** and **[0°,90°,180°,270°]**.

- `set_color: bool` Whether assigning colors

- `colors: int` The number of possible colors. {1,2,3}.

- `random_mode: bool` Randomly choose which (images of) labels will be rotated/colored.

- `rotated_label_number: int` The number of labels to be rotated if `random_mode`.

- `colored_label_number: int` The numbers of labels to be colored if `random_mode`.

- `rotated_label_list: list ` If not `random_mode`, provide a list of types of labels to be rotated.

- `colored_label_list: list ` If not `random_mode`, provide a list of types of labels to be colored.

- `std_dev: float` Standard deviation of the normal distribution for the data partition. A higher number indicates more unbalanced.

- `permute: bool` Shuffle the data before splitting.

- `verbose: bool` Verbose. Print clients' swapping maps.

### feature_condition_skew_with_label_skew
- Creating datasets that P(x|y) differs across clients by label swapping, while labels are already skewed. A label-swapping pool is created (e.g. {1,4,7}) and data points of those labels are re-assigned to one label in the pool.

- `scaling_label_low, scaling_label_high: float` A scaling factor for non-IID-ness with a softmax function. **0** as uniformly distributed, (fine-tune a higher scaling factor for optimal results).

- `random_mode: bool` Randomly choose which labels are in the swapping pool.

- `mixing_label_number: int ` The number of types of labels in the pool if `random_mode`.

- `mixing_label_list: list ` If not `random_mode`, provide a list of types of labels for the swapping pool.

- `scaling_swapping_low, scaling_swapping_high: float [0,1]` The probability if a label will be swapped, **0** as of no swapping, **1** as of always swapping.

- `verbose: bool` Verbose. Print clients' swapping maps.

### label_condition_skew_with_label_skew
- Creating datasets that P(y|x) differs across clients by targeted rotation/coloring, while labels are already skewed.

- `scaling_swapping_low, scaling_swapping_high: float [0,1]` The probability if a label will be swapped, **0** as of no swapping, **1** as of always swapping.

- `set_rotation: bool` Whether assigning rotations

- `rotations: int > 1` The number of possible rotations. Recommended to be {**2**,**4**}, as of **[0°,180°]** and **[0°,90°,180°,270°]**.

- `set_color: bool` Whether assigning colors

- `colors: int` The number of possible colors. {1,2,3}.

- `random_mode: bool` Randomly choose which (images of) labels will be rotated/colored.

- `rotated_label_number: int` The number of labels to be rotated if `random_mode`.

- `colored_label_number: int` The numbers of labels to be colored if `random_mode`.

- `rotated_label_list: list ` If not `random_mode`, provide a list of types of labels to be rotated.

- `colored_label_list: list ` If not `random_mode`, provide a list of types of labels to be colored.

- `verbose: bool` Verbose. Print clients' swapping maps.
