from torch.utils.data import Dataset


class TabularDataset(Dataset):
    def __init__(self, x, z, y) -> None:
        """
        Initialize the custom dataset with x (features), z (sensitive values), and y (targets).

        Args:
        x (list of tensors): List of input feature tensors.
        z (list): List of sensitive values.
        y (list): List of target values.

        """
        self.samples = x
        self.sensitive_features = z
        self.targets = y
        self.indexes = range(len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a single data point from the dataset.

        Args:
        idx (int): Index to retrieve the data point.

        Returns:
        sample (dict): A dictionary containing 'x', 'z', and 'y'.

        """
        x_sample = self.samples[idx]
        if self.sensitive_features:
            z_sample = self.sensitive_features[idx]
        else:
            z_sample = None
        y_sample = self.targets[idx]

        return x_sample, z_sample, y_sample
