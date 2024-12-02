import json
import numpy as np
from torch.utils.data import Dataset


class MeroDataset(Dataset):
    def __init__(self, num_classes, dataset, hierarchy_path):
        """
        Args:
            dataset: An instance of a PyTorch Dataset (e.g., datasets.CIFAR100)
            hierarchy_path: Path to a dictionary mapping base labels to lists of meronym labels
        """
        with open(hierarchy_path, 'r') as f:
            hierarchy = json.load(f)
        assert hierarchy != {}, "hierarchy is empty, please use Assistant to construct it first."
        mero_set = set()
        max_num_mero = 0
        for meros in hierarchy.values():
            max_num_mero = max(max_num_mero, len(meros))
            mero_set.update(meros)

        self.dataset = dataset
        self.max_num_mero = max_num_mero
        self.mero_labels = sorted(mero_set)

        # Build meronym label to index mapping
        self.mero_label_to_idx = {label: idx for idx, label in enumerate(self.mero_labels)}
        self.num_mero_classes = len(self.mero_labels)
        self.num_base_classes = num_classes

        self.hierarchy = {}
        for k, meros in hierarchy.items():
            self.hierarchy[self.dataset.class_to_idx[k]] = np.zeros(self.max_num_mero, dtype=np.int64) + len(self.mero_labels)
            for i in range(len(meros)):
                if meros[i] in self.mero_label_to_idx:
                    self.hierarchy[self.dataset.class_to_idx[k]][i] = self.mero_label_to_idx[meros[i]]
                else:
                    raise ValueError(f"Meronym label '{meros[i]}' not found in meronym_label_to_idx mapping.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, base_label = self.dataset[idx]
        mero_labels = self.hierarchy.get(base_label, [])
        labels = {
            'base': base_label,
            'mero': mero_labels
        }
        return data, labels
