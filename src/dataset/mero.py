import json
import clip
import torch
import numpy as np
from torch.utils.data import Dataset
from utils import meronyms_with_definition


class MeroDataset(Dataset):
    def __init__(self, clip_root, num_classes, dataset, hierarchy_path):
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
        min_num_mero = 10
        for meros in hierarchy.values():
            max_num_mero = max(max_num_mero, len(meros))
            min_num_mero = min(min_num_mero, len(meros))
            mero_set.update(meros)

        self.max_num_mero = max_num_mero
        self.min_num_mero = min_num_mero
        self.dataset = dataset
        self.mero_labels = sorted(mero_set)

        # Build meronym label to index mapping
        self.mero_label_to_idx = {label: idx for idx, label in enumerate(self.mero_labels)}
        self.num_mero_classes = len(self.mero_labels)

        self.base_to_mero = [[] for i in range(num_classes)]
        for k, meros in hierarchy.items():
            for i in range(len(meros)):
                if meros[i] in self.mero_label_to_idx:
                    self.base_to_mero[self.dataset.class_to_idx[k]].append(self.mero_label_to_idx[meros[i]])
                else:
                    raise ValueError(f"Meronym label '{meros[i]}' not found in meronym_label_to_idx mapping.")
        for i in range(len(self.base_to_mero)):
            self.base_to_mero[i] = np.array(self.base_to_mero[i])

        with torch.no_grad():
            # Load CLIP model for text embeddings
            clip_model, _ = clip.load(clip_root, device='cpu')
            # include 'no object' meronym class
            meronyms = meronyms_with_definition(self.mero_label_to_idx)
            text_tokens = clip.tokenize(meronyms)
            self.text_weights = clip_model.encode_text(text_tokens).float()
            self.text_weights = torch.nn.functional.normalize(self.text_weights, p=2, dim=-1)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, base_label = self.dataset[idx]
        mero_labels = self.base_to_mero[base_label]

        embed_no = torch.randperm(mero_labels.shape[0])[:self.min_num_mero]
        embed_no = mero_labels[embed_no]
        embed = self.text_weights[embed_no]

        padding = np.full(self.max_num_mero - mero_labels.shape[0], -1)
        mero_labels = np.concatenate([mero_labels, padding])

        data = {
            "images": image,
            "embeds": embed
        }
        labels = {
            'base': base_label,
            'mero': mero_labels
        }
        return data, labels
