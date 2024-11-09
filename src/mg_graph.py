import nltk
from nltk.corpus import wordnet as wn
import torch

nltk.download('wordnet')
nltk.download('omw-1.4')  # For extended WordNet

class MultiGranGraph:
    def __init__(self, config):
        self.depth = config['depth']
        self.labels = config['labels']  # List of fine-grained labels
        self.label_hierarchy = self.build_hierarchy()
        self.labels_per_stage = self.get_labels_per_stage()
        self.H_matrices = self.build_H_matrices()
        self.label_to_index = self.get_label_to_index()
        self.index_to_label = self.get_index_to_label()
    
    def build_hierarchy(self):
        """
        Build a label hierarchy using WordNet.
        Returns a dictionary mapping child labels to parent labels.
        """
        hierarchy = {}
        for label in self.labels:
            synsets = wn.synsets(label, pos=wn.NOUN)
            if not synsets:
                hierarchy[label] = None  # No parent found
                continue
            synset = synsets[0]  # Take the first synset
            hypernyms = synset.hypernyms()
            if hypernyms:
                parent = hypernyms[0].lemmas()[0].name().replace('_', ' ')
                hierarchy[label] = parent
            else:
                hierarchy[label] = None  # Root node
        return hierarchy
    
    def get_labels_per_stage(self):
        """
        Organize labels into granularity levels based on hierarchy.
        Returns a list of lists, where each sublist contains labels at that stage.
        Stage 0 is the root labels.
        """
        labels_per_stage = []
        # Stage 0: root labels (labels with no parents)
        roots = [label for label, parent in self.label_hierarchy.items() if parent is None]
        labels_per_stage.append(roots)
    
        # Subsequent stages
        for d in range(1, self.depth + 1):
            prev_stage = labels_per_stage[-1]
            current_stage = [label for label, parent in self.label_hierarchy.items() if parent in prev_stage]
            if not current_stage:
                break
            labels_per_stage.append(current_stage)
    
        return labels_per_stage
    
    def build_H_matrices(self):
        """
        Build Hierarchical Relationship Matrices for each stage.
        H_matrices[t] corresponds to stage t (1-based indexing)
        Each H is a binary matrix indicating parent-child relationships.
        """
        H_matrices = []
        for t in range(1, len(self.labels_per_stage)):
            parent_labels = self.labels_per_stage[t-1]
            current_labels = self.labels_per_stage[t]
            H = torch.zeros(len(parent_labels), len(current_labels), dtype=torch.float32)
            for i, parent in enumerate(parent_labels):
                for j, child in enumerate(current_labels):
                    if self.label_hierarchy.get(child) == parent:
                        H[i, j] = 1.0
            H_matrices.append(H)
        return H_matrices
    
    def get_H_matrix(self, stage):
        """
        Retrieve the H matrix for a specific stage (1-based indexing)
        """
        if stage < 1 or stage > len(self.H_matrices):
            raise ValueError(f"Stage {stage} is out of bounds.")
        return self.H_matrices[stage - 1]
    
    def get_label_to_index(self):
        """
        Create label to index mapping for each stage
        Returns a list of dicts
        """
        label_to_index = []
        for stage_labels in self.labels_per_stage:
            mapping = {label: idx for idx, label in enumerate(stage_labels)}
            label_to_index.append(mapping)
        return label_to_index
    
    def get_index_to_label(self):
        """
        Create index to label mapping for each stage
        Returns a list of dicts
        """
        index_to_label = []
        for stage_labels in self.labels_per_stage:
            mapping = {idx: label for idx, label in enumerate(stage_labels)}
            index_to_label.append(mapping)
        return index_to_label
    
    def get_num_classes_per_stage(self):
        """
        Returns a list indicating the number of classes per stage
        """
        return [len(stage_labels) for stage_labels in self.labels_per_stage]
    
    def get_label_sequence(self, fine_label):
        """
        Given a fine-grained label, return the label sequence up to that stage
        """
        sequence = []
        current_label = fine_label
        while current_label in self.label_hierarchy and self.label_hierarchy[current_label]:
            sequence.insert(0, current_label)
            current_label = self.label_hierarchy[current_label]
        if current_label:
            sequence.insert(0, current_label)
        return sequence
