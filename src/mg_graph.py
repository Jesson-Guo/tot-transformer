import nltk
from typing import List, Optional
from nltk.corpus import wordnet as wn
from typing import Dict, List, Set

class LabelNode:
    """
    Represents a node in the multi-granularity structure graph.
    """
    def __init__(self, 
                 label: str, 
                 semantic_description: str, 
                 depth: int):
        """
        Args:
            label (str): The label or name of the node (l_v).
            semantic_description (str): The semantic description or definition of the node (s_v).
            depth (int): The depth or level of the node in the hierarchy (d_v).
        """
        self.label = label              # l_v
        self.semantic_description = semantic_description  # s_v
        self.depth = depth              # d_v
        self.parents: List['LabelNode'] = []  # Parent nodes
        self.children: List['LabelNode'] = [] # Child nodes
        # Additional attributes can be added here if necessary
    
    def add_parent(self, parent_node: 'LabelNode'):
        """
        Adds a parent node to this node.
        """
        if parent_node not in self.parents:
            self.parents.append(parent_node)
            parent_node.children.append(self)
    
    def __repr__(self):
        return f"LabelNode(label='{self.label}', depth={self.depth})"

class MultiGranGraph:
    """
    Represents the multi-granularity structure graph.
    """
    def __init__(self, labels: List[str], max_depth: int = 5):
        """
        Args:
            labels (List[str]): List of labels (leaf nodes) to build the graph from.
            max_depth (int): Maximum depth for hypernym extraction.
        """
        self.max_depth = max_depth
        self.nodes: Dict[str, LabelNode] = {}      # All nodes by label
        self.layers: Dict[int, List[LabelNode]] = {}  # Nodes organized by depth
        self.edges: Set[tuple] = set()             # Edges represented as (parent_label, child_label)
        
        # Build the graph
        self.build_graph(labels)
    
    def build_graph(self, labels: List[str]):
        """
        Builds the multi-granularity structure graph from the given labels.
        """
        # Ensure WordNet data is downloaded
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        
        # Process each label
        for label in labels:
            self._process_label(label)
        
        # Ensure all non-root nodes have parent nodes
        for node in self.nodes.values():
            if node.depth > 0 and not node.parents:
                # Assign a default parent if none exists (e.g., 'entity' in WordNet)
                root_synset = wn.synset('entity.n.01')
                root_label = root_synset.name().split('.')[0]
                root_node = self.nodes.get(root_label)
                if not root_node:
                    root_node = LabelNode(
                        label=root_label,
                        semantic_description=root_synset.definition(),
                        depth=0
                    )
                    self.nodes[root_label] = root_node
                    self.layers.setdefault(0, []).append(root_node)
                node.add_parent(root_node)
                self.edges.add((root_node.label, node.label))
    
    def _process_label(self, label: str):
        """
        Processes a single label to add it and its hypernyms to the graph.
        """
        synsets = wn.synsets(label)
        if not synsets:
            # Handle cases where the label is not found in WordNet
            # Create a node with depth 0 (could adjust as needed)
            node = self.nodes.get(label)
            if not node:
                node = LabelNode(label=label, semantic_description="", depth=0)
                self.nodes[label] = node
                self.layers.setdefault(0, []).append(node)
            return
        
        # Use the first synset as the primary one
        synset = synsets[0]
        depth = synset.min_depth()
        node = self.nodes.get(label)
        if not node:
            node = LabelNode(
                label=label,
                semantic_description=synset.definition(),
                depth=depth
            )
            self.nodes[label] = node
            self.layers.setdefault(depth, []).append(node)
        
        # Recursively process hypernyms up to max_depth
        self._add_hypernyms(node, synset, current_depth=depth)
    
    def _add_hypernyms(self, node: LabelNode, synset, current_depth: int):
        """
        Recursively adds hypernyms of the given synset to the graph.
        """
        if current_depth == 0 or current_depth >= self.max_depth:
            return
        
        hypernyms = synset.hypernyms()
        for hypernym in hypernyms:
            hypernym_label = hypernym.name().split('.')[0]
            hypernym_node = self.nodes.get(hypernym_label)
            hypernym_depth = hypernym.min_depth()
            if not hypernym_node:
                # Create new hypernym node
                hypernym_node = LabelNode(
                    label=hypernym_label,
                    semantic_description=hypernym.definition(),
                    depth=hypernym_depth
                )
                self.nodes[hypernym_label] = hypernym_node
                self.layers.setdefault(hypernym_depth, []).append(hypernym_node)
            # Add parent-child relationship
            node.add_parent(hypernym_node)
            self.edges.add((hypernym_node.label, node.label))
            # Continue recursively
            self._add_hypernyms(hypernym_node, hypernym, current_depth=hypernym_depth)
    
    def get_node(self, label: str) -> Optional[LabelNode]:
        """
        Retrieves a node by its label.
        """
        return self.nodes.get(label)
    
    def __repr__(self):
        return f"MultiGranGraph(num_nodes={len(self.nodes)}, num_edges={len(self.edges)})"

if __name__ == '__main__':
    # Example labels (leaf nodes)
    labels = ['dog', 'cat', 'apple', 'banana']

    # Initialize the graph with these labels
    graph = MultiGranGraph(labels=labels, max_depth=5)

    # Print graph summary
    print(graph)

    # Access a specific node
    node = graph.get_node('dog')
    print(f"Node: {node}")
    print(f"Parents of '{node.label}': {[parent.label for parent in node.parents]}")
    print(f"Children of '{node.label}': {[child.label for child in node.children]}")

    # Print nodes at each layer
    for depth, nodes_at_depth in graph.layers.items():
        print(f"Depth {depth}: {[node.label for node in nodes_at_depth]}")
