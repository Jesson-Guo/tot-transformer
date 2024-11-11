import re
import pickle
import collections
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import wordnet as wn
from typing import Dict, List, Set, Optional

class LabelNode:
    """
    Represents a node in the multi-granularity structure graph.
    """
    def __init__(self, 
                 label: str, 
                 semantic_description: str, 
                 depth: int):
        """
        Initializes a LabelNode.

        Args:
            label (str): The label or name of the node (l_v).
            upper_labels (Dict[int, List[str]]): labels of upper nodes, divided by level (s_v).
            depth (int): The depth or level of the node in the hierarchy (d_v).
        """
        self.label = label                        # l_v
        self.depth = depth                        # d_v
        self.parents: List['LabelNode'] = []      # Parent nodes
        self.children: List['LabelNode'] = []     # Child nodes
        self.upper_labels = {}                    # s_v

    def add_parent(self, parent_node: 'LabelNode'):
        """
        Adds a parent node to this node and updates the parent's children list.

        Args:
            parent_node (LabelNode): The parent node to be added.
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
    def __init__(self):
        """
        Initializes an empty MultiGranGraph.
        """
        self.nodes: Dict[str, LabelNode] = {}          # All nodes by label and depth
        self.layers: Dict[int, List[LabelNode]] = {}   # Nodes organized by depth
        self.edges: Set[tuple] = set()                 # Edges represented as (parent_label, child_label)

        self.H_matrices = []

    def add_node(self, node: LabelNode):
        """
        Adds a node to the graph.

        Args:
            node (LabelNode): The node to be added.
        """
        node_key = f"{node.label}_{node.depth}"
        if node_key not in self.nodes:
            self.nodes[node_key] = node
            if node.depth not in self.layers:
                self.layers[node.depth] = []
            self.layers[node.depth].append(node)

    def add_edge(self, parent_label: str, child_label: str):
        """
        Adds an edge between two nodes in the graph.

        Args:
            parent_label (str): The label of the parent node.
            child_label (str): The label of the child node.
        """
        parent_node = self.get_node_by_label_and_depth(parent_label)
        child_node = self.get_node_by_label_and_depth(child_label)
        if parent_node and child_node:
            edge = (parent_label, child_label)
            if edge not in self.edges:
                self.edges.add(edge)
                child_node.add_parent(parent_node)

    def get_node(self, label: str) -> Optional[LabelNode]:
        """
        Retrieves a node by its label. Returns the first node found with that label.

        Args:
            label (str): The label of the node to retrieve.

        Returns:
            Optional[LabelNode]: The corresponding LabelNode if found, else None.
        """
        for node in self.nodes.values():
            if node.label == label:
                return node
        return None

    def get_node_by_label_and_depth(self, label: str) -> Optional[LabelNode]:
        """
        Retrieves a node by its label and depth.

        Args:
            label (str): The label of the node to retrieve.

        Returns:
            Optional[LabelNode]: The corresponding LabelNode if found, else None.
        """
        node_key = f"{label}_{self.get_depth_by_label(label)}"
        return self.nodes.get(node_key)

    def get_depth_by_label(self, label: str) -> Optional[int]:
        """
        Retrieves the depth of a node by its label.

        Args:
            label (str): The label of the node.

        Returns:
            Optional[int]: The depth of the node if found, else None.
        """
        for depth, nodes in self.layers.items():
            for node in nodes:
                if node.label == label:
                    return depth
        return None

    def build_H_matrices(self):
        """
        Build Hierarchical Relationship Matrices for each stage.
        H_matrices[t] corresponds to stage t (1-based indexing)
        Each H is a binary matrix indicating parent-child relationships.
        """
        for t in range(1, len(self.labels_per_stage)):
            parent_labels = self.labels_per_stage[t-1]
            current_labels = self.labels_per_stage[t]
            H = np.zeros(len(parent_labels), len(current_labels), dtype=np.float32)
            for i, parent in enumerate(parent_labels):
                for j, child in enumerate(current_labels):
                    if self.label_hierarchy.get(child) == parent:
                        H[i, j] = 1.0
            self.H_matrices.append(H)

    def save(self, filename: str):
        """
        Saves the current graph to a file using pickle serialization.

        Args:
            filename (str): The path to the file where the graph will be saved.
        """
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
            print(f"Graph successfully saved to '{filename}'.")
        except Exception as e:
            print(f"An error occurred while saving the graph: {e}")

    @classmethod
    def load(cls, filename: str) -> 'MultiGranGraph':
        """
        Loads a graph from a file using pickle deserialization.

        Args:
            filename (str): The path to the file from which to load the graph.

        Returns:
            MultiGranGraph: The loaded graph instance.
        """
        try:
            with open(filename, 'rb') as f:
                graph = pickle.load(f)
            if not isinstance(graph, cls):
                raise TypeError("The loaded object is not a MultiGranGraph instance.")
            print(f"Graph successfully loaded from '{filename}'.")
            return graph
        except FileNotFoundError:
            print(f"File '{filename}' not found.")
            raise
        except Exception as e:
            print(f"An error occurred while loading the graph: {e}")
            raise

    def __repr__(self):
        return f"MultiGranGraph(num_nodes={len(self.nodes)}, num_edges={len(self.edges)})"

def build_graph(labels: List[str], graph: 'MultiGranGraph', max_depth: int = 3, max_parent: int = 2, depth_diff: int = 2):
    """
    Constructs the multi-granularity structure graph using an efficient algorithm.

    Args:
        labels (List[str]): List of labels to build the graph from (nodes in layer T).
        graph (MultiGranGraph): The graph object to be constructed.
        max_depth (int): The maximum depth (number of layers), T.
        max_parent (int): The maximum number of parent nodes a single node can have.
        depth_diff (int): The maximum allowed depth difference for hypernyms.
    """
    T = max_depth  # The finest-grained layer
    graph.layers = {}  # Reset layers
    graph.layers[T] = []

    # Get the depth of 'Thing' in WordNet
    thing_depth = get_thing_depth()

    # Keep track of labels in each layer to prevent duplicates
    layer_labels = {t: set() for t in range(T + 1)}

    # Step 1: Initialize layer T with input labels (no duplicates)
    for label in labels:
        if label not in layer_labels[T]:
            node = LabelNode(label=label, depth=T)
            graph.add_node(node)
            layer_labels[T].add(label)

    # Step 2: Initialize root node at layer 0 with label 'Thing'
    root_label = 'Thing'
    root_node = LabelNode(label=root_label, depth=0)
    graph.add_node(root_node)
    layer_labels[0].add(root_label)

    # Step 3: Build layers from T-1 down to 1
    for t in range(T - 1, 0, -1):
        graph.layers[t] = []
        layer_labels[t] = set()
        nodes_in_next_layer = graph.layers[t + 1]
        label_to_node = {node.label: node for node in nodes_in_next_layer}

        # Collect hypernyms for all nodes in the next layer
        hypernym_groups: Dict[str, Set[str]] = collections.defaultdict(set)
        for node in nodes_in_next_layer:
            node_label = node.label
            synsets = wn.synsets(node_label)
            if not synsets:
                continue
            synset = synsets[0]
            depth_node = synset.min_depth()

            # Get hypernyms within depth difference ≤ depth_diff
            hypernyms = get_hypernyms_within_depth(synset, max_depth_difference=depth_diff)
            for hypernym_synset in hypernyms:
                hypernym_label = hypernym_synset.name().split('.')[0]
                depth_hypernym = hypernym_synset.min_depth()
                current_depth_diff = depth_node - depth_hypernym
                # Enforce the depth difference constraint from 'Thing'
                if (depth_hypernym - thing_depth) < t:
                    continue  # Skip if depth difference from 'Thing' is less than t
                # Ensure hypernym is not in lower layers
                lower_layer_labels = set()
                for depth in range(t + 1, T + 1):
                    lower_layer_labels.update(layer_labels[depth])
                if hypernym_label in lower_layer_labels:
                    continue
                hypernym_groups[hypernym_label].add(node_label)

        # Create parent nodes in layer t
        for hypernym_label, child_labels in hypernym_groups.items():
            if hypernym_label in layer_labels[t]:
                parent_node = next(node for node in graph.layers[t] if node.label == hypernym_label)
            else:
                parent_node = LabelNode(label=hypernym_label, depth=t)
                # Ensure the depth difference from 'Thing' is at least t
                parent_synsets = wn.synsets(hypernym_label)
                if not parent_synsets:
                    continue
                parent_synset = parent_synsets[0]
                parent_depth = parent_synset.min_depth()
                if (parent_depth - thing_depth) < t:
                    continue  # Skip if depth difference from 'Thing' is less than t
                graph.add_node(parent_node)
                layer_labels[t].add(hypernym_label)
            # Connect parent to child nodes
            for child_label in child_labels:
                child_node = label_to_node[child_label]
                if len(child_node.parents) >= max_parent:
                    continue  # Respect max_parent constraint
                graph.add_edge(parent_label=parent_node.label, child_label=child_node.label)

        # Ensure that nodes in layer t+1 have at least one parent
        for node in nodes_in_next_layer:
            if not node.parents:
                # Supplement parent nodes using hypernyms and descriptive words
                supplement_parents(node, graph, t, max_parent=max_parent, depth_diff=depth_diff, layer_labels=layer_labels, thing_depth=thing_depth)

    # Step 4: Ensure all nodes have at least one parent
    for t in range(1, T + 1):
        for node in graph.layers[t]:
            if not node.parents:
                # Connect to root node
                graph.add_edge(parent_label=root_label, child_label=node.label)

    # Step 5: After building the graph, initialize upper_labels for each node
    initialize_upper_labels(graph)

def get_hypernyms_within_depth(synset: 'Synset', max_depth_difference: int) -> Set['Synset']:
    """
    Retrieves hypernyms of a synset within a specified depth difference.
    
    Args:
        synset (Synset): The synset to retrieve hypernyms for.
        max_depth_difference (int): The maximum allowed depth difference.
    
    Returns:
        Set[Synset]: A set of hypernym synsets within the depth difference.
    """
    result = set()
    synset_depth = synset.min_depth()
    queue = [(synset, synset_depth)]
    visited = set()
    while queue:
        current_synset, current_depth = queue.pop(0)
        if current_synset in visited:
            continue
        visited.add(current_synset)
        depth_diff = synset_depth - current_depth
        if 0 < depth_diff <= max_depth_difference:
            result.add(current_synset)
        if depth_diff < max_depth_difference:
            hypernyms = current_synset.hypernyms()
            for hypernym in hypernyms:
                queue.append((hypernym, hypernym.min_depth()))
    return result

def supplement_parents(node: 'LabelNode', graph: 'MultiGranGraph', t: int, max_parent: int, depth_diff: int, layer_labels: Dict[int, Set[str]], thing_depth: int):
    """
    Supplements parent nodes for a node with fewer than max_parent parents.

    Args:
        node (LabelNode): The node to supplement parents for.
        graph (MultiGranGraph): The graph object.
        t (int): The current layer.
        max_parent (int): The maximum number of parents to supplement.
        depth_diff (int): The maximum allowed depth difference for hypernyms.
        layer_labels (Dict[int, Set[str]]): Labels present in each layer.
        thing_depth (int): The depth of 'Thing' in WordNet.
    """
    synsets = wn.synsets(node.label)
    if not synsets:
        return
    synset = synsets[0]
    depth_node = synset.min_depth()
    # Get hypernyms within depth_diff
    hypernyms = synset.hypernyms()
    for hypernym in hypernyms:
        hypernym_label = hypernym.name().split('.')[0]
        depth_hypernym = hypernym.min_depth()
        current_depth_diff = depth_node - depth_hypernym
        # Enforce the depth difference constraint from 'Thing'
        if (depth_hypernym - thing_depth) < t:
            continue  # Skip if depth difference from 'Thing' is less than t
        if current_depth_diff > depth_diff or current_depth_diff < 1:
            continue
        # Ensure hypernym is not in lower layers
        lower_layer_labels = set()
        for depth in range(t + 1, max(graph.layers.keys()) + 1):
            lower_layer_labels.update(layer_labels[depth])
        if hypernym_label in lower_layer_labels:
            continue
        # Check if parent node already exists in layer t
        if hypernym_label in layer_labels[t]:
            parent_node = next((n for n in graph.layers[t] if n.label == hypernym_label), None)
        else:
            parent_node = LabelNode(label=hypernym_label, depth=t)
            # Ensure the depth difference from 'Thing' is at least t
            parent_synsets = wn.synsets(hypernym_label)
            if not parent_synsets:
                continue
            parent_synset = parent_synsets[0]
            parent_depth = parent_synset.min_depth()
            if (parent_depth - thing_depth) < t:
                continue  # Skip if depth difference from 'Thing' is less than t
            graph.add_node(parent_node)
            layer_labels[t].add(hypernym_label)
        # Add edge from parent to child
        graph.add_edge(parent_label=parent_node.label, child_label=node.label)
        if len(node.parents) >= max_parent:
            break  # Stop if we've reached the required number of parents

def get_thing_depth() -> int:
    """
    Retrieves the minimum depth of 'Thing' in WordNet.

    Returns:
        int: The minimum depth of 'Thing'.
    """
    thing_synsets = wn.synsets('thing', pos=wn.NOUN)
    if not thing_synsets:
        return 0  # Default to 0 if 'Thing' is not found
    # Use the first synset for 'Thing'
    thing_synset = thing_synsets[0]
    thing_depth = thing_synset.min_depth()
    return thing_depth

def initialize_upper_labels(graph: 'MultiGranGraph'):
    """
    Initializes the upper_labels attribute for each node in the graph.

    Args:
        graph (MultiGranGraph): The graph whose nodes will have their upper_labels initialized.
    """
    for node in graph.nodes.values():
        node.upper_labels = {}  # Initialize the dictionary
        visited = set()
        queue = [(parent, 1) for parent in node.parents]  # Start with immediate parents, depth_diff = 1
        while queue:
            current_node, depth_diff = queue.pop(0)
            if current_node in visited:
                continue
            visited.add(current_node)
            # Add the label to upper_labels at depth_diff
            if depth_diff not in node.upper_labels:
                node.upper_labels[depth_diff] = set()
            node.upper_labels[depth_diff].add(current_node.label)
            # Enqueue parent nodes
            for parent in current_node.parents:
                queue.append((parent, depth_diff + 1))
        # Convert sets to lists
        for depth_diff in node.upper_labels:
            node.upper_labels[depth_diff] = list(node.upper_labels[depth_diff])

def extract_descriptive_words(definition: str) -> List[str]:
    """
    Extracts adjectives and nouns from a definition string.

    Args:
        definition (str): The definition text from which to extract words.

    Returns:
        List[str]: A list of adjectives and nouns.
    """
    from nltk.tokenize import word_tokenize
    from nltk import pos_tag

    # Tokenize the definition
    tokens = word_tokenize(definition)
    # POS tagging
    tagged_tokens = pos_tag(tokens)
    # Extract adjectives (JJ, JJR, JJS) and nouns (NN, NNS, NNP, NNPS)
    descriptive_words = [word for word, tag in tagged_tokens if tag.startswith('JJ') or tag.startswith('NN')]
    # Remove duplicates and non-alphanumeric characters
    descriptive_words = [re.sub(r'\W+', '', word.lower()) for word in descriptive_words]
    # Remove empty strings
    descriptive_words = [word for word in descriptive_words if word]
    return descriptive_words

def visualize_graph(graph: MultiGranGraph):
    """
    Visualizes the multi-granularity structure graph.

    Args:
        graph (MultiGranGraph): The graph to visualize.
    """
    G = nx.DiGraph()
    for parent, child in graph.edges:
        G.add_edge(parent, child)

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=0.5, iterations=100)

    # Differentiate placeholder nodes
    actual_labels = [node.label for node in graph.nodes.values() if node.label and not node.label.startswith('placeholder')]
    placeholder_labels = [node.label for node in graph.nodes.values() if node.label and node.label.startswith('placeholder')]

    nx.draw_networkx_nodes(G, pos, nodelist=actual_labels, node_color='lightblue', node_size=1500, label='Actual Labels')
    nx.draw_networkx_nodes(G, pos, nodelist=placeholder_labels, node_color='lightgray', node_size=1500, label='Placeholder Nodes')
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='->', arrowsize=20)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

    plt.legend(scatterpoints=1)
    plt.title("Multi-Granularity Structure Graph with Layered Construction and Placeholders")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # Define your labels (nodes in layer T)
    labels = ['dog', 'cat', 'apple', 'banana']

    # Initialize the graph
    graph = MultiGranGraph()

    # Build the graph with max_parent=2 and depth_diff=2
    build_graph(labels=labels, graph=graph, max_depth=3, max_parent=2, depth_diff=2)

    # Display the graph summary
    print(graph)

    # Access and display information about a specific node
    test_node = graph.get_node('dog')
    if test_node:
        print(f"Node: {test_node}")
        print(f"Parents of '{test_node.label}': {[parent.label for parent in test_node.parents]}")
        print(f"Children of '{test_node.label}': {[child.label for child in test_node.children]}")

    # Display nodes organized by layers
    for depth in sorted(graph.layers.keys()):
        nodes_in_layer = graph.layers[depth]
        aaaaa = [node.label for node in nodes_in_layer]
        print(f"Depth {depth}: {[node.label for node in nodes_in_layer]}")
