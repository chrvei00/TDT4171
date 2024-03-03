from Node import Node
import numpy as np

class Decision_Tree:
    def __init__(self, training_data, measure: str):
        self.training_data = training_data
        self.measure = measure
        self.tree = learn_decision_tree(training_data, np.arange(training_data.shape[1] - 1), None, None, None, measure)
    
    def accuracy(self, examples: np.ndarray) -> float:
        """ Calculates accuracy of tree on examples """
        correct = 0
        for example in examples:
            pred = self.tree.classify(example[:-1])
            correct += pred == example[-1]
        return correct / examples.shape[0]

def plurality_value(examples: np.ndarray) -> int:
    """Implements the PLURALITY-VALUE (Figure 19.5)"""
    labels = examples[:, -1]
    value, count = 0, 0
    for label in np.unique(labels):
        label_count = np.count_nonzero(labels == label)
        if label_count > count:
            value = label
            count = label_count
    return value

def entropy(examples):
    """Calculate the entropy of a dataset."""
    labels = examples[:, -1]  # Assuming the last column is the target attribute
    label_counts = np.unique(labels, return_counts=True)[1]
    probabilities = label_counts / label_counts.sum()
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Add a small value to avoid log(0)

def information_gain(examples, attribute):
    """Calculate the information gain of splitting on the given attribute."""
    # Initial entropy of the whole dataset
    initial_entropy = entropy(examples)

    # Values and counts of the attribute to split on
    values, counts = np.unique(examples[:, attribute], return_counts=True)
    weighted_entropy = 0

    # Calculate the weighted entropy of splitting the data based on attribute values
    for value, count in zip(values, counts):
        subset = examples[examples[:, attribute] == value]
        subset_entropy = entropy(subset)
        weighted_entropy += (count / np.sum(counts)) * subset_entropy

    # Information gain is the reduction in entropy
    return initial_entropy - weighted_entropy

def importance(attributes: np.ndarray, examples: np.ndarray, measure: str) -> int:
    """
    This function should compute the importance of each attribute and choose the one with highest importance,
    A ← argmax a ∈ attributes IMPORTANCE (a, examples) (Figure 19.5)

    Parameters:
        attributes (np.ndarray): The set of attributes from which the attribute with highest importance is to be chosen
        examples (np.ndarray): The set of examples to calculate attribute importance from
        measure (str): Measure is either "random" for calculating random importance, or "information_gain" for
                        caulculating importance from information gain (see Section 19.3.3. on page 679 in the book)

    Returns:
        (int): The index of the attribute chosen as the test

    """
    if measure == "random":
        # Randomly select an attribute
        return np.random.choice(attributes)
    elif measure == "information_gain":
        # Initialize maximum information gain and the best attribute
        max_gain = -np.inf
        best_attribute = None
        for attribute in attributes:
            # Compute the information gain for each attribute
            gain = information_gain(examples, attribute)
            if gain > max_gain:
                max_gain = gain
                best_attribute = attribute
        return best_attribute
    else:
        raise ValueError("Invalid importance measure specified.")

def learn_decision_tree(examples: np.ndarray, attributes: np.ndarray, parent_examples: np.ndarray,
                        parent: Node, branch_value: int, measure: str):
    """
    This is the decision tree learning algorithm. The pseudocode for the algorithm can be
    found in Figure 19.5 on Page 678 in the book.

    Parameters:
        examples (np.ndarray): The set data examples to consider at the current node
        attributes (np.ndarray): The set of attributes that can be chosen as the test at the current node
        parent_examples (np.ndarray): The set of examples that were used in constructing the current node’s parent.
                                        If at the root of the tree, parent_examples = None
        parent (Node): The parent node of the current node. If at the root of the tree, parent = None
        branch_value (int): The attribute value corresponding to reaching the current node from its parent.
                        If at the root of the tree, branch_value = None
        measure (str): The measure to use for the Importance-function. measure is either "random" or "information_gain"

    Returns:
        (Node): The subtree with the current node as its root
    """


    # Creates a node and links the node to its parent if the parent exists
    node = Node()
    if parent is not None:
        parent.children[branch_value] = node
        node.parent = parent

    # Check for the base cases
    if len(examples) == 0:
        node.value = plurality_value(parent_examples)
        return node
    elif np.all(examples[:, -1] == examples[0, -1]):
        node.value = examples[0, -1]
        return node
    elif len(attributes) == 0:
        node.value = plurality_value(examples)
        return node
    else:
        # Find the best attribute to split on
        best_attribute = importance(attributes, examples, measure)
        node.attribute = best_attribute
        
        # Remove the best attribute from the list
        remaining_attributes = attributes[attributes != best_attribute]

        # Split the dataset and recurse for each value of the best attribute
        for value in np.unique(examples[:, best_attribute]):
            subset = examples[examples[:, best_attribute] == value]
            if subset.size == 0:
                child_node = Node()
                child_node.value = plurality_value(examples)
                node.children[value] = child_node
            else:
                child_node = learn_decision_tree(subset, remaining_attributes, examples, node, value, measure)
                node.children[value] = child_node

    return node