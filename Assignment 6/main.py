# from learn_decision_tree import learn_decision_tree, accuracy
from Decision_Tree import Decision_Tree as DT
import Node
import numpy as np
from pathlib import Path
from typing import Tuple

def print_decision_tree(node, depth=0, value=None):
    indent = "    " * depth  # Increase the indentation for each level of depth
    branch = " -> " if depth > 0 else ""  # Branch symbol for non-root nodes
    
    # Print the branch/value leading to this node (except for the root)
    if value is not None:
        print(f"{indent[:-4]}{branch}{value}")
    
    # For leaf nodes, print the classification value
    if node.value is not None:
        print(f"{indent}(Class: {node.value})")
    else:
        # For decision nodes, print the attribute and recurse for each child
        print(f"{indent}[Attribute: {node.attribute}]")
        for val, child_node in node.children.items():
            print_decision_tree(child_node, depth + 1, val)

def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """ Load the data for the assignment,
    Assumes that the data files is in the same folder as the script"""
    with (Path.cwd() / "train.csv").open("r") as f:
        train = np.genfromtxt(f, delimiter=",", dtype=int)
    with (Path.cwd() / "test.csv").open("r") as f:
        test = np.genfromtxt(f, delimiter=",", dtype=int)
    return train, test

def test_learn_decision_tree(train, measure):
    """ Test the learn_decision_tree function """
    return DT(train, measure)

if __name__ == "__main__":
    train, test = load_data()
    measures = ["random", "information_gain"]
    sample_size = 100
    for measure in measures:
        print(f"\nUsing measure: {measure}")
        accurries = []
        for i in range(sample_size):
            tree = test_learn_decision_tree(train, measure)
            accurries.append(tree.accuracy(test))
        print(f"Average accuracy: {round(np.mean(accurries)*100)}")