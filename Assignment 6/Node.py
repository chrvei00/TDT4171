class Node:
    """ Node class used to build the decision tree"""
    def __init__(self):
        self.children = {}
        self.parent = None
        self.attribute = None
        self.value = None

    def classify(self, example):
        try:
            if self.value is not None:
                return self.value
            return self.children[example[self.attribute]].classify(example)
        except KeyError:
            return self.value