class Node:
    def __init__(self, state, parent=None, action=None, cost=0, f=0, g=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        self.f = 0;
        self.g = 0;

        if parent is None: #root node
            self.depth = 0
        else:
            self.depth = parent.depth + 1

    def __hash__(self):
        return hash(self.state)

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state
    
    def __gt__(self, other): # this will be helpful when comparing nodes in the priority queue
        return isinstance(other, Node) and self.f > other.f
