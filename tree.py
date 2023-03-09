import numpy as np
import random
from scipy.stats import multivariate_normal

class Node:

    def __init__(self, node_id, data=None, level=None, parent_id=None):
        self.node_id = node_id      # Node ID
        self.data = data            # Data
        self.level = level          # Level
        self.parent_id = parent_id  # Parent ID


class Tree:
    def __init__(self, x, levels, log_branch_factor):
        self.levels = levels                                                                # Number of Levels
        self.log_branch_factor = log_branch_factor                                          # Log Branch Factor
        self.design = np.array([1] + [2 ** self.log_branch_factor for _ in range(self.levels)])     # Design Matrix
        self.scenarios = 1 * 2 ** (self.levels * self.log_branch_factor)                    # Number of Scenarios
        self.x = np.split(x, self.scenarios)                                                # Split Data into Scenarios

    def build_tree(self):
        D = []
        node = 0
        level_start = self.scenarios
        for i in range(self.levels+1):                      # Iterate through levels
            print("level size: ", level_start)
            print("!!!!!!N O D E!!!!!!: ", node)
            if i == 0:                                      # If level is 0
                print("leaf level")
                D.append([])                                # Append empty list
                z = int(2**(self.log_branch_factor*(self.levels-i-1)))                     # Number of nodes in level
                print(z)
                for j in range(self.scenarios):             # Iterate through nodes
                    
                    print("***************")
                    node_id = node                          # Node ID
                    print("node_id: ", node_id)
                    data = self.x[j]                        # Data
                    parent_id = self.scenarios + j % z      # Parent ID
                    print("parent_id: ", parent_id)
                    level = self.levels                     # Level
                    print("level: ", level)
                    D[i].append(Node(node_id, data, level, parent_id))  # Append Node
                    node += 1                               # Increment node

            elif i == self.levels:                          # If level is last level
                print("root level")
                D.append([])                                # Append empty list
                print("***************")
                node_id = node                      # Node ID
                print("node_id: ", node_id)
                data = self.x                        # Data
                parent_id = None                        # Parent ID
                print("parent_id: ", parent_id)
                level = 0                               # Level
                print("level: ", level)
                D[i].append(Node(node_id = node_id, level = level,parent_id = parent_id))  # Append Node
                node += 1                               # Increment node

            else:
                print("intermediate level")
                D.append([])
                z = int(2**(self.log_branch_factor*(self.levels-i-1)))                     # Number of nodes in level
                print(z)
                for j in range(2**(self.log_branch_factor*(self.levels-i))):                             # Iterate through nodes
                    print("***************")
                    node_id = node                                      # Node ID
                    print("node_id: ", node_id)
                    data = self.x[j]                                        # Data
                    parent_id = level_start + j % z  # Parent ID
                    print("parent_id: ", parent_id)
                    level = self.levels - i                                 # Level
                    print("level: ", level)
                    D[i].append(Node(node_id=node_id, level=level, parent_id=parent_id))      # Append Node
                    node += 1                                               # Increment node
                    
            level_start = node + 2**(self.log_branch_factor*(self.levels-i-1))                     # Number of nodes in level
        D = [elem for sublist in D for elem in sublist]                     # Flatten list
        self.data_tree = D                                                  # Data Tree
        leaves = []                     # List of leaf nodes
        for x in self.data_tree:             # Iterate through nodes
            if x.level == self.levels:  # If node is a leaf node
                leaves.append(x)        # Add leaf node to list
        for i in self.data_tree:
            if i.data is None:
                i.data = self.data_merge(self.child(i))
        self.leaves = leaves            # Leaf nodes

    def get_node(self, node_id):
        for x in self.data_tree:
            if x.node_id == node_id:
                return x

    def rand_leaf_selection(self):              # Random Leaf Selection
        rand_leaf = random.choice(self.leaves)
        return rand_leaf

    def level_set(self, level):                 # Level Set
        level_set = []
        for x in self.data_tree:
            if x.level == level:
                level_set.append(x)
        return level_set

    def common_parent(self, node):        # Common Parent Node Set
        common_parent_set = []
        level_set = self.level_set(node.level)  # Level Set
        for x in level_set:
            if x.parent_id == node.parent_id:
                common_parent_set.append(x)
        return common_parent_set

    def parent(self, node):
        level_set = self.level_set(node.level - 1)  # Level Set
        for x in level_set:           # ADD SOMETHING TO FACTOR LEVELS IN
            if x.node_id == node.parent_id:
                return x

    def data_merge(self, node_set):
        merge_set=[]
        for i in node_set:
            merge_set.append(i.data)
        merge_set = np.array(merge_set)
        merge_set = np.squeeze(merge_set)
        return merge_set

    def path(self, node):
        path = []
        print(node.level)
        print(node.__dict__)
        level = node.level
        print(level)
        while level != 0:
            node = self.parent(node)
            path.append(node)
            level = node.level
        return path

    def child(self, node):
        child_set = []
        level_set = self.level_set(node.level + 1)  # Level Set
        for x in level_set:           # ADD SOMETHING TO FACTOR LEVELS IN
            if x.parent_id == node.node_id:
                child_set.append(x)
        return child_set

x = multivariate_normal.rvs(0, 1, 1024)
tree = Tree(x, 5, 2)
tree.build_tree()
z=[]
for i in tree.data_tree:
    z.append(i.__dict__)