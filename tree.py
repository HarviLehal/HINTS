import numpy as np


class Node:

    def __init__(self, node_id, data, level=None):
        self.node_id = node_id      # Node ID
        self.data = data            # Data
        self.level = level          # Level


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
        for i in range(self.levels):
            if i == 0:
                D.append([])
                for j in range(self.scenarios):
                    node_id = node + j
                    data = self.x[j]
                    D[i].append(Node(node_id, data, self.levels - i))
                z = int(self.scenarios/self.design[-1])
                for j in range(self.scenarios):
                    D[i][j].parent_id = self.scenarios + j % z
                    D[i][j].level = self.levels - i

            else:
                D.append([])
                for j in range(self.levels**(self.levels-i)):
                    node_id = node + j
                    data = self.x[j]
                    D[i].append(Node(node_id, data, self.levels - i))

                z = int(self.levels**(self.levels-i-1))
                for j in range(self.levels**(self.levels-i)):
                    D[i][j].parent_id = node + self.levels**(self.levels-i) + j % z
                    D[i][j].level = self.levels - i
            node += self.levels**(self.levels-i)
        D.append([])
        node_id = node
        data = self.x
        level = 0
        D[-1].append(Node(node_id, data, level))
        D = [elem for sublist in D for elem in sublist]
        self.data = D

    def leaf_nodes(self):
        leaves = [] # List of leaf nodes
        for x in self.data: # Iterate through nodes
            if x.level == self.levels:  # If node is a leaf node
                leaves.append(x)    # Add leaf node to list
        self.leaves = leaves    # Leaf nodes

    # def build_tree(self):
    #     D = []
    #     node = 0
    #     for i in range(self.levels):    # Iterate through levels
    #         if i == 0:  # If first level
    #             D.append([])
    #             for j in range(self.scenarios):                         # Add child nodes to parent node
    #                 node_id = node + j                                  # Node ID
    #                 data = self.x[j]                                    # Data
    #                 D[i].append(Node(node_id, data, self.levels - i))   # Add node to list
    #             z = int(self.scenarios/self.design[-1])                 # Number of parent nodes
    #             for j in range(self.scenarios):                         # Add parent node ids to child nodes
    #                 D[i][j].parent_id = self.scenarios + j % z          # Parent node ID
    #                 D[i][j].level = self.levels - i                     # Level

    #         else:    # If not first level
    #             D.append([])
    #             for j in range(self.levels**(self.levels-i)):           # Add child nodes to parent node
    #                 node_id = node + j                                  # Node ID
    #                 data = self.x[j]                                    # Data
    #                 D[i].append(Node(node_id, data, self.levels - i))   # Add node to list

    #             # Add parent node ids to child nodes
    #             z = int(self.levels**(self.levels-i-1))                 # Number of parent nodes
    #             for j in range(self.levels**(self.levels-i)):           # Add parent node ids to child nodes
    #                 D[i][j].parent_id = node + self.levels**(self.levels-i) + j % z     # Parent node ID
    #                 D[i][j].level = self.levels - i                     # Level

    #                 # Add child node ids to parent node
    #                 parent_node = D[i][j]                               # Parent node
    #                 parent_node.child_ids = []                          # Child node ids
    #                 for child_node in D[i-1]:                           # Iterate through child nodes
    #                     if child_node.parent_id == parent_node.node_id:     # If child node is a child of parent node
    #                         parent_node.child_ids.append(child_node.node_id)    # Add child node id to parent node
    #         node += self.levels**(self.levels-i)                        # Update node id
    #     D.append([])
    #     node_id = node
    #     data = self.x
    #     level = 0
    #     D[-1].append(Node(node_id, data, level))                        # Add root node
    #     D = [elem for sublist in D for elem in sublist]                 # Flatten list
    #     self.data = D                                                   # Tree data
