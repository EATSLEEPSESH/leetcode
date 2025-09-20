class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


class Solution(object):
    def cloneGraph(self, node):
        if not node:
            return None

        cloned = dict()

        def dfs(curr):
            if curr in cloned:
                return cloned[curr]

            copy = Node(curr.vall)
            cloned[curr] = copy
            for nei in curr.neighbors:
                copy.neighbors.append(dfs(nei))
            return copy

        return dfs(node)