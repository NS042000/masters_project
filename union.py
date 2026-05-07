import numpy as np

class UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int32)

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]

        while self.parent[x] != x:
            parent = self.parent[x]
            self.parent[x] = root
            x = parent

        return root

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return ra

        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
            return rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
            return ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1
            return ra