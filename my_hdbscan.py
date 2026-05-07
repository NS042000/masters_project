import numpy as np
import time
from core_dist_calculation import core_dist_calc_worker
from mrd import mrd_worker
from union import UnionFind
from boruvka_mst import boruvka
from multiprocessing import Pool, cpu_count
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import coo_matrix


class HDBSCAN:
    def __init__(self, min_samples=5, min_cluster_size=5):
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size

    def fit(self, X):
        s1 = time.perf_counter()
        cpu_s1 = time.process_time()
        n = X.shape[0]

        # 1. Core distances

        with Pool(cpu_count()) as pool:
            args = [(X, self.min_samples)]

            r = pool.map(core_dist_calc_worker, args)

            for x, y in r:
                distances = x
                indices = y

        core_dist = distances[:, -1]
        e1 = time.perf_counter()
        tt1 = e1 - s1
        print(f"Core Distances Runtime: {tt1:.4f}")

        cpu_e1 = time.process_time()
        t_cpu1 = cpu_e1 - cpu_s1
        print(f"Core Distances CPU time: {t_cpu1:.4f}")

        # 2. Mutual reachability

        s2 = time.perf_counter()
        cpu_s2 = time.process_time()

        rows, cols, vals = [], [], []

        with Pool(cpu_count()) as pool:
                args = [
                    (i, indices[i], distances[i], core_dist)
                    for i in range(n)
                ]

                results = pool.map(mrd_worker, args)

                for r, c, v in results:
                    rows.extend(r)
                    cols.extend(c)
                    vals.extend(v)


        e2 = time.perf_counter()
        tt2 = e2 - s2
        print(f"Mutual Reachability Runtime: {tt2:.4f}")

        cpu_e2 = time.process_time()
        t_cpu2 = cpu_e2 - cpu_s2
        print(f"Mutual Reachability CPU Time: {t_cpu2:.4f}")

        # 3. Minimum spanning tree

        s3 = time.perf_counter()
        cpu_s3 = time.process_time()

        graph = coo_matrix((vals, (rows, cols)), shape=(n, n))
        graph = graph.minimum(graph.T)

        # OPTIONAL: Create MST via Boruvka's Algorithm
        #edges = boruvka(n, rows, cols, vals)
        #edges.sort(key=lambda x: x[2])

        mst = minimum_spanning_tree(graph)

        rows, cols = mst.nonzero()
        weights = mst.data
        edges = list(zip(rows, cols, weights))
        edges.sort(key=lambda x: x[2])

        e3 = time.perf_counter()
        tt3 = e3 - s3
        print(f"MST Creation Runtime: {tt3:.4f}")

        cpu_e3 = time.process_time()
        t_cpu3 = cpu_e3 - cpu_s3
        print(f"MST Creation CPU Time: {t_cpu3:.4f}")

        # 4. Hierarchical clustering via union-find

        s4 = time.perf_counter()
        cpu_s4 = time.process_time()

        uf = UnionFind(n)
        cluster_members = {i: [i] for i in range(n)}
        hierarchy = []

        for i, j, d in edges:
            ri = uf.find(i)
            rj = uf.find(j)

            if ri != rj:
                lam = 1.0 / d if d != 0 else np.inf
                new_root = uf.union(ri, rj)
                merged = cluster_members[ri] + cluster_members[rj]
                cluster_members[new_root] = merged
                hierarchy.append((ri, rj, new_root, lam, len(merged)))

        e4 = time.perf_counter()
        tt4 = e4 - s4
        print(f"Hierarchical Clustering Runtime: {tt4:.4f}")

        cpu_e4 = time.process_time()
        t_cpu4 = cpu_e4 - cpu_s4
        print(f"Hierarchical Clustering CPU Time: {t_cpu4:.4f}")

        # 5. Condense the cluster hierarchy

        s5 = time.perf_counter()
        cpu_s5 = time.process_time()

        clusters = []
        cluster_id = 0
        for c in hierarchy:
            r1, r2, new_root, lam, merge_size = c
            if merge_size >= self.min_cluster_size:
                clusters.append(
                {
                    "id": cluster_id,
                    "lambda_birth": lam,
                    "points": cluster_members[new_root],
                    "size": len(cluster_members[new_root]),
                })
            cluster_id += 1

        e5 = time.perf_counter()
        tt5 = e5 - s5
        print(f"Condensed Tree Runtime: {tt5:.4f}")

        cpu_e5 = time.process_time()
        t_cpu5 = cpu_e5 - cpu_s5
        print(f"Condensed Tree CPU Time: {t_cpu5:.4f}")

        # 6. Stability calculation

        s6 = time.perf_counter()
        cpu_s6 = time.process_time()

        for c in clusters:
            c["lambda_death"] = 0

        for i in range(len(clusters) - 1):
            clusters[i]["lambda_death"] = clusters[i + 1]["lambda_birth"]

        if clusters:
            clusters[-1]["lambda_death"] = clusters[-1]["lambda_birth"]

        for c in clusters:
            c["stability"] = c["size"] * (c["lambda_death"] - c["lambda_birth"])

        e6 = time.perf_counter()
        tt6 = e6 - s6
        print(f"Stability Calculation Runtime: {tt6:.4f}")

        cpu_e6 = time.process_time()
        t_cpu6 = cpu_e6 - cpu_s6
        print(f"Stability Calculation CPU Time: {t_cpu6:.4f}")

        # 7. Select best clusters

        s7 = time.perf_counter()
        cpu_s7 = time.process_time()

        clusters = sorted(clusters, key=lambda x: x["stability"], reverse=True)
        labels = -np.ones(n, dtype=int)
        cid = 0

        for c in clusters:
            assigned = False

            for p in c["points"]:
                if labels[p] != -1:
                    assigned = True
                    break

            if not assigned:
                for p in c["points"]:
                    labels[p] = cid
                cid += 1

        self.labels_ = labels

        e7 = time.perf_counter()
        tt7 = e7 - s7
        print(f"Cluster Selection Runtime: {tt7:.4f}")

        cpu_e7 = time.process_time()
        t_cpu7 = cpu_e7 - cpu_s7
        print(f"Cluster Selection CPU Time: {t_cpu7:.4f}")

        return self