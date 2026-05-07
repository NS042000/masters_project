import numpy as np

def find_roots(parent):

    #Vectorized find with path compression (iterative).

    roots = parent.copy()
    while True:
        new_roots = parent[roots]
        if np.array_equal(new_roots, roots):
            break
        roots = new_roots
    return roots

def boruvka(n, rows, cols, vals):
    edges = np.column_stack((rows, cols, vals))

    mask = edges[:, 0] < edges[:, 1]
    edges = edges[mask]

    parent = np.arange(n)
    rank = np.zeros(n, dtype=int)

    mst_edges = []
    num_components = n

    # Find roots for all edges
    while num_components > 1:
        u = edges[:, 0].astype(int)
        v = edges[:, 1].astype(int)
        w = edges[:, 2]

        ru = find_roots(parent)[u]
        rv = find_roots(parent)[v]

        valid = ru != rv
        if not np.any(valid):
            break

        ru = ru[valid]
        rv = rv[valid]
        u = u[valid]
        v = v[valid]
        w = w[valid]

        # Find cheapest outgoing edge per component
        cheapest_weight = np.full(n, np.inf)
        cheapest_edge_idx = np.full(n, -1)

        np.minimum.at(cheapest_weight, ru, w)
        mask_ru = (cheapest_weight[ru] == w)
        cheapest_edge_idx[ru[mask_ru]] = np.where(valid)[0][mask_ru]

        np.minimum.at(cheapest_weight, rv, w)
        mask_rv = (cheapest_weight[rv] == w)
        cheapest_edge_idx[rv[mask_rv]] = np.where(valid)[0][mask_rv]

        chosen = np.unique(cheapest_edge_idx[cheapest_edge_idx != -1])

        # Union
        for idx in chosen:
            u0, v0, w0 = edges[idx]
            u0 = int(u0)
            v0 = int(v0)

            # Find roots again
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            ru0 = find(u0)
            rv0 = find(v0)

            if ru0 == rv0:
                continue

            # Union by rank
            if rank[ru0] < rank[rv0]:
                parent[ru0] = rv0
            elif rank[ru0] > rank[rv0]:
                parent[rv0] = ru0
            else:
                parent[rv0] = ru0
                rank[ru0] += 1

            mst_edges.append((u0, v0, w0))
            num_components -= 1

        # Shrink edge set
        ru_all = find_roots(parent)[edges[:, 0].astype(int)]
        rv_all = find_roots(parent)[edges[:, 1].astype(int)]
        edges = edges[ru_all != rv_all]

    return mst_edges