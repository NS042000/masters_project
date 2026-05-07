
def mrd_worker(args):
    i, indices, dists, core_dist = args

    rows = []
    cols = []
    vals = []

    for j_idx, j in enumerate(indices):
        if i == j:
            continue

        d = dists[j_idx]
        mreach = max(core_dist[i], core_dist[j], d)

        rows.append(i)
        cols.append(j)
        vals.append(mreach)

    return rows, cols, vals


