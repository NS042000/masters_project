from sklearn.neighbors import NearestNeighbors

def core_dist_calc_worker(args):
    X, min_samples = args
    nbrs = NearestNeighbors(n_neighbors=min_samples, algorithm="kd_tree")
    nbrs.fit(X)

    return nbrs.kneighbors(X)