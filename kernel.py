"""
Implement an n-dimensional kernel density estimator,
where the kernel is the multivariate gaussian density.

A kernel estimator creates one kernel per data instance,
which is why we get n of them. The model is:

\hat{f}_x(x) = \frac{1}{nh} \sum_{i=1}^n
    K\left(
        \frac{\bm{x} - \bm{x}_i}{h}
    \right),

where $K$ is a symmetric density function. If we set $K$ to be a multivariate
Gaussian (which is a symmetric density), we have a finite (n-component) Gaussian
mixture model.
"""
import heapq
import logging
import argparse

import numpy as np
from scipy import stats
import scipy.spatial.distance as spsd


def gen_gmm_data(n, m, k):
    """Generate synthetic data generated from a Gaussian mixture model.

    Args:
        n (int): Number of data instances to generate.
        m (int): Number of columns (number of covariates) to generate.
        k (int): Number of true components (with equal weights).
    """
    weights = np.ones(k) * (1. / k)
    means = np.ones((k, m))
    for i in range(1, k + 1):
        means[i-1] *= (i * 2)

    data = np.zeros((n, m), dtype=np.double)
    for i in range(k):
        data += weights[i] * stats.multivariate_normal.rvs(means[i], 1.0, n)
    return data


def gaussian_kernel(X):
    """Select the bandwidth h using

        Scott's rule: h_S = n^{-1 / (m + 4)}

    which assumes the columns of X have been rescaled. We intentionally
    set the bandwidth to be lower than this to ensure overfitting. We
    use a multiplier of 1/10 for this purpose.
    """
    n, m = X.shape
    h = (n ** (-1 / (m + 4))) * (1. / 5)
    logging.info('bandwidth of %.4f selected' % h)
    cov = (h ** 2) * np.eye(m)
    comps = [(1. / n, X[i], cov) for i in range(n)]
    return comps


def hellinger(c1, c2):
    """Return Hellinger similarity between two multivariate Gaussian
    components.
    """
    w1, m1, S1 = c1
    w2, m2, S2 = c2
    p = len(m1)
    integral = (2 * np.sqrt(2 * np.pi)) ** p
    integral += np.linalg.det(S1) ** (1 / 4) * np.linalg.det(S2) ** (1 / 4)
    integral += stats.multivariate_normal.pdf(0, m1 - m2, 2 * S1 + 2 * S2)
    return w1 + w2 - 2 * np.sqrt(w1 * w2) * integral


def pairwise_dist(comps, distance=hellinger):
    """Return n x n similarity matrix for n multivariate Gaussian components.
    """
    n = len(comps)
    dm = np.zeros((n * (n - 1)) // 2, dtype=np.double)
    k = 0
    for i in xrange(0, n - 1):
        for j in xrange(i + 1, n):
            dm[k] = distance(comps[i], comps[j])
            k += 1

    return spsd.squareform(dm)


def find_merge_pair(dist):
    # temporarily set diagonal to max values
    diag_idx = np.diag_indices_from(dist)
    diag = dist[diag_idx]  # save diagonal
    dist[diag_idx] = np.finfo(dist.dtype).max
    pair = np.unravel_index(np.argmin(dist), dist.shape)
    dist[diag_idx] = diag  # restore diagonal
    return pair


def comp_merge_mom(c1, c2):
    """Merge two multivariate Gaussian components using the method of moments
    (MoM). Return the parameters for the merged component: (weight, mu, Sigma).
    """
    w1, m1, S1 = c1
    w2, m2, S2 = c2
    w = w1 + w2
    rw1 = w1 / w
    rw2 = w2 / w
    m = rw1 * m1 + rw2 * m2
    mdiff = m1 - m2
    S = rw1 * S1 + rw2 * S2 + (w1 * w2 / w) * mdiff.dot(mdiff)
    return (w, m, S)


def prim_mst(X, copy_X=True):
    """X is a square np.ndarray with edge weights of fully connected graph."""
    if copy_X:
        X = X.copy()

    n_vertices = X.shape[0]
    if n_vertices != X.shape[1]:
        raise ValueError("X needs to be square matrix of edge weights")

    # initialize with node 0:
    spanning_edges = []
    visited_vertices = [0]
    num_visited = 1

    # exclude self connections:
    diag_indices = np.arange(n_vertices)
    X[diag_indices, diag_indices] = np.inf

    while num_visited != n_vertices:
        new_edge = np.argmin(X[visited_vertices], axis=None)

        # 2d encoding of new_edge from flat, get correct indices
        new_edge = divmod(new_edge, n_vertices)
        new_edge = [visited_vertices[new_edge[0]], new_edge[1]]
        distance = X[new_edge[0], new_edge[1]]

        # add edge to tree
        spanning_edges.append([distance] + new_edge)
        visited_vertices.append(new_edge[1])

        # remove all edges inside current tree
        X[visited_vertices, new_edge[1]] = np.inf
        X[new_edge[1], visited_vertices] = np.inf
        num_visited += 1

    spanning_edges.sort(key=lambda tup: tup[2])
    return np.vstack(spanning_edges)


def merge_update(comps, mst, dist, distance=hellinger):
    """Merge most similar component, then update component
    similarities and the mst to reflect the merge. The MST contains edges of the
    form (distance, source, target). We replace comps[source] with the merged
    component and set all dist involving target to infinite. The MST edges
    involving target are replaced with source (the new component).

    This is an approximation.
    """
    n = len(comps)
    min_edge = mst[:, 0].argmin()
    close = np.nonzero(np.isclose(mst[:, 0], mst[min_edge, 0]))[0]

    # sort so targets are in descending order;
    # this allows popping components w/o disrupting list order.
    close_edges = sorted(
        mst[close].tolist(), key=lambda tup: tup[2], reverse=True)
    close_edges = [map(int, edge[1:]) for edge in close_edges]

    merged = set()  # track components already merged
    for source, target in close_edges:
        if target in merged:
            continue

        # merge the components
        comps[source] = comp_merge_mom(comps[source], comps[target])

        # update all distance measures
        for i in range(n):
            dist[source, i] = distance(comps[source], comps[i])

        # copy over new measures to columns of other comps
        for j in range(n):
            dist[j, source] = dist[source, j]

        # remove the merged (target) component
        dist = np.delete(dist, target, axis=0)
        dist = np.delete(dist, target, axis=1)

        merged.add(source)
        merged.add(target)

        comps.pop(target)
        n -= 1

    return merged, dist


def make_parser():
    parser = argparse.ArgumentParser(
        description="auto-select k for a GMM.")
    parser.add_argument(
        '-n', type=int, default=100,
        help='number of synthetic data instances')
    parser.add_argument(
        '-m', type=int, default=2,
        help='dimensionality of synthetic data (number of columns)')
    parser.add_argument(
        '-k', type=int, default=2,
        help='true number of components')
    parser.add_argument(
        '-v', '--verbose',
        type=int, default=1,
        help='adjust verbosity level of logging output')
    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    # Setup logging.
    logging.basicConfig(
        level=(logging.DEBUG if args.verbose == 2 else
               logging.INFO if args.verbose == 1 else
               logging.ERROR),
        format="[%(asctime)s]: %(message)s")

    # generate synthetic data
    data = gen_gmm_data(args.n, args.m, args.k)

    # kernel estimator
    comps = gaussian_kernel(data)

    # pairwise distance computation
    dist = pairwise_dist(comps)

    # build the minimum spanning tree (mst) from the distance measures
    mst = prim_mst(dist)

    # merge once
    merged, dist = merge_update(comps, mst, dist)
    new_mst = prim_mst(dist)
    merged2, dist2 = merge_update(comps, new_mst, dist)
