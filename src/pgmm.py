"""
Profiling GMM + EM Implementation.

This is a Gaussian Mixture Model with per-entity membership weights. In other
words, instead of K global membership weights, one per each of the K components,
we have E X K components, where E is the number of entities. These entities
could be users, items, seasons, etc. The idea is that each entity is described
to different extents by each of the componenets. So the shared components are
combined in different ways to describe each entity.

"""
import logging
import argparse

import numpy as np
import scipy as sp
import scipy.cluster.vq as vq
from scipy import stats


def gen_igmm(N, E, K):
    ids = np.arange(E)
    eids = np.random.choice(ids, replace=True, size=N)

    # TODO: generate from multivariate normal distributions rather than 3 norms
    X = np.vstack([
        np.random.normal(m * 3.0, 1.0, N)
        for m in range(K)
    ]).T

    # Now we combine the components in different proportions for each entity.
    props = abs(np.random.randn(E, K))
    props = props / props.sum(axis=1)[:, np.newaxis]
    y = X * props[eids]

    return eids, X, y


def make_parser():
    parser = argparse.ArgumentParser(
        description='Gaussian Mixture Model learned via EM')
    parser.add_argument(
        '-k', '--nclusters',
        type=int, default=0,
        help='number of clusters')
    parser.add_argument(
        '-ns', '--nsteps',
        type=int, default=100,
        help='max number of EM steps to perform')
    parser.add_argument(
        '-m', '--init_method',
        choices=('random', 'kmeans'),
        help='parameter initialization method')
    parser.add_argument(
        '-e', '--epsilon',
        type=float, default=0.0001,
        help='minimum improvement required per step; otherwise learning stops')
    parser.add_argument(
        '-v', '--verbose',
        type=int, choices=(0, 1, 2), default=0,
        help='enable verbose logging output')
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

    # Randomly generate some data.
    N = 200
    E = 20
    K = 3
    eids, X, y = gen_igmm(N, E, K)

    # Assign values for model arguments.
    N, nf = X.shape  # number of records, features
    ns = args.nsteps
    method = args.init_method
    eps = args.epsilon

    K = args.nclusters
    if K == 0:
        K = nf

    # Initialize model parameters.
    model = {}
    if method == 'warm':  # not currently implemented; meant to load from file
        mu = model['mu']
        sigma = model['sigma']
        pi = model['pi']
    elif method == 'random':
        mu = np.random.randn(K, nf)
        sigma = np.ndarray((K, nf, nf))
        for k in xrange(K):
            sigma[k] = np.eye(nf)

        pi = abs(np.random.randn(E, K))
        pi = pi / pi.sum(axis=1)[:, np.newaxis]  # normalize
    else:  # kmeans
        centroids, labels = vq.kmeans2(X, K, minit='points', iter=100)
        clusters = [X[labels == k] for k in xrange(K)]
        mu = np.array([cluster.mean(axis=0) for cluster in clusters])
        sigma = np.array([np.cov(cluster, rowvar=0) for cluster in clusters])

        counts = np.array([len(cluster) for cluster in clusters])
        props = np.ones(K, dtype='double') / counts
        props = props / props.sum()
        pi = np.ndarray((E, K))
        for e in xrange(E):
            pi[e] = props

    model = {
        'mu': mu,
        'sigma': sigma,
        'pi': pi
    }

    # Find entity counts to use during updating
    n_entities = np.ndarray(E)
    emask = []
    for e in xrange(E):
        mask = eids == e
        n_entities[e] = mask.sum()
        emask.append(np.nonzero(mask)[0])

    prev_ll = -np.inf
    resp = np.ndarray((N, K))
    for step in xrange(ns):
        logging.info('step # %d' % step)

        # E-step. Calculate responsibilities using current param values.
        for n in xrange(N):
            X_n = X[n]
            eid = eids[n]
            for k in xrange(K):
                p = stats.multivariate_normal.pdf(X_n, mu[k], sigma[k],
                                                  allow_singular=True)
                resp[n,k] = pi[eid,k] * p

            resp[n] = resp[n] / resp[n].sum()  # normalize

        # M-step. Update params using current responsibility values.
        N_ = resp.sum(axis=0)
        for k in xrange(K):
            N_k = N_[k]
            mu[k] = np.array([resp[n,k] * X[n] for n in xrange(N)])\
                      .sum(axis=0) / N_k

            for n in xrange(N):
                tmp = X[n] - mu[k]
                tmp = tmp[:, np.newaxis].dot(tmp[np.newaxis, :])
                sigma[k] += resp[n,k] * tmp
            sigma[k] = sigma[k] / N_k

            for e in xrange(E):
                mask = emask[e]
                pi[e,:] = resp[mask,:].sum(axis=0)

        # Calculate log-likelihood.
        ll = 0
        for n in xrange(N):
            X_n = X[n]
            eid = eids[n]
            k_lik = 0
            for k in xrange(K):
                p = stats.multivariate_normal.pdf(X_n, mu[k], sigma[k],
                                                  allow_singular=True)
                k_lik += pi[eid,k] * p

            ll += np.log(k_lik)
        logging.info('LL:\t%.4f' % ll)

        if prev_ll - ll > -eps:  # negative numbers
            logging.info('stopping threshold reached')
            break
        else:
            prev_ll = ll
