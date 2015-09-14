import sys
import logging
import argparse

import numpy as np
import scipy as sp
import pandas as pd
import scipy.cluster.vq as vq
from scipy import stats


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
    N = 1000
    a = np.random.normal(0.0, 1.0, N)
    b = np.random.normal(3.0, 1.0, N)
    c = np.random.normal(6.0, 1.0, N)
    y = a + b + c
    data = np.vstack([y, a, b, c]).T

    # x = np.arange(-10, 30)
    # y = x + np.random.randn(40)
    # data = np.vstack([x, y]).T

    # Assign values for model arguments.
    N, nf = data.shape  # number of records, features
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

        pi = abs(np.random.randn(K))
        pi = pi / pi.sum()  # normalize to enfore sum-to-1 constraint.
    else:  # kmeans
        centroids, labels = vq.kmeans2(data, K, minit='points', iter=100)
        clusters = [data[labels == k] for k in xrange(K)]
        mu = np.array([cluster.mean(axis=0) for cluster in clusters])
        sigma = np.array([np.cov(cluster, rowvar=0) for cluster in clusters])
        counts = np.array([len(cluster) for cluster in clusters])
        pi = np.ones(K, dtype='double') / counts
        pi = pi / pi.sum()

    model = {
        'mu': mu,
        'sigma': sigma,
        'pi': pi
    }

    prev_ll = -np.inf
    resp = np.ndarray((N, K))
    for step in xrange(ns):
        logging.info('step # %d' % step)

        # E-step. Calculate responsibilities using current param values.
        for n in xrange(N):
            X_n = data[n]
            for k in xrange(K):
                p = stats.multivariate_normal.pdf(X_n, mu[k], sigma[k],
                                                  allow_singular=True)
                resp[n,k] = pi[k] * p

            resp[n] = resp[n] / resp[n].sum()  # normalize

        # M-step. Update params using current responsibility values.
        N_ = resp.sum(axis=0)
        for k in xrange(K):
            N_k = N_[k]
            pi[k] = N_k / N

            mu[k] = np.array([resp[n,k] * data[n] for n in xrange(N)])\
                      .sum(axis=0) / N_k

            for n in xrange(N):
                tmp = data[n] - mu[k]
                tmp = tmp[:, np.newaxis].dot(tmp[np.newaxis, :])
                sigma[k] += resp[n,k] * tmp
            sigma[k] = sigma[k] / N_k

        # Calculate log-likelihood.
        ll = 0
        for n in xrange(N):
            X_n = data[n]
            k_lik = 0
            for k in xrange(K):
                p = stats.multivariate_normal.pdf(X_n, mu[k], sigma[k],
                                                  allow_singular=True)
                k_lik += pi[k] * p

            ll += np.log(k_lik)
        logging.info('LL:\t%.4f' % ll)

        if prev_ll - ll > -eps:  # negative numbers
            logging.info('stopping threshold reached')
            break
        else:
            prev_ll = ll
