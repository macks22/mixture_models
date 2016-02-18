"""
MCMC sampler implemented using collapsed Gibbs sampling for the Guassian
Mixtures of Regressions Model.

"""
import sys
import time
import logging
import argparse

import numpy as np
import scipy as sp
from scipy import stats

from rmix_pymc import gen_data, make_parser


def make_gibbs_parser():
    parser = make_parser()
    parser.add_argument(
        '-b', '--burn-in',
        type=int, default=50,
        help='number of samples to discard as burn-in')
    parser.add_argument(
        '-t', '--thin-step',
        type=int, default=2,
        help='what step size to use for thinning samples')
    parser.add_argument(
        '-ns', '--nsamples',
        type=int, default=500,
        help='number of Gibbs samples to simulate')
    return parser


if __name__ == "__main__":

    # SETUP STAGE
    # =========================================================================
    parser = make_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s]: %(message)s')

    nusers = args.nusers
    gen_nsamples = args.nsamples_to_generate
    K = args.nclusters
    F = args.nfeatures
    std = args.init_std

    logging.info('number of users: %d' % nusers)
    logging.info('number of samples: %d' % nsamples)
    logging.info('number of clusters: %d' % K)
    logging.info('number of features: %d' % F)

    data, params = gen_data(nusers, gen_nsamples, F, K)
    X = data['X']
    y = data['y']
    I = data['I']

    # Extract problem dimensions.
    N = X.shape[0]  # number of samples
    F = X.shape[1]  # number of features
    pids = I[:, 0].astype(np.int)  # primary entity ids
    uniq_pids = np.unique(pids)  # array of unique primary entity ids
    n_pe = len(uniq_pids)  # number of primary entities


    # MODEL BUILDING STAGE
    # =========================================================================

    # Initialize hyperparameters
    alpha = np.ones(K)  # Dirichlet shape parameter
    a0 = 1              # Gamma shape parameter
    b0 = 1              # Gamma rate parameter
    mu0 = np.zeros(F)   # MvNormal prior mean
    V = np.eye(F)       # MvNormal cov param

    # Set up space for posterior hyperparameters.
    alpha_post = np.zeros((nsamples,))
    a0_post = np.ndarray((nsamples, K))
    b0_post = np.ndarray((nsamples, K))
    V_post = np.ndarray((nsamples, K, F, F))
    mu0_post = np.zeros((nsamples, K, F))

    # Set up space for posterior parameters.
    pi_post = np.zeros((nsamples, K))
    sigma_post = np.zeros((nsamples, K))
    W_post = np.zeros((nsamples, K, F))
    Z_post = np.zeros((nsamples, n_pe, K))

    # Set up space for sufficient stats.
    n_k = np.ndarray((nsamples, K))
    X_k_ssq = np.ndarray((nsamples, K, F, F))
    y_k_ssq = np.ndarray((nsamples, K))
    Xy_k = np.ndarray((nsamples, K, F))

    # Initialize parameters
    pi_post[0] = np.random.dirichlet(alpha)  # K-dimensional
    Z_post[0] = np.random.multinomial(1, pi, n_pe)  # n_pe x K
    Z_cat = Z.nonzero()[1]  # convert to categorical
    Z_idx = Z_cat[pids]

    sigma_post[0] = np.random.gamma(a0, b0, K)  # K-dimensional
    for k in range(K):
        W_post[0, k] = np.random.multivariate_normal(mu0, sigma_post[0, k] * V)

    # INFERENCE STAGE
    # =========================================================================

    # Main training loop
    for i in range(1, nsamples + 1):

        # Draw alpha* | Z
        alpha_post[i] = alpha + Z.sum(0)

        # Draw pi | alpha*
        pi_post[i] = np.random.dirichlet(alpha_post[i])

        for k in range(K):
            mask_k = (Z_idx == k)
            X_k = X[mask_k]
            y_k = y[mask_k]

            # Calculate sufficient stats.
            n_k[i,k] = X_k.shape[0]
            X_k_ssq[i,k] = X_k.T.dot(X_k)
            y_k_ssq[i,k] = y_k.T.dot(y_k)
            Xy_k[i,k] = X_k.T.dot(y_k)

            # Compute mu0_post and V_post
            V_inv = np.linalg.inv(V)
            V_post[i,k] = np.linalg.inv(V_inv + X_k_ssq[i,k])

            term2 = V_inv.dot(mu0) + Xy_k[i,k]
            mu0_post[i,k] = mu0_p = V_post[i,k].dot(term2)

            # Draw alpha0*, beta0* | data
            # TODO: optimize based on mu0 being zeros and V being identity.
            a0_post[i,k] = a0 + (n_k[i,k] / 2.)

            V_post_inv = np.linalg.inv(V_post[i,k])
            b0_post[i,k] = b0 + 0.5 * (
                mu0.T.dot(V_inv).dot(mu0)
                + y_k_ssq[i,k]
                - mu0_p.T.dot(V_post_inv).dot(mu0_p))

            # Draw sigma_k | alpha0, beta0
            sigma_post[i,k] = np.random.gamma(a0_post[i,k], b0_post[i,k])

            # Draw W_k | mu0, sigma_k
            W_post[i,k] = np.random.multivariate_normal(
                mu0_post[i,k], sigma_post[i,k] * V)

        # Draw Z_n | pi, Z_{n-}
        # for each user:


    # DIAGNOSTICS STAGE
    # =========================================================================

    # Traceplots

    # Autocorrelation plots

    # Compare induced hard clustering to true clustering

    # Posterior predictive checks
