"""
MCMC sampler implemented using PyMC3 for the Guassian Mixtures of Regressions
Model.

"""
import sys
import time
import logging
import argparse

import pymc3 as pm
import numpy as np
import theano
import theano.tensor as T
from scipy import stats

from mixture import gendata


def make_parser():
    parser = argparse.ArgumentParser(
        description='build gmreg model and train on synthetic data')
    parser.add_argument(
        '-nu', '--nusers',
        type=int, default=4,
        help='number of users')
    parser.add_argument(
        '-nsg', '--nsamples-to-generate',
        type=int, default=20,
        help='number of samples')
    parser.add_argument(
        '-nf', '--nfeatures',
        type=int, default=2,
        help='number of features')
    parser.add_argument(
        '-K', '--nclusters',
        type=int, default=3,
        help='number of clusters')
    parser.add_argument(
        '-s', '--init-std',
        type=float, default=0.01,
        help='standard deviation of gaussian noise for param initialization')
    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s]: %(message)s')

    nusers = args.nusers
    nsamples = args.nsamples_to_generate
    K = args.nclusters
    F = args.nfeatures
    std = args.init_std

    logging.info('number of users: %d' % nusers)
    logging.info('number of samples: %d' % nsamples)
    logging.info('number of clusters: %d' % K)
    logging.info('number of features: %d' % F)

    data, params = gendata.gen_prmix_data(nusers, nsamples, F, K)
    X = data['X']
    y = data['y']
    I = data['I']

    # Extract problem dimensions.
    N = X.shape[0]  # number of samples
    F = X.shape[1]  # number of features
    pids = I[:, 0].astype(np.int)  # primary entity ids
    uniq_pids = np.unique(pids)  # array of unique primary entity ids
    n_pe = len(uniq_pids)  # number of primary entities

    # Create model.
    logging.info('building the gmreg model')
    with pm.Model() as gmreg:
        # Init hyperparameters.
        a0 = 1
        b0 = 1
        mu0 = pm.constant(np.zeros(F))
        alpha = pm.constant(np.ones(K))
        coeff_precisions = pm.constant(1 / X.var(0))

        # Init parameters.
        # Dirichlet shape parameter, prior on indicators.
        pi = pm.Dirichlet(
            'pi', a=alpha, shape=K)

        # Ensure that each component has some samples assigned.
        # pi_min_potential = pm.Potential(
        #     'pi_min_potential', T.switch(T.min(pi) < .1, -np.inf, 0))

        # The multinomial (and by extension, the Categorical), is a symmetric
        # distribution. Using this as a prior for the indicator variables Z
        # makes the likelihood invariant under the many possible permutations of
        # the indices. This invariance is inherited in posterior inference.
        # This invariance model implies unidentifiability and induces label
        # switching during inference.

        # How to deal with label switching problem? Various approaches. (1)
        # selecting a particular permutation e.g. by means of a loss function.
        # (2) constrain parameters to break symmetry and favor single labeling.

        # On top of the label switching problem, there is also parameter
        # identifiability. In other words, different parameter values (W, beta)
        # could lead to the same likelihood value.

        # Another approach for dealing with these problems simultaneously (3) is
        # to define a non-symmetric prior suitable for the indicators.

        # We use a simple version of approach (2) here, which has been
        # identified as problematic in some settings. We simply use a potential
        # factor that enforces an ordering on the weights of the components.
        # This does not deal with the parameter identifiability issue.
        order_pi_potential = pm.Potential(
            'order_pi_potential',
            T.sum([T.switch(pi[k] - pi[k-1] < 0, -np.inf, 0)
                   for k in range(1, K)]))

        # Indicators, specifying which cluster each primary entity belongs to.
        # These are draws from Multinomial with 1 trial.
        init_pi = stats.dirichlet.rvs(alpha.eval())[0]
        test_Z = np.random.multinomial(n=1, pvals=init_pi, size=n_pe)
        as_cat = np.nonzero(test_Z)[1]
        Z = pm.Categorical(
            'Z', p=pi, shape=n_pe, testval=as_cat)

        # Ensure each component is not empty.
        # sizes = [T.eq(Z, k).nonzero()[0].shape[0] for k in range(K)]
        # nonempty_potential = pm.Potential(
        #     'comp_nonempty_potential',
        #     np.sum([T.switch(sizes[k] < 1, -np.inf, 0) for k in range(K)]))

        # Add the same sample to each cluster to avoid empty clusters.
        shared_X = X.mean(0)[None, :]
        shared_y = y.mean().reshape(1)
        X = T.concatenate((shared_X.repeat(K).reshape(K, F), X))
        y = T.concatenate((shared_y.repeat(K), y))

        # Add range(K) on to the beginning to include shared instance.
        Z_expanded = Z[pids]
        Z_with_shared = T.concatenate((range(K), Z_expanded))
        pid_idx = pm.Deterministic('pid_idx', Z_with_shared)

        # Expand user cluster indicators to each observation for each user.
        # pid_idx = pm.Deterministic('pid_idx', Z[pids])
        # X = T.cast(X, 'float64')
        # y = T.cast(y, 'float64')

        # Construct masks for each component.
        masks = [T.eq(pid_idx, k).nonzero() for k in range(K)]
        comp_sizes = [masks[k][0].shape[0] for k in range(K)]

        # # Ensure each component is not empty.
        # nonempty_potential = [
        #     pm.Potential(
        #         'comp_%d_nonempty_potential' % k,
        #         T.sum([T.switch(comp_sizes[k] < 1, -np.inf, 0)
        #                for k in range(K)]))
        # ]

        # Component regression precision parameters.
        beta = pm.Gamma(
            'beta', alpha=a0, beta=b0, shape=(K,),
            testval=np.random.gamma(a0, b0, size=K))
        # Regression coefficient matrix, with coeffs for each component.
        W = pm.MvNormal(
            'W', mu=mu0, tau=T.diag(coeff_precisions), shape=(K, F),
            testval=np.random.randn(K, F) * std)

        # The mean of the observations is the result of a regression, with
        # coefficients determined by the cluster the sample belongs to.

        # # k_values is a vector with the assigned k for each primary entity.
        #ndim, k_values = T.nonzero(Z)
        # # We can expand these to the same dimension as X using fancy indexing.
        #ks = k_values[pids]
        # Then compute the regressions to get the means.
        # means = (X * W[pid_idx]).sum(1)
        # precisions = beta[pid_idx]  # expand to match sample size

        # # Finally, the mixture likelihood.
        # points = pm.Normal(
        #     'obs', mu=means, tau=precisions, observed=y)

        # Now we have K different multivariate normal distributions.
        comps = []
        for k in range(K):
            mask_k = masks[k]

            X_k = X[mask_k]
            y_k = y[mask_k]

            n_k = comp_sizes[k]
            precision_matrix = beta[k] * T.eye(n_k)

            comp_k = pm.MvNormal(
                'comp_%d' % k,
                mu=T.dot(X_k, W[k]), tau=precision_matrix,
                observed=y_k)
            comps.append(comp_k)

    # with gmreg:
    #     step1 = pm.Metropolis(vars=[pi, beta, W])
    #     step2 = pm.ElemwiseCategoricalStep(vars=[Z], values=np.arange(K))
    #     tr = pm.sample(1000, step=[step1, step2])
