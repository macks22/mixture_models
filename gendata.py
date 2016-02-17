"""
Synthetic data generation functions.

"""
import logging
import numpy as np


def gen_3cluster_mixture(t=4, M=10):
    """Sample t trajectories, M observations each, from three clusters
    (polynomials). Let the first column of X be all 1s for the global intercept,
    the second column be the time of the observation in the trajectory, and the
    third column be some independent feature.

    Args:
        N (int): number of trajectories per component
        M (int): number of time steps per trajectory
    """
    f1 = lambda x: 120 + 4 * x
    f2 = lambda x: 10 + 2 * x + 0.1 * x ** 2
    f3 = lambda x: 250 - 0.75 * x

    K = 3      # number of components
    N = t * K  # total number of trajectories
    O = N * M  # total number of observations

    samples = np.zeros((N, M))
    xs = np.random.normal(1.0, 1.0, (N, M))

    I = np.zeros((O, 1))
    I_idx = 0
    for i, model in enumerate([f1, f2, f3]):
        for traj in range(t):
            idx = i * t + traj  # stride + step
            for obs in range(M):
                samples[idx, obs] = model(xs[idx, obs])
                I[I_idx] = traj
                I_idx += 1

    X = np.zeros((O, 4))
    y = np.zeros(O)
    ids = np.arange(O).reshape(N, M)

    X[:, 0] = 1  # intercept term
    idx = 0
    for (traj, obs), y_sample in np.ndenumerate(samples):
        X[idx, 1] = obs
        X[idx, 2] = xs[traj, obs]
        X[idx, 3] = xs[traj, obs] ** 2
        y[idx] = y_sample
        idx += 1

    return X, y, I, ids


def gen_prmix_data(nusers, nsamples, F, K):
    """Generate hyperparameters, parameters, and data for the Personalized
    Mixture of Gaussian Regressions model.

    Args:
        nusers (int): Number of distinct users.
        nsamples (int): Total number of samples to generate.
        F (int): Number of features for feature vectors.
        K (int): Number of clusters.
    Return:
        data, params (tuple of dicts):
            `data` contains X, y, and I.
                X (nsamples x F): the feature vectors for each sample.
                y (nsamples): the target variables for each sample.
                I (nsamples): the user ID for each sample.
            `params` contains pi, Z, beta, and W.
                pi (K): component weights.
                Z (nusers): categorical indicator variables for each user.
                beta (K): precision parameters for each component.
                W (K x F): regression coefficients for each component.
    """

    # Hyperparameters.
    alpha = np.ones(K)
    a0 = 1
    b0 = 1
    mu0 = np.zeros(F)
    coeff_variances = np.ones(F)

    # Parameters.
    pi = np.random.dirichlet(alpha)
    Z = np.random.multinomial(n=1, pvals=pi, size=nusers)
    Z_as_cat = np.nonzero(Z)[1]  # the nonzero column is the assigned cluster
    logging.info('assigned clusters: %s' % str(Z_as_cat))

    beta = np.random.gamma(a0, b0, K)
    sigma_sq = 1. / beta
    W = np.random.multivariate_normal(
            mean=mu0, cov=np.diag(coeff_variances), size=K)

    # Now generate samples according to which cluster the user belongs to.
    I = np.ndarray((nsamples, 1), dtype=np.int32)
    y = np.ndarray((nsamples,))

    # Randomly generate features, uniform [0, 10] + standard gaussian noise
    X = (np.random.uniform(1, 10, size=(nsamples, F))
            + np.random.randn(nsamples, F))

    # Randomly select user to sample observation for, for all samples.
    pids = np.arange(nusers)
    I[:nusers, 0] = pids  # make sure each user gets at least one observation
    rem = nsamples - nusers

    I[nusers:, 0] = np.random.choice(pids, replace=True, size=rem)
    Z_idx = Z_as_cat[I[:, 0]]
    Ws = W[Z_idx]
    means = (X * Ws).sum(1)
    sds = np.sqrt(sigma_sq[Z_idx])
    y[:] = np.random.normal(means, sds)

    data = {
        'X': X,
        'y': y,
        'I': I
    }

    params = {
        'pi': pi,
        'Z': Z_as_cat,
        'beta': beta,
        'sigma': sigma_sq,
        'W': W
    }

    return data, params
