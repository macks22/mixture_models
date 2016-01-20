"""
Implement mixture of regressions model from the paper:

    Trajectory Clustering with Mixtures of Regression Models
    Scott Gaffney and Padhraic Smyth

"""
import sys
import logging
import argparse

import numpy as np
from scipy import stats

MODEL = {}


def gen_data(t=4, M=10):
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

    for i, model in enumerate([f1, f2, f3]):
        for traj in range(t):
            idx = i * t + traj  # stride + step
            for obs in range(M):
                samples[idx, obs] = model(xs[idx, obs])

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

    return X, y, ids


class RMix(object):
    """Regression Mixture for Clustering Trajectories."""

    def __init__(self, K):
        self.K = K

        # Set parameters to initial values where possible.
        # Singular naming denotes vector-valued; plural naming denotes matrix.
        self.weight = np.ndarray((K,))    # 1 x K
        self.variance = np.ndarray((K,))  # 1 x K
        self.coefficients = None          # K x p  (p unknown)
        self.memberships = None           # N x K  (N unknown)

    def fit(self, X_mat, y_mat, niters=np.iinfo(np.int).max, epsilon=0.00001,
            warmup=5):
        """Fit the model to the given data.

        The Regression Mixture (RMix) model clusters users based on the
        observations associated with them. The parameters learned for each
        cluster include the:

            mixing weights: $\pi_k$,
            regression coefficients: $\beta_k$, and
            variance terms: $\sigma_k^2$.

        The posterior mixing weights ($H_{n,k}$) are also learned -- one for
        each user for each cluster (N * K total). These are soft membership
        weights that constitute the clustering obtained by the model fit.

        Args:
            X_mat (np.ndarray[ndim=3, dtype=np.float]):
                Matrix of feature vectors for user-item dyads. These should be
                arranged in an N x M x p sparse 3D matrix (tensor).
            y_mat (np.ndarray[ndim=2, dtype=np.float]):
                Matrix of target features for user-item dyads. These should be
                arranged in an N x M sparse matrix.
            niters (int):
                Maximum number of iterations to train the model for. The model
                is trained until the stopping threshold is reached by default.
            epsilon (float):
                The stopping threshold used for early stopping. Set this to None
                if early stopping is not desired (be sure to set niters in this
                case).
            warmup (int):
                Number of warmup iterations. The early stopping check will not
                be performed during the first `warmup` iterations.
        """
        N, M, p = X_mat.shape
        T = N * M
        K = self.K

        X = X_mat.reshape(T, p)
        y = y_mat.reshape(T)

        np.seterr(divide='raise')  # raise divide by 0 errors
        eps = np.finfo(float).eps  # add to denominators which may go to 0

        # We allow minor fluctuation of log-likelihood for early stopping.
        # We track the number of iterations it has increased with a counter.
        # If this goes above 1, we terminate.
        bad_iterations = 0

        # Init params.
        B = np.ndarray((K, p))
        var = np.ndarray(K)
        pi = np.ndarray(K)

        # Randomly init the N * K posterior membership probs H_{n,k}.
        H = np.random.uniform(0, 1, (N, K))
        H = H / H.sum(axis=1)[:, None].repeat(K, axis=1)

        # Set human-readable aliases for parameters.
        self.memberships = H
        self.coefficients = B
        self.weight = pi
        self.variance = var

        # Initialize log-likelihood trackers.
        prev_llik = -np.inf  # set to lowest possible number
        llik = np.finfo(np.inf).min  # set to next lowest

        # Useful things.
        I_M = np.eye(M)  # multiplied by variances for diagonal cov matrices.
        lliks = np.zeros((N, K))  # overwritten each iteration

        # Begin main training loop.
        for iteration in xrange(niters):

            # Estimate B_k, \sigma_k^2, \pi_k from weighted least squares
            # solutions, using current membership probs as weights.
            # We construct P_k as an N x N matrix whose diagonal contains the
            # weights to be applied to X and Y during regression.

            # first calculate new weights.
            H_sums = H.sum(axis=0)
            pi = (1. / N) * H_sums

            for k in range(K):
                P_k = np.diag(H[:, k].repeat(M, axis=0))

                # calculate B_k
                X_weighted = X.T.dot(P_k)
                p1 = np.linalg.inv(X_weighted.dot(X))
                p2 = X_weighted.dot(y)
                B[k] = p1.dot(p2)

                # calculate sigma_k^2
                regress = X.dot(B[k])
                residuals = y - regress
                var[k] = residuals.T.dot(P_k).dot(residuals)

            # divide sigmas by the membership weight sums for each component
            var = var / H_sums

            # params['B'].append(B)
            # params['var'].append(var)
            # params['w'].append(weight)

            # Compute new membership probs using new param estimates.
            means = X_mat.dot(B.T)
            sigma = np.sqrt(var)
            # for n in range(N):
            #     y_n = y_mat[n]
            #     for k in range(K):
            #         H[n, k] *= stats.multivariate_normal.pdf(
            #             y_n, means[n, :, k], I_M * var[k])

            for k in range(K):
                lliks[:, k] = (
                    stats.norm.logpdf(
                        y_mat.reshape(T),
                        means[:, :, k].reshape(T),
                        sigma[k]
                    ).reshape(N, M)).sum(axis=1)

            # re-normalize the membership weights
            H = np.exp(lliks + np.log(pi))
            H = H / H.sum(axis=1)[:, None].repeat(K, axis=1)

            # Loop to step 2 until log-likelihood stabilizes.
            # Calculate expectation of complete data log-likelihood.
            llik = (H * np.log(pi)).sum()

            # means = X_mat.dot(B.T)
            # for n in range(N):
            #     y_n = y_mat[n]
            #     for k in range(K):
            #         lliks[n, k] = stats.multivariate_normal.logpdf(
            #             y_n, means[n, :, k], I_M * var[k])

            # Re-use sigma and means from membership calculations.
            # for k in range(K):
            #     lliks[:, k] = (
            #         stats.norm.logpdf(
            #             y_mat.reshape(T),
            #             means[:, :, k].reshape(T),
            #             sigma[k]
            #         ).reshape(N, M)).sum(1)

            lliks *= H
            llik += lliks.sum()
            logging.info('%d: log-likelihood: %.4f' % (iteration, llik))
            if np.isinf(llik):
                raise ValueError('log-likelihood has become infinite')

            # early stopping check if enabled (after iteration #5)
            if epsilon is not None and iteration >= warmup:
                diff = llik - prev_llik
                if diff == 0:  # done
                    logging.info('stopping threshold reached')
                    break
                elif diff <= epsilon:
                    # allow small reductions which could be due to rounding error
                    if diff < -0.01:
                        bad_iterations += 1
                        if bad_iterations > 2:
                            logging.info('log-likelihood increased for more'
                                         ' than two iteration')
                            break
                else:
                    bad_iterations = 0
                    prev_llik = llik

    def cluster(self):
        """Hard clustering of users based on max of soft membership weights."""
        pass

    def log_likelihood(self):
        pass

    def predict(self, X):
        """Make predictions for a set of dyads using learned model."""
        pass


def make_parser():
    parser = argparse.ArgumentParser(
        description='train mixture of regressions on synthetic data')
    parser.add_argument(
        '-i', '--niters',
        type=int, default=None,
        help='max number of iterations to train for')
    parser.add_argument(
        '-e', '--epsilon',
        type=float, default=None,
        help='early stopping threshold for training; disabled by default')
    parser.add_argument(
        '-t', '--ntraj', type=int, default=4,
        help='number of trajectories per component')
    parser.add_argument(
        '-M', type=int, default=10,
        help='number of time steps per trajectory')
    parser.add_argument(
        '-v', '--verbose',
        type=int, default=1,
        help='adjust verbosity of logging output')
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

    K = 3               # number of components
    N = args.ntraj * K  # number of users
    M = args.M          # number of items

    # X and y are (entity x time x predictors).
    X, y, ids = gen_data(args.ntraj, M)
    _, p = X.shape
    X_rw = X.reshape((N, M, p))  # rw = row-wise
    y_rw = y.reshape((N, M))

    rmix = RMix(K)
    rmix.fit(X_rw, y_rw, args.niters, args.epsilon)
