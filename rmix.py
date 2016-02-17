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
from sklearn import preprocessing
from sklearn import cluster


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

    def init_params(self, X_mat, ids, init='k-means++', max_iter=100):
        """Initialize the parameters of the model using kmeans clustering on the
        feature vectors X.
        """
        N, M, p = X_mat.shape
        T = N * M
        X = X_mat.reshape(T, p)

        # centroids, codebook = \
        #     cluster.vq.kmeans2(X, self.K, iter=100, minit='points')

        kmeans = cluster.KMeans(self.K, init=init, max_iter=max_iter, n_init=1)
        kmeans.fit(X)
        centroids = kmeans.cluster_centers_
        codebook = kmeans.labels_

        self.coefficients = centroids
        grouped = codebook[ids]

        # Count number of observations assigned to each cluster for each user.
        # Assign user memberships to proportion in each cluster.
        self.memberships = np.apply_along_axis(
            lambda row: np.bincount(row, minlength=self.K), 1, grouped) / 10.

        # Then average over all these memberships to get the overall weights.
        self.weight[:] = self.memberships.mean(axis=0)

        # Find squared deviations for each predictor from the centroids of the
        # model it was clustered into. Then find the variances by summing these
        # over all the samples in each cluster.
        sqsums = np.power(X - centroids[codebook], 2)
        self.variance[:] = [sqsums[codebook == k].sum() for k in range(self.K)]

    def fit(self, X_mat, y_mat, ids, niters=None,
            init_iters=50, epsilon=0.00001, warmup=5):
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

        # We allow minor fluctuation of log-likelihood for early stopping.
        # We track the number of iterations it has increased with a counter.
        # If this goes above 1, we terminate.
        bad_iterations = 0

        # Init params and set abbreviations for use in computations.
        self.init_params(X_mat, ids, max_iter=init_iters)
        B = self.coefficients
        var = self.variance
        pi = self.weight
        H = self.memberships

        # Initialize log-likelihood trackers.
        prev_llik = -np.inf  # set to lowest possible number
        llik = np.finfo(np.inf).min  # set to next lowest

        # Useful things.
        I_M = np.eye(M)  # multiplied by variances for diagonal cov matrices.
        lliks = np.zeros((N, K))  # overwritten each iteration

        # Begin main training loop.
        if niters is None:
            niters = np.iinfo(np.int).max
        for iteration in xrange(niters):

            # Estimate B_k, \sigma_k^2, \pi_k from weighted least squares
            # solutions, using current membership probs as weights.
            # We construct P_k as an N x N matrix whose diagonal contains the
            # weights to be applied to X and Y during regression.

            # first calculate new weights.
            H_sums = H.sum(axis=0)
            pi[:] = (1. / N) * H_sums

            for k in range(K):
                P_k = np.diag(H[:, k].repeat(M, axis=0))

                # calculate B_k
                X_weighted = X.T.dot(P_k)
                p1 = np.linalg.inv(X_weighted.dot(X))
                p2 = X_weighted.dot(y)
                B[k] = p1.dot(p2)

                # A = X_weighted.dot(X)
                # b = X_weighted.dot(y)
                # chol = sp.linalg.cho_factor(A, check_finite=False, lower=True)
                # B[k] = sp.linalg.cho_solve(chol, b)

                # calculate sigma_k^2
                regress = X.dot(B[k])
                residuals = y - regress
                var[k] = residuals.T.dot(P_k).dot(residuals)

            # divide sigmas by the membership weight sums for each component
            var[:] = var / H_sums

            # Compute new membership probs using new param estimates.
            means = X_mat.dot(B.T)
            sigma = np.sqrt(var)

            for k in range(K):
                lliks[:, k] = (
                    stats.norm.logpdf(
                        y_mat.reshape(T),
                        means[:, :, k].reshape(T),
                        sigma[k]
                    ).reshape(N, M)).sum(axis=1)

            # re-normalize the membership weights
            H[:] = np.exp(lliks + np.log(pi))
            H[:] = H / H.sum(axis=1)[:, None].repeat(K, axis=1)

            # Loop to step 2 until log-likelihood stabilizes.
            # Calculate expectation of complete data log-likelihood.
            lliks *= H
            llik = (H * np.log(pi)).sum() + lliks.sum()
            logging.info('%d: log-likelihood: %.4f' % (iteration, llik))
            if np.isinf(llik):
                raise ValueError('log-likelihood has become infinite')

            # early stopping check if enabled (after iteration #5)
            if epsilon is not None and iteration >= warmup:
                diff = llik - prev_llik
                # allow small reductions which could be due to rounding error
                if diff < -0.01:
                    bad_iterations += 1
                    if bad_iterations > 2:
                        logging.info('log-likelihood increased for more'
                                     ' than two iterations')
                        # np.random.seed(np.random.randint(0, 2^30))
                        init_iters = max(1, init_iters - 10)
                        self.init_params(X_mat, ids, init='random',
                                         max_iter=init_iters)
                        B = self.coefficients
                        var = self.variance
                        pi = self.weight
                        H = self.memberships
                        bad_iterations = 0
                elif diff <= epsilon:  # and >= -0.01
                    logging.info('stopping threshold reached')
                    break
                else:  # > epsilon
                    bad_iterations = 0

            prev_llik = llik

    def cluster(self):
        """Hard clustering of users based on max of soft membership weights."""
        pass

    def log_likelihood(self):
        pass

    def predict(self, X_mat, ids):
        """Make predictions for a set of dyads using learned model."""
        N, M, p = X_mat.shape
        X = X_mat.reshape(N * M)

        # temporary, eventually need to actually compute from ids.
        idx = np.repeat(np.arange(N), M)

        y_hat = (X.dot(self.coefficients.T) * self.memberships[idx]).sum(1)
        return y_hat.reshape(N, M)


def make_parser():
    parser = argparse.ArgumentParser(
        description='train mixture of regressions on synthetic data')
    parser.add_argument(
        '-i', '--niters',
        type=int, default=None,
        help='max number of iterations to train for')
    parser.add_argument(
        '-e', '--epsilon',
        type=float, default=0.000001,
        help='early stopping threshold for training')
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
    T, p = X.shape

    # scaler = preprocessing.StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    # X_mat = X_scaled.reshape((N, M, p))  # rw = row-wise

    X_mat = X.reshape((N, M, p))
    y_mat = y.reshape((N, M))

    rmix = RMix(K)
    rmix.fit(X_mat, y_mat, ids, niters=args.niters, epsilon=args.epsilon)
