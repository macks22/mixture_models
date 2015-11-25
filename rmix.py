"""
Implement mixture of regressions model from the paper:

    Trajectory Clustering with Mixtures of Regression Models
    Scott Gaffney and Padhraic Smyth

"""
import logging
import argparse

import numpy as np
from scipy import stats


def gen_data(l=4, n=10):
    """Sample 4 trajectories, 10 observations each, from three clusters
    (polynomials). Let the first column of X be all 1s for the global intercept,
    the second column be the time of the observation in the trajectory, and the
    third column be some independent feature.

    Args:
        l (int): number of trajectories per component
        n (int): number of time steps per trajectory
    """
    f1 = lambda x: 120 + 4 * x
    f2 = lambda x: 10 + 2 * x + 0.1 * x ** 2
    f3 = lambda x: 250 - 0.75 * x

    K = 3      # number of components
    M = l * K  # total number of trajectories
    N = M * n  # total number of observations

    samples = np.zeros((M, n))
    xs = np.random.normal(1.0, 1.0, (M, n))

    for i, model in enumerate([f1, f2, f3]):
        for traj in range(l):
            idx = i * l + traj  # stride + step
            for time in range(n):
                samples[idx, time] = model(xs[idx, time])

    X = np.zeros((N, 4))
    y = np.zeros(N)
    X[:, 0] = 1  # intercept term
    idx = 0
    for (traj, time), y_sample in np.ndenumerate(samples):
        X[idx, 1] = time
        X[idx, 2] = xs[traj, time]
        X[idx, 3] = xs[traj, time] ** 2
        y[idx] = y_sample
        idx += 1

    return X, y


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
        '-l', type=int, default=4,
        help='number of trajectories per component')
    parser.add_argument(
        '-n', type=int, default=10,
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

    l = args.l
    n = args.n

    K = 3           # number of components
    M = l * K  # total number of trajectories
    # l = 4      # trajectories per component
    # n = 10     # observations per trajectory
    # N = M * n  # total number of observations

    # X and y are (entity x time x predictors).
    X, y = gen_data(l, n)
    N, p = X.shape
    X_rw = X.reshape((M, n, p))  # rw = row-wise
    y_rw = y.reshape((M, n))

    # Init params to 0s.
    B = np.ndarray((K, p))
    var = np.ndarray(K)
    weight = np.ndarray(K)

    # params = {
    #     'B': [],
    #     'var': [],
    #     'w': []
    # }

    # handle divide by 0 errors
    np.seterr(divide='raise')  # raise divide by 0 errors
    eps = np.finfo(float).eps  # add to denominators which may go to 0

    # We allow minor fluctuation of log-likelihood for stopping.
    # It can increase for one iteration; we'll stop on second.
    # We track it with a counter.
    bad_iterations = 0

    # Randomly init membership probs h_{jk}.
    # There are M * K of these.
    # Note: the posteriors of this model end up with per-row membership weights
    H = np.random.uniform(0, 1, (M, K))
    H = H / H.sum(axis=1)[:, np.newaxis].repeat(K, axis=1)

    prev_llik = -np.inf  # set to lowest possible number
    llik = np.finfo(np.inf).min  # set to next lowest

    niters = np.iinfo(np.int).max if args.niters is None else args.niters
    iter_range = xrange(niters)
    for iteration in iter_range:

        # Estimate B_k, \sigma_k^2, w_k from weighted least squares solutions,
        # using current membership probs as weights.
        # We construct H_k as an N x N matrix whose diagonal contains the
        # weights to be applied to X and Y during regression.

        # first calculate new weights.
        H_sums = H.sum(axis=0)
        weight = (1. / M) * H_sums

        for k in range(K):
            H_k = np.diag(H[:, k].repeat(n, axis=0))

            # calculate B_k
            tmp = X.T.dot(H_k)
            p1 = np.linalg.inv(tmp.dot(X))
            p2 = tmp.dot(y)
            B[k] = p1.dot(p2)

            # calculate sigma_k^2
            regress = X.dot(B[k])
            residuals = y - regress
            var[k] = residuals.T.dot(H_k).dot(residuals)

        # divide sigmas by the membership weight sums for each component
        var = var / H_sums

        # params['B'].append(B)
        # params['var'].append(var)
        # params['w'].append(weight)

        # Compute new membership probs using new param estimates.
        H[:] = weight
        for j in range(M):
            X_j = X_rw[j]
            y_j = y_rw[j]
            for k in range(K):
                means = X_j.dot(B[k])
                sigma = np.sqrt(var[k])
                for time in range(n):
                    H[j, k] *= stats.norm.pdf(y_j[time], means[time], sigma)

        # re-normalize the membership weights
        H = H / H.sum(axis=1)[:, np.newaxis].repeat(K, axis=1)

        # Loop to step 2 until log-likelihood stabilizes.
        # Calculate expectation of complete data log-likelihood.
        llik = 0
        lliks = np.zeros((M, K))
        for j in range(M):
            X_j = X_rw[j]
            y_j = y_rw[j]
            for k in range(K):
                llik += H[j, k] * np.log(weight[k])

                means = X_j.dot(B[k])
                sigma = np.sqrt(var[k])
                for time in range(n):
                    lliks[j, k] += np.log(
                        stats.norm.pdf(y_j[time], means[time], sigma) + eps)

        lliks *= H
        llik += lliks.sum()
        logging.info('%d: log-likelihood: %.4f' % (iteration, llik))
        if np.isinf(llik):
            raise ValueError('log-likelihood has become infinite')

        # early stopping check if enabled (after iteration #5)
        if args.epsilon is not None and iteration > 5:
            diff = llik - prev_llik
            if diff == 0:  # done
                break
            elif diff <= args.epsilon:
                # allow small reductions which could be due to rounding error
                if diff < -0.01:
                    bad_iterations += 1
                    if bad_iterations > 1:
                        break
            else:
                bad_iterations = 0
                prev_llik = llik

