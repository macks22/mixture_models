# -*- coding: utf-8 -*-
import sys
import itertools

import numpy as np
import scipy.linalg as spla
import scipy.cluster.vq as vq
from scipy import stats

# Plotting
import matplotlib.pyplot as plt
from sklearn import manifold
import matplotlib as mpl


"""
Since we are using objects for our components, we need to be careful
to avoid over-replication of the data. Each component will be assigned
particular instances from the data X. These will change over time. The
way we choose to implement this is to initialize with the matrix X as
well as a boolean mask on X. Then adding/removing data instances is as
simple as flipping a boolean value.
"""

class Gaussian(object):
    """A multivariate Gaussian component in a mixture model."""

    def __init__(self, X, instances):
        self._X = X
        self.instances = instances  # boolean mask takes n bytes

        # Init hyperparams.
        self.mu0 = np.zeros(self.nf)  # 0 prior means on covariates
        self.k0 = 0.0          # prior (virtual) sample size
        self.nu0 = self.nf     # degrees of freedom = number of features
        self.Psi0 = np.eye(self.nf) * self.nf  # inverse scale matrix
        self.sigma = 100.0     # prior covariance (scalar)

        # The precision matrix is the inverse of the covariance.
        # Whenever it is asked for, we'll need to get the inverse of
        # the current covariance matrix. We use a hash to avoid
        # recomputation as long as the covariance matrix is the same.
        self._cov_hash = 0

        # Fit params to the data.
        self.fit_params()

    @property
    def X(self):
        return self._X[self.instances]

    @X.setter
    def X(self, X):
        self._X = X

    @property
    def n(self):
        return self.X.shape[0]

    @property
    def nf(self):
        return self.X.shape[1]

    @property
    def is_empty(self):
        return self.n == 0

    @property
    def precision(self):
        current_hash = hash(np.matrix(self.cov))
        if self._cov_hash != current_hash:  # current precision is stale
            self._cov_hash = current_hash
            self._precision = np.linalg.inv(self.cov)
        return self._precision

    def fit_params(self):
        """Compute the posterior mean and covariance of the component
        from the data and the hyperparameters.
        """
        n = self.n
        if n == 0:  # empty component; set default params (hyperparams for G_0)
            nf = self.nf
            self.mean = np.zeros(nf)
            self.cov = np.eye(nf) * self.sigma  # could sample from invwishart
            return

        xbar = self.X.sum(axis=0)
        kappa = self.k0 + n
        nu = self.nu0 + n
        self.mean = (self.k0 * self.mu0 + n * xbar) / kappa

        dev = self.X - xbar
        C = dev.T.dot(dev)
        dev = xbar - self.mu0
        Psi = self.Psi0 + C + (self.k0 * n * dev.dot(dev) / kappa)
        self.cov = Psi * (kappa + 1) / (kappa * (nu - self.nf + 1))
        assert(np.linalg.det(self.cov) != 0)

    def add_instance(self, i):
        """Add an instance to this Gaussian cluster.
        This is done by setting element i of the `instances` mask to True.
        """
        if self.instances[i]:  # already in component
            return

        self.instances[i] = True
        self.fit_params()

    def rm_instance(self, i):
        if not self.instances[i]:
            raise IndexError('index %i not currently in component' % i)

        self.instances[i] = False  # remove instance
        self.fit_params()  # handles empty component case

    def _pdf(self, i):
        """PDF over a data instance in X."""
        return stats.multivariate_normal.pdf(self._X[i], self.mean, self.cov)

    def pdf(self, x):
        """Probability distribution function for data instance x.
        x should have the same number of features as this component.
        """
        return stats.multivariate_normal.pdf(x, self.mean, self.cov)


def log_likelihood(X, comps, z):
    log_2pi = np.log(2 * np.pi)
    llik = 0.0
    n, nf = X.shape
    for i in xrange(n):
        comp = comps[z[i]]
        llik -= 0.5 * n * log_2pi + 0.5 * np.log(np.linalg.det(comp.cov))

        # TODO should compute self.params[self.z[n]]._X.mean(axis=0) less often
        mean_var = X[i] - comp.X.mean(axis=0)
        # assert(mean_var.shape == (1, self.params[self.z[n]].nf))

        llik -= 0.5 * mean_var.dot(comp.precision).dot(mean_var.T)
        # TODO add the influence of n_components

    return llik


if __name__ == "__main__":
    np.random.seed(1234)

    max_iters = 100  # hard limit on number of learning iterations
    M = 50           # number of samples per component
    K = 2            # initial guess for number of clusters
    method = 'kmeans'  # parameter initialization method

    # Generate two 2D Gaussians
    C = np.array([[0., -0.1], [1.7, .4]])
    X = np.r_[np.dot(np.random.randn(M, 2), C),
              .7 * np.random.randn(M, 2) + np.array([-6, 3])]
    n, nf = X.shape  # number of instances, number of features

    # Draw dispersion parameter alpha from weakly informative conjugate prior
    # Inverse-Beta(1, 1).
    # alpha = stats.invgamma.rvs(1, 1)
    alpha = 0.5

    # We will iterate over all indices in random order.
    indices = np.arange(n)

    # We will sample only for the clusters currently associated with some data
    # point. All data points are associated with a cluster. We initialize the
    # clusterings using k-means.
    centroids, z = vq.kmeans2(X, K, minit='points', iter=100)
    comps = {k: Gaussian(X, z == k) for k in range(K)}

    # We use a hard limit on the number of learning iterations to avoid
    # cluster oscillations in a non-converging situation.
    iternum = 0
    while (iternum < max_iters):
        iternum += 1
        prev_means = np.r_[np.array([g.mean for g in comps.values()])]

        for i in np.random.permutation(indices):  # randomize instances
            x = X[i]
            comp = comps[z[i]]

            # remove X[i]'s sufficient stats from z[i]
            comp.rm_instance(i)

            # if it empties the cluster, remove it and decrease K
            if comp.is_empty:
                del comps[z[i]]

            # Probability that X[i] belongs to cluster k given all previous data
            # belonging to cluster k. Compute for all clusters.
            P_k = []
            for k, comp in comps.iteritems():
                # compute P_k(X[i]) = P(X[i] | X_{-i} = k)
                marginal_lik_Xi = comp.pdf(X[i])
                # set N_{k,-i} = dim({X[-i] = k})
                # compute P(z[i] = k | z[-i], Data) = N_{k,-i}/(α+N-1)
                mixing_Xi = comp.n / (alpha + n - 1)
                P_k.append(marginal_lik_Xi * mixing_Xi)

            # Probability X[i] belongs to a new cluster/distribution k*.
            # compute P*(X[i]) = P(X[i]|λ)
            base_distrib = Gaussian(X, np.zeros(n).astype(np.bool))
            prior_predictive = base_distrib.pdf(X[i])
            # compute P(z[i] = k* | z_{-i}, Data) = α/(α+N-1)
            prob_new_cluster = alpha / (alpha + n - 1)
            P_k.append(prior_predictive * prob_new_cluster)

            # normalize P(z[i])
            P_k = np.array(P_k)
            P_k = P_k / P_k.sum()

            # sample z[i] ~ P(z[i])
            k = np.random.choice(np.arange(P_k.shape[0]), p=P_k)

            # add X[i]'s sufficient statistics to cluster z[i]
            new_key = max(comps.keys()) + 1
            if k == len(comps): # create a new cluster
                z[i] = new_key
                comps[new_key] = Gaussian(X, z == new_key)
            else:
                _k = comps.keys()[k]
                z[i] = _k
                comps[_k].add_instance(i)

        print("still sampling, %i clusters, log-likelihood %f" % (
            len(comps), log_likelihood(X, comps, z)))

        means = np.r_[np.array([g.mean for g in comps.values()])]
        if (means.shape == prev_means.shape and
                np.isclose(means, prev_means).all()):
            # means and number of components have converged.
            break


    # Plot results.
    X_repr = X
    if X.shape[1] > 2:
        X_repr = manifold.Isomap(n_samples/10, n_components=2).fit_transform(X)

    color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])
    splot = plt.subplot(1, 1, 1)
    means = np.array([g.mean for g in comps.values()])
    covs = np.array([g.cov for g in comps.values()])
    Y_ = z

    for j, (mean, covar, color) in enumerate(zip(means, covs, color_iter)):
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == j):
            continue

        plt.scatter(X_repr[Y_ == j, 0], X_repr[Y_ == j, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        v, w = spla.eigh(covar)
        u = w[0] / spla.norm(w[0])
        angle = np.arctan(u[1] / u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(xy=mean, width=v[0], height=v[1],
                                  angle=(180 + angle), color='k')
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-10, 10)
    plt.ylim(-3, 6)
    plt.xticks(())
    plt.yticks(())
