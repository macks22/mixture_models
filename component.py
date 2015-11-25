"""
Mixture component distributions.

"""
import numpy as np
import scipy as sp
import scipy.stats as stats
import scipy.special as spsp
import matplotlib.pyplot as plt

from distributions import multivariate_t, GIW


class GaussianComponentCache(object):
    slots = ['xbar', 'ssq', 'mu', 'kappa', 'nu', 'Psi', 'ppc', 'ppdf']

    def __init__(self, f):
        """Allocate space for cache variables based on number of features f."""
        self.xbar = np.zeros(f)
        self.ssq = np.zeros((f, f))

        self.mu = np.zeros(f)
        self.kappa = 0
        self.nu = 0
        self.Psi = np.zeros((f, f))

        # self.ppm = np.zeros(f)  -> posterior predictive mean = mu
        self.ppc = np.zeros((f, f))
        self.ppdf = 0

    def store(self, xbar, ssq, mu, kappa, nu, Psi, ppc, ppdf):
        """Store stats as instance variables."""
        self.xbar[:] = xbar
        self.ssq[:] = ssq

        self.mu[:] = mu
        self.kappa = kappa
        self.nu = nu
        self.Psi[:] = Psi

        self.ppc[:] = ppc
        self.ppdf = ppdf

    def restore(self, comp):
        """Restore cached stats to component instance variables."""
        comp._xbar[:] = self.xbar
        comp._ssq[:] = self.ssq

        comp.posterior.mu[:] = self.mu
        comp.posterior.kappa = self.kappa
        comp.posterior.nu = self.nu
        comp.posterior.Psi[:] = self.Psi

        comp.pp.cov[:] = self.ppc
        comp.pp.df = self.ppdf


class GaussianComponent(object):
    """Multivariate Gaussian distribution; for use as mixture component."""

    def __init__(self, X, instances, prior=None):
        """Assign some subset `instances` of the data `X` to this component.
        We use a boolean mask on the rows of X for efficient add/remove
        for data instances. This is important for the collapsed Gibbs sampling
        procedures, which are removing/adding instances from components during
        each sampling iteration.

        This class models a multivariate Gaussian with a conjugate
        Gaussian-Inverse-Wishart (GIW) prior. Hence we have hyper-parameters:

            mu0: prior mean on mu
            k0:  prior virtual sample size for mu0

            nu0: Wishart degrees of freedom
            Psi0: Wishart inverse scale matrix

        These hyper-parameters are intialized using an empirical Bayes estimate
        from the entire data matrix X. Specifically, mu0 is set to the sample
        mean, k0 is set to 1, nu0 is set to the number of features + 2, and Psi0
        is set to the sample sum of squared deviations.

        Args:
            X (np.ndarray): The matrix of data, where rows are data instances.
            instances (np.ndarray): Boolean mask to select which rows of the
                data matrix X belong to this component.
        """
        self._instances = instances
        self.X = X

        self.prior = prior
        if self.prior is None:

            # Init mu ~ Normal hyperparams.
            mu = self.X.mean(0)
            kappa = 1.0

            # Init Sigma ~ Inv-Wishart hyperparams.
            nu = self.nf + 2
            Psi = np.eye(self.nf) * nu

            self.prior = GIW(mu, kappa, nu, Psi)

        # These are set during fitting.
        self.posterior = self.prior.copy()
        self.pp = multivariate_t(mu, Psi, nu)  # placeholder values

        # When adding/removing instances from the components during fitting,
        # the same instance will often be added back immediately. We look for
        # case and avoid recomputing the sufficient stats, posterior, and
        # posterior predictive. Initially the cache is empty. We cache before
        # removing an instance.
        self._cache = GaussianComponentCache(self.nf)
        self._last_i_removed = -1

        # The precision matrix is the inverse of the covariance.
        # Whenever it is asked for, we'll need to get the inverse of
        # the current covariance matrix. We use a hash to avoid
        # recomputation as long as the covariance matrix is the same.
        self._cov_hash = 0

        # Fit params to the data.
        self.fit()

    @property
    def X(self):
        return self._X[self._instances]

    @X.setter
    def X(self, X):
        self._X = X

        # Cache stats used during fitting.
        n = self.n
        self._xbar = self.X.mean(0) if n else np.zeros(self.nf) # sample mean
        self._ssq = self.X.T.dot(self.X) # sample sum of squares

    @property
    def n(self):
        """Number of instances assigned to this component."""
        return self.X.shape[0]

    @property
    def nf(self):
        """Number of features."""
        return self.X.shape[1]

    @property
    def is_empty(self):
        """True if there are 0 instances assigned to this component."""
        return self.n == 0

    @property
    def precision(self):
        """Inverse of covariance matrix."""
        current_hash = hash(np.matrix(self.cov))
        if self._cov_hash != current_hash:  # current precision is stale
            self._cov_hash = current_hash
            self._precision = np.linalg.inv(self.cov)
        return self._precision

    @property
    def mean(self):
        return self.prior.mu

    @property
    def cov(self):
        return self.prior.Psi / (self.prior.nu - self.prior.Psi.shape[0] - 1)

    def sufficient_stats(self):
        """Return sample size, mean, and sum of squares."""
        return self.n, self._xbar, self._ssq

    def fit_posterior(self):
        """Update posterior using conjugate hyper-parameter updates based on
        observations X.
        """
        self.prior.conjugate_updates(
            self.n, self._xbar, self._ssq, self.posterior)

    def _cache_stats(self):
        self._cache.store(
            self._xbar, self._ssq, self.posterior.mu, self.posterior.kappa,
            self.posterior.nu, self.posterior.Psi, self.pp.cov, self.pp.df)

    def _restore_from_cache(self):
        self._cache.restore(self)

    def rm_instance(self, i):
        if not self._instances[i]:
            raise IndexError('index %i not currently in component' % i)

        self._cache_stats()
        self._last_i_removed = i

        # Remove sufficient stats from sample mean & sum of squares.
        x = self._X[i]
        self._ssq -= x[:, None].dot(x[None, :])
        new_n = self.n - 1
        if new_n == 0:
            self._xbar[:] = 0
        else:
            self._xbar = (self._xbar * self.n - x) / new_n

        self._instances[i] = False  # remove instance
        self.fit()  # handles empty component case

    def add_instance(self, i):
        """Add an instance to this Gaussian component.
        This is done by setting element i of the `instances` mask to True.
        """
        if self._instances[i]:  # already in component
            return

        if self._last_i_removed == i:
            self._restore_from_cache()
        else:
            # Add sufficient stats to the cached stats.
            x = self._X[i]
            self._ssq += x[:, None].dot(x[None, :])
            self._xbar = (self._xbar * self.n + x) / (self.n + 1)
            self.fit()

        self._instances[i] = True

    def fit(self):
        """Perform conjugate updates of the GIW prior parameters to compute the
        posterior mean and convaraince. We also compute the posterior predictive
        mean and covariance for use in the students-t posterior predictive pdf.

        Eqs. (8) and (15) in Kamper's guide.
        """
        self.fit_posterior()

        # Posterior predictive parameter calculations.
        # Eq. (15) from Kamper's guide.
        self.pp.mean = self.posterior.mu
        self.pp.df = self.posterior.nu - self.nf + 1
        self.pp.cov = ((self.posterior.Psi * (self.posterior.kappa + 1))
                       / (self.posterior.kappa * self.pp.df))

    def pdf(self, x):
        """Multivariate normal probability density function."""
        return stats.multivariate_normal.pdf(x, self.mean, self.cov)

    def llikelihood(self):
        """Compute marginal log likelihood of data given the observed
        data instances assigned to this component.

        Eq. (266) from Murphy (2007).
        """
        half_d = self.nf * 0.5
        nu_prior = self.prior.nu * 0.5
        nu_post = self.posterior.nu * 0.5

        return (spsp.gammaln(nu_post) - spsp.gammaln(nu_prior)
                + nu_prior * np.log(np.linalg.det(self.prior.Psi))
                - nu_post * np.log(np.linalg.det(self.posterior.Psi))
                + half_d * (np.log(self.prior.kappa) - np.log(self.posterior.kappa))
                - (self.n * half_d) * np.log(np.pi))

    def likelihood(self):
        """Compute marginal likelihood of data given the observed
        data instances assigned to this component.
        """
        return np.exp(self.llikelihood())


