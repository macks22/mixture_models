"""
Mixture component distributions.

"""
import numpy as np
import scipy as sp
import scipy.stats as stats
import scipy.special as spsp
import matplotlib.pyplot as plt

from distributions import multivariate_t, GIG, GIW, NormalGamma
from mixture import MixtureComponent, MixtureComponentCache
import ccomp


class GaussianComponentCache(MixtureComponentCache):
    __slots__ = ['xbar', 'ssq', 'mu', 'kappa', 'nu', 'Psi', 'ppc', 'ppdf']

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

    def store(self, comp):
        """Store stats as instance variables."""
        self.xbar[:] = comp._xbar
        self.ssq[:] = comp._ssq

        self.mu[:] = comp.posterior.mu
        self.kappa = comp.posterior.kappa
        self.nu = comp.posterior.nu
        self.Psi[:] = comp.posterior.Psi

        self.ppc[:] = comp.pp.cov
        self.ppdf = comp.pp.df

    def restore(self, comp):
        """Restore cached stats to component instance variables."""
        comp._xbar[:] = self.xbar
        comp._ssq[:] = self.ssq

        GIW.__init__(comp.posterior, self.mu, self.kappa, self.nu, self.Psi)

        comp.pp.cov[:] = self.ppc
        comp.pp.df = self.ppdf


class GaussianComponent(MixtureComponent):
    """Multivariate Gaussian distribution; for use as mixture component."""

    cache_class = GaussianComponentCache

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
        MixtureComponent.__init__(self, X, instances, prior)

        # This is set during fitting.
        self.pp = multivariate_t(
            self.prior.mu, self.prior.Psi, self.prior.nu)  # placeholder values

        # The precision matrix is the inverse of the covariance.
        # Whenever it is asked for, we'll need to get the inverse of
        # the current covariance matrix. We use a hash to avoid
        # recomputation as long as the covariance matrix is the same.
        self._cov_hash = 0

        # Fit params to the data.
        self.fit()

    def default_prior(self):
        """Return default prior distribution."""
        # Init mu ~ Normal hyperparams.
        # mu = self.X.mean(0) if self.n else np.zeros(self.nf)
        mu = np.zeros(self.nf)
        kappa = 1.0

        # Init Sigma ~ Inv-Wishart hyperparams.
        nu = self.nf + 2
        Psi = np.eye(self.nf) * nu

        return GIW(mu, kappa, nu, Psi)

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

    def _populate_cache(self):
        """Cache stats used during fitting."""
        X = self.X
        self._xbar = X.mean(0) if self.n \
                else np.zeros(self.nf) # sample mean
        self._ssq = X.T.dot(X) # sample sum of squares

    def _cache_rm_instance(self, i):
        """Remove sufficient stats from sample mean & sum of squares."""
        x = self._X[i]
        self._ssq -= x[:, None].dot(x[None, :])
        new_n = self.n - 1
        if new_n == 0:
            self._xbar[:] = 0
        else:
            self._xbar = (self._xbar * self.n - x) / new_n

    def _cache_add_instance(self, i):
        """Add sufficient stats from this instance to cached stats."""
        x = self._X[i]
        self._ssq += x[:, None].dot(x[None, :])
        self._xbar = (self._xbar * self.n + x) / (self.n + 1)

    def sufficient_stats(self):
        """Return sample size, mean, and sum of squares."""
        return self.n, self._xbar, self._ssq

    def fit_pp(self):
        """Posterior predictive parameter calculations.
        Eq. (15) from Kamper's guide.
        """
        self.pp.mean = self.posterior.mu
        self.pp.df = self.posterior.nu - self.nf + 1
        pp_cov = ((self.posterior.Psi * (self.posterior.kappa + 1))
                   / (self.posterior.kappa * self.pp.df))
        try:
            self.pp.cov = pp_cov
        except sp.linalg.LinAlgError:
            # attempt to resolve positive semi-definite issues by setting
            # non-pd minors to machine epsilon.
            evals, evecs = np.linalg.eigh(pp_cov)
            evals[evals <= 0] = np.finfo(np.float32).eps
            pp_cov = evecs.dot(np.diag(evals)).dot(evecs.T)
            self.pp.cov = pp_cov

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
                + nu_prior * self.prior._Psi_logdet
                - nu_post * self.posterior._Psi_logdet
                + half_d * (np.log(self.prior.kappa) - np.log(self.posterior.kappa))
                - (self.n * half_d) * np.log(np.pi))


class MGLRComponentCache(MixtureComponentCache):
    slots = ['x_ssq', 'y_ssq', 'xy', 'mu', 'V', 'a', 'b']#, 'ppm', 'ppc', 'ppdf']

    def __init__(self, f):
        """Allocate space for cache variables based on number of features f."""
        self.x_ssq = np.zeros((f, f))
        self.y_ssq = 0.0
        self.xy = np.zeros((f,))

        self.mu = np.zeros(f)
        self.V = np.zeros((f, f))
        self.a = 0
        self.b = 0

        # self.ppm = np.zeros((f,))
        # self.ppc = np.zeros((f, f))
        # self.ppdf = 0

    def store(self, comp):
        """Store stats as instance variables."""
        self.x_ssq[:] = comp._x_ssq
        self.y_ssq = comp._y_ssq
        self.xy[:] = comp._xy

        self.mu[:] = comp.posterior.mu
        self.V[:] = comp.posterior.V
        self.a = comp.posterior.a
        self.b = comp.posterior.b

        # self.ppm[:] = comp.pp.mean
        # self.ppc[:] = comp.pp.cov
        # self.ppdf = comp.pp.df

    def restore(self, comp):
        """Restore cached stats to component instance variables."""
        comp._x_ssq[:] = self.x_ssq
        comp._y_ssq = self.y_ssq
        comp._xy[:] = self.xy

        GIG.__init__(comp.posterior, self.mu, self.V, self.a, self.b)

        # comp.pp.mean[:] = self.ppm
        # comp.pp.cov[:] = self.ppc
        # comp.pp.df = self.ppdf


class MGLRComponent(MixtureComponent):
    """Mixture of Gaussian Linear Regressions with conjugate GIG prior."""

    cache_class = MGLRComponentCache

    def __init__(self, X, y, instances, prior=None):
        """Assign a subset `instances` of the data `X`, `y` to this component.
        We use a boolean mask on the instances for efficient add/remove. This is
        important for the collapsed Gibbs sampling procedures, which are
        removing/adding instances from components during each sampling
        iteration.

        This class models a multivariate Gaussian with a conjugate
        Gaussian-Inverse-Gamma (GIG) prior. Hence we have hyper-parameters:

            mu: prior mean on mu
            V:  prior scale matrix (multiplies by sigma)

            a: prior shape
            b: prior rate

        These hyper-parameters are intialized according to `default_prior`.

        Args:
            X (np.ndarray): The matrix of data, where rows are data instances.
            y (np.ndarray): Observations corresponding to X.
            instances (np.ndarray): Boolean mask to select which rows of the
                data matrix X belong to this component.
            prior (GIG): Optional prior distribution; defaults to
                `default_prior`.
        """
        self._y = y  # bypass cache population in setter

        # This sets X, prior and posterior, and populates the cache.
        super(MGLRComponent, self).__init__(X, instances, prior)

        # This is set during fitting; populate with placeholder values for now.
        self.pp = multivariate_t(
            self.prior.mu, self.prior.V, self.prior.V.shape[0] * 2)

        # Fit params to the data.
        self.fit()

    @property
    def y(self):
        return self._y[self._instances]

    @y.setter
    def y(self, y):
        self._y = y
        self._populate_cache()

    def rm_instances(self, _is):
        if not self._instances[_is].all():
            raise IndexError(
                'some indices from %s not currently in component' % (
                    str(_is)))

        self._cache_stats()
        self._last_i_removed = _is
        self._instances[_is] = False  # remove instances
        self._cache_rm_instances(_is)  # remove from cached stats
        self.fit()  # can deal with empty component case

    def add_instances(self, _is):
        """Add more than one instance to this component.
        This is done by setting all elements _is of the `instances` mask True.
        """
        # ignore instances already in component.
        _is = np.array([i for i in _is if not self._instances[i]])
        if _is.shape[0] == 0:
            return

        self._instances[_is] = True

        # If _is is an index mask (rather than a boolean mask) and the masks are
        # of unequal length, then __eq__ returns a bool.
        equal = self._last_i_removed == _is
        if not isinstance(equal, bool):
            if equal.all():  # same dimension, all elements same?
                self._restore_from_cache()
        else:
            self._cache_add_instances(_is)  # add to cached stats
            self.fit()

    @property
    def sigma(self):
        # TODO: make sure this is the correct formula
        return self.posterior.a / self.posterior.b

    def mean(self, X):
        return X.dot(self.posterior.mu)

    def cov(self):
        return self.sigma * np.eye(self.nf)

    def default_prior(self):
        """Return default prior distribution."""
        f = self.nf
        mu = np.zeros((f,))
        V = np.eye(f)
        return GIG(mu, V, 1.0, 1.0)

    def _populate_cache(self):
        """Cache stats used during fitting."""
        X = self.X
        self._x_ssq = X.T.dot(X)  # sample X sum of squares
        self._y_ssq = self.y.dot(self.y)    # sample y sum of squares
        self._xy = X.T.dot(self.y)  # matrix multiplication of X^T y

    def _cache_rm_instance(self, i):
        """Remove sufficient stats from sample mean & sum of squares."""
        x = self._X[i]
        y = self._y[i]
        new_n = self.n - 1
        if new_n == 0:
            self._x_ssq[:] = 0
            self._y_ssq = 0
            self._xy[:] = 0
        else:
            self._x_ssq -= x[:, None].dot(x[None, :])
            self._y_ssq -= (y ** 2)
            self._xy -= x.dot(y)

    def _cache_rm_instances(self, _is):
        """Remove more than one instnace from cached sufficient stats."""
        X = self._X[_is]
        y = self._y[_is]
        new_n = self.n - X.shape[0]
        if new_n == 0:
            self._x_ssq[:] = 0
            self._y_ssq = 0
            self._xy[:] = 0
        else:
            self._x_ssq = self._x_ssq - X.T.dot(X)
            self._y_ssq = self._y_ssq - y.dot(y)
            self._xy = self._xy - X.T.dot(y)

    def _cache_add_instance(self, i):
        """Add sufficient stats from this instance to cached stats."""
        x = self._X[i]
        y = self._y[i]
        self._x_ssq += x[:, None].dot(x[None, :])
        self._y_ssq += (y ** 2)
        self._xy += x.dot(y)

    def _cache_add_instances(self, _is):
        """Add sufficient stats from more than one instance to cached stats."""
        X = self._X[_is]
        y = self._y[_is]
        self._x_ssq += X.T.dot(X)
        self._y_ssq += y.dot(y)
        self._xy += X.T.dot(y)

    def sufficient_stats(self):
        """Return sample size, mean, and sum of squares."""
        return self.n, self._x_ssq, self._y_ssq, self._xy

    def fit_pp(self, X):
        """Posterior predictive parameter calculations.
        Eq. (20) from Banerjee's BLM: Gory Details.

        Raises:
            sp.linalg.LinAlgError: if the posterior predictive covariance matrix
                is not positive definite.
        """
        # Instead of creating a new object, reuse the current one.
        # This is an optimization that replaces the __init__.
        # self.pp = multivariate_t(mean, pp_cov, df)
        self.pp._mean, self.pp.cov, self.pp.df = ccomp.mglr_fit_pp(
            X, self.posterior.V, self.posterior.b, self.posterior.a,
            self.posterior.mu)

    def fit(self):
        self.fit_posterior()

    def logpdf(self, X, y):
        """GMLR log probability density function."""
        return stats.multivariate_normal.logpdf(y, self.mean(X), self.cov())

    def pdf(self, X, y):
        """GMLR probability density function."""
        return np.exp(self.logpdf(X, y))

    def llikelihood(self):
        """Compute marginal log likelihood of data given the observed
        data instances assigned to this component.

        Eq. (4) from Banerjee's BLM: Gory Details.
        """
        # Optimizations
        X = self.X
        n = X.shape[0]

        sigma = self.sigma
        reg = X.dot(self.posterior.mu)
        residuals = self.y - reg
        return (n * np.log(2 * np.pi * sigma)
                + (1 / sigma) * residuals.dot(residuals)) * -0.5


class NormalBiasVector(object):

    def __init__(self, biases, prior=None):
        """
        Args:
            samples (np.ndarray): bias samples for init
            prior (object): optional prior to override default.
        """
        self._n = biases.shape[0]
        self.biases = biases

        # Set prior to given value or uninformative normal-gamma.
        self.prior = NormalGamma.uninformative() if prior is None else prior

        # Fit initial posterior.
        self.posterior = self.prior.copy()
        self.fit()

    def __repr__(self):
        return "NormalBiasVector(biases={}, prior={})".format(self.biases, self.prior)

    def __str__(self):
        return self.__repr__()

    def __getitem__(self, idx):
        return self.biases[idx]

    def __setitem__(self, idx, value):
        self.biases[idx] = value

    def rvs(self):
        mu, precision = self.posterior.rvs()
        std = np.sqrt(1. / precision)
        n = self.biases.shape[0]
        return stats.norm.rvs(loc=mu, scale=std, size=n)

    def sufficient_stats(self):
        n = self.biases.shape[0]
        xbar = np.nanmean(self.biases)
        ssqd = n * np.nanvar(self.biases)
        return n, xbar, ssqd

    def fit_posterior(self):
        """Update posterior using conjugate hyperparameter updates from observed bias
        terms.
        """
        args = tuple(list(self.sufficient_stats()) + [self.posterior])
        self.prior.conjugate_updates(*args)

    def fit(self):
        self.fit_posterior()
