"""
Base class for mixture models.

"""
import numpy as np


class MixtureModel(object):
    """General mixture model base class."""

    """Initialization methods supported & not yet implemented."""
    _supported = ('kmeans', 'random', 'load')
    _not_implemented = ('load',)

    @classmethod
    def supported_init_method(cls, init_method):
        return init_method in cls._supported

    @classmethod
    def implemented_init_method(cls, init_method):
        return init_method not in cls._not_implemented

    @classmethod
    def validate_init_method(cls, init_method):
        if init_method not in cls._supported:
            raise ValueError(
                '%s is not a supported init method; must be one of: %s' % (
                    init_method, ', '.join(supported)))

        if init_method in cls._not_implemented:
            raise NotImplemented(
                '%s initialization not yet implemented' % init_method)


    def __init__(self):
        self.comps = []  # mixture components
        self.z = []      # assignments of data instances to mixture components

    def __iter__(self):
        return (comp for comp in self.comps)

    @property
    def counts(self):
        return np.array([comp.n for comp in self])

    @property
    def n(self):
        return self.counts.sum()

    @property
    def nf(self):
        compiter = iter(self)
        try:
            comp = compiter.next()
            return comp.nf
        except StopIteration:
            return 0

    def init_comps(self):
        raise NotImplemented(
            'init_comps should be implemented by base classes')

    def fit(self):
        raise NotImplemented(
            'fit should be implemeneted by base classes')

    def label_llikelihood(self):
        raise NotImplemented(
            'label_llikelihood should be implemeneted by base classes')

    def label_likelihood(self):
        """Calculate P(z | alpha), the marginal likelihood of the component
        instance assignments.
        """
        return np.exp(self.label_llikelihood())

    def llikelihood(self):
        raise NotImplemented(
            'llikelihood should be implemeneted by base classes')

    def likelihood(self):
        """Calculate P(X, z | alpha, pi, mu, Sigma), the marginal likelihood
        of the data and the component instance assignments given the parameters
        and hyper-parameters.
        """
        return np.exp(self.llikelihood())


class MixtureComponentCache(object):

    def store(self, comp):
        raise NotImplemented()

    def restore(self, comp):
        raise NotImplemented()


class MixtureComponent(object):
    """Models a mixture model component."""

    cache_class = MixtureComponentCache

    def __init__(self, X, instances, prior=None):
        # When adding/removing instances from the components during fitting,
        # the same instance will often be added back immediately. We look for
        # this case and avoid recomputing the sufficient stats, posterior, and
        # posterior predictive. Initially the cache is empty. We cache before
        # removing an instance.
        self._cache = self.cache_class(X.shape[1])
        self._last_i_removed = -1

        self._instances = instances
        self.X = X
        self.prior = self.default_prior() if prior is None else prior
        self.posterior = self.prior.copy()

    def default_prior(self):
        raise NotImplemented()

    def _populate_cache(self):
        """Cache stats used during fitting."""
        raise NotImplemented()

    @property
    def X(self):
        return self._X[self._instances]

    @X.setter
    def X(self, X):
        self._X = X
        self._populate_cache()

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

    def _cache_stats(self):
        self._cache.store(self)

    def _restore_from_cache(self):
        self._cache.restore(self)

    def _cache_rm_instance(self, i):
        """Remove sufficient stats from sample mean & sum of squares."""
        raise NotImplemented()

    def rm_instance(self, i):
        if not self._instances[i]:
            raise IndexError('index %i not currently in component' % i)

        self._cache_stats()
        self._last_i_removed = i
        self._instances[i] = False  # remove instance
        self._cache_rm_instance(i)  # remove from cached stats
        self.fit()  # handles empty component case

    def _cache_add_instance(self, i):
        """Add sufficient stats from this instance to cached stats."""
        raise NotImplemented()

    def add_instance(self, i):
        """Add an instance to this Gaussian component.
        This is done by setting element i of the `instances` mask to True.
        """
        if self._instances[i]:  # already in component
            return

        self._instances[i] = True
        if self._last_i_removed == i:
            self._restore_from_cache()
        else:
            self._cache_add_instance(i)  # add to cached stats
            self.fit()

    def sufficient_stats(self):
        """Return sufficient statistics."""
        raise NotImplemented()

    def fit_posterior(self):
        """Update posterior using conjugate hyper-parameter updates based on
        observations X.
        """
        args = tuple(list(self.sufficient_stats()) + [self.posterior])
        self.prior.conjugate_updates(*args)

    def fit_pp(self):
        """Posterior predictive parameter calculations."""
        raise NotImplemented()

    def fit(self):
        """Perform conjugate updates of prior and posterior predictive
        parameters.
        """
        self.fit_posterior()
        self.fit_pp()

    def pdf(self, x):
        """Multivariate normal probability density function."""
        raise NotImplemented()

    def llikelihood(self):
        """Compute marginal log likelihood of data given the observed
        data instances assigned to this component.
        """
        raise NotImplemented()

    def likelihood(self):
        """Compute marginal likelihood of data given the observed
        data instances assigned to this component.
        """
        return np.exp(self.llikelihood())

