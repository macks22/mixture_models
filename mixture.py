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
