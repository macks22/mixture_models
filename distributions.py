import numpy as np
import scipy as sp
import scipy.stats as stats
import scipy.special as spsp


class multivariate_t_gen(stats._multivariate.multi_rv_generic):

    def __call__(self, mean, cov, df):
        return multivariate_t_frozen(mean, cov, df)

    def _logpdf(self, x, mean, cov, df):
        """Multivariate Students-t log probability density function.
        """
        p = x.shape[0]  # dimension, should match mean and cov
        chodecomp = sp.linalg.cholesky(cov)
        sols = sp.linalg.solve(chodecomp, x - mean)
        rss = (sols ** 2).sum()

        return (spsp.gammaln((p + df) * 0.5)
                - (spsp.gammaln(df * 0.5)
                    + sum(np.log(np.diag(chodecomp)))
                    + (p / 2.) * np.log(np.pi * df))
                - 0.5 * (p + df) * np.log1p(rss / df))

    def logpdf(self, x, mean, cov, df):
        """Multivariate Students-t log probability density function.

        Translated from R's `pmvt`.

        Args:
            x (np.ndarray): 1 x p data vector observation.
            mean (np.ndarray): 1 x p mean vector of the Students-t.
            cov (np.ndarray): p x p covariance matrix of the Students-t.
            df (int): Scalar degrees of freedom of the Students-t.

        Returns:
            (float): The probability of observing x from a multivariate
                Students-t distribution with the given parameters.
        """
        x = np.array(x)
        p = x.shape[0]  # dimension, should match mean and cov

        if df == 0 or np.isinf(df):
            return stats.multivariate_normal.logpdf(x, mean, cov)

        if p != mean.shape[0]:
            raise ValueError(
                'mean.shape[0] must match x.shape[0] (%d != %d)' % (
                    mean.shape[0], p))

        if p != cov.shape[0]:
            raise ValueError(
                'cov.shape[0] must match x.shape[0] (%d != %d)' % (
                    cov.shape[0], p))

        if not (cov.T == cov).all():
            raise ValueError('cov matrix must be symmetric')

        return self._logpdf(x, mean, cov, df)

    def _pdf(self, x, mean, cov, df):
        return np.exp(self._logpdf(x, mean, cov, df))

    def pdf(self, x, mean, cov, df):
        """Multivariate Students-t probability density function."""
        return self._pdf(x, mean, cov, df)

    def _rvs(self, mean, cov, df, size=1):
        """Produce random variates from the multivariate Students-t
        distribution.
        """
        d = mean.shape[0]
        g = np.tile(np.random.gamma(df / 2., 2. / df, size), (d, 1)).T
        Z = np.random.multivariate_normal(np.zeros(d), cov, size)
        return np.squeeze(mean + Z / np.sqrt(g))

    def rvs(self, mean, cov, df, size=1):
        """Draw random variates from the multivariate Students-t
        distribution.

        Args:
            mean (np.ndarray): 1 x p mean vector of the Students-t.
            cov (np.ndarray): p x p covariance matrix of the Students-t.
            df (int): Scalar degrees of freedom of the Students-t.
        """
        return self._rvs(mean, cov, df, size)

multivariate_t = multivariate_t_gen()


class multivariate_t_frozen(stats._multivariate.multi_rv_frozen):

    def __init__(self, mean, cov, df):
        self.mean = mean
        self.cov = cov
        self.df = df
        self._dist = multivariate_t

    def pdf(self, x):
        return self._dist.pdf(x, self.mean, self.cov, self.df)

    def logpdf(self, x):
        return self._dist.pdf(x, self.mean, self.cov, self.df)

    def rvs(self, size):
        return self._dist.rvs(self.mean, self.cov, self.df, size)


class GIW(object):
    """Gaussian-Inverse Wishart distribution."""
    __slots__ = ['mu', 'kappa', 'nu', 'Psi']

    # def __init__(self, d, mu=None, k=None, nu=None, Psi=None):
        # self.mu0 = np.zeros(d) if mu is None else mu
        # self.k0 = 1.0 if k is None else k
        # self.nu0 = d + 2 if nu is None else nu
        # self.Psi0 = np.eye(d) * self.nu0 if Psi is None else Psi

    def __init__(self, mu, kappa, nu, Psi):
        self.mu = mu
        self.kappa = kappa
        self.nu = nu
        self.Psi = Psi
        if nu < len(Psi):
            raise ValueError('nu0 must be >= len(Psi0)')

    @property
    def d(self):
        return len(self.Psi)

    def rvs(self):
        cov = stats.invwishart.rvs(self.nu, self.Psi)
        mu = stats.multivariate_normal.rvs(self.mu, cov / self.kappa)
        return mu, cov

    def conjugate_updates(self, n, xbar, S):
        """Return GIW with conjugate hyper-parameter updates given sufficient
        statistics.

        Args:
            xbar (np.ndarray): 1 x d sample mean from data matrix X.
            S (np.ndarray):  d x d sample sum of squared deviations from X.
            n (int): Number of samples observed (rows in X).
        """
        kappa = self.kappa + n
        nu = self.nu + n
        mu = (self.kappa * self.mu + n * xbar) / kappa

        dev = xbar - self.mu
        uncertainty = \
            ((self.kappa * n) / kappa) * dev[:, None].dot(dev[None, :])
        Psi = self.Psi + S + uncertainty

        return GIW(mu, kappa, nu, Psi)

