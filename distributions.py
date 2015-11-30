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
        self._mean = np.array(mean)
        self.cov = cov
        self.df = df
        self._dist = multivariate_t

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, mean):
        if self.cov.shape[0] != mean.shape[0]:
            raise ValueError(
                'incompatible shape: cov.shape[0] != mean.shape[0] '
                '(%d != %d)' % (self.cov.shape[0], mean.shape[0]))

        self._mean = mean

    @property
    def cov(self):
        return self._cov

    @cov.setter
    def cov(self, cov):
        f = cov.shape[0]
        if self.mean.shape[0] != f:
            raise ValueError(
                'incompatible shape: mean.shape[0] != cov.shape[0] '
                '(%d != %d)' % (self.mean.shape[0], f))

        # if not (cov.T == cov).all():
        #     raise ValueError('cov matrix must be symmetric')

        self._cov = cov
        self._cholesky = sp.linalg.cholesky(cov)
        self._sum_lndiag = np.log(np.diag(self._cholesky)).sum()

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, df):
        p = self.mean.shape[0]
        if df < p:
            raise ValueError('degrees of freedom must be >= %d' % p)

        self._df = df
        self._half_dfp = (df + p) / 2.
        self._df_terms = (spsp.gammaln(self._half_dfp)
                          - spsp.gammaln(df / 2.)
                          - (p / 2.) * np.log(np.pi * df))

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def logpdf(self, x):
        """Multivariate Students-t log probability density function.
        """
        if self.df == 0 or np.isinf(self.df):
            return stats.multivariate_normal.logpdf(x, self.mean, self.cov)

        sols = sp.linalg.solve(self._cholesky, x - self.mean)
        rss = (sols ** 2).sum()

        return (self._df_terms
                - self._sum_lndiag
                - self._half_dfp * np.log1p(rss / self.df))

    def rvs(self, size):
        return self._dist.rvs(self.mean, self.cov, self.df, size)


class GIW(object):
    """Gaussian-Inverse Wishart distribution."""
    __slots__ = ['mu', 'kappa', 'nu', 'Psi',
                 '_Psi_logdet', '_cache', '_Psi_zeros']

    def __init__(self, mu, kappa, nu, Psi):
        self.mu = mu
        self.kappa = kappa
        self.nu = nu
        self.Psi = Psi
        if nu < len(Psi):
            raise ValueError('nu must be >= len(Psi)')

        # Cache log determinant of Psi
        L = sp.linalg.cholesky(Psi)
        self._Psi_logdet = 2 * np.log(np.diag(L)).sum()

        # Terms for Psi update invariant to data.
        self._cache = Psi + kappa * mu[:, None].dot(mu[None, :])
        self._Psi_zeros = np.zeros(Psi.shape)

    @property
    def d(self):
        return self.Psi.shape[0]

    def rvs(self):
        cov = stats.invwishart.rvs(self.nu, self.Psi)
        mu = stats.multivariate_normal.rvs(self.mu, cov / self.kappa)
        return mu, cov

    def copy(self):
        return GIW(self.mu, self.kappa, self.nu, self.Psi)

    def conjugate_updates(self, n, xbar, S, obj=None):
        """Return GIW with conjugate hyper-parameter updates given sufficient
        statistics.

        Args:
            n (int): Number of samples observed (rows in X).
            xbar (np.ndarray): 1 x d sample mean from data matrix X.
            S (np.ndarray): d x d sample sum of squares from X.
            obj (GIW): Update the instance variables of this object to be the
                posteriors resulting from the conjugate updates.
        """
        kappa = self.kappa + n
        nu = self.nu + n
        mu = (self.kappa * self.mu + n * xbar) / kappa
        tmp = mu[:, None].dot(mu[None, :])
        Psi = self._cache + S - kappa * tmp
        Psi = np.maximum(self._Psi_zeros, Psi)

        return GIW(mu, kappa, nu, Psi) if obj is None else \
               GIW.__init__(obj, mu, kappa, nu, Psi)

    def mean(self):
        """Return expected value for mu, Sigma."""
        Sigma = self.Psi / (self.nu - self.d - 1)
        return self.mu, Sigma


class AlphaGammaPrior(object):
    """Gamma prior distribution for Dirichlet concentration parameter alpha."""

    def __init__(self, alpha=1.0, n=1.0):
        self.alpha = alpha
        self.n = n
        self.a = 1.0
        self.b = 1.0

    def draw(self, k):
        """Draw alpha conditional on realizations of x and k."""
        x = self._draw_x(k)
        b_minus_logx = self.b - np.log(x)
        pi_1 = self.a + k - 1
        pi_2 = self.n * b_minus_logx

        return (pi_1 * stats.gamma.pdf(pi_1 + 1, b_minus_logx) +
                pi_2 * stats.gamma.pdf(pi_1, b_minus_logx))

    def _draw_x(self, k):
        """Draw x conditional on realizations of alpha and k."""
        return stats.beta.rvs(self.a + 1, self.n)

