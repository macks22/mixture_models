import numpy as np
import scipy as sp
import scipy.stats as stats
import scipy.special as spsp


class multivariate_t_gen(stats.rv_continuous):

    def _logpdf(self, x, mean, cov, df):
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
        return self._pdf(x, mean, cov, df)

    def _rvs(self, mean, cov, df, size):
        """Produce random variates from the multivariate Students-t
        distribution.

        Args:
            mean (np.ndarray): 1 x p mean vector of the Students-t.
            cov (np.ndarray): p x p covariance matrix of the Students-t.
            df (int): Scalar degrees of freedom of the Students-t.

        """
        d = mean.shape[0]
        g = np.tile(np.random.gamma(df / 2., 2. / df, size), (d, 1)).T
        Z = np.random.multivariate_normal(np.zeros(d), cov, df)
        return mu + Z / np.sqrt(g)

multivariate_t = multivariate_t_gen(name='multivariate_t', shapes='mean, cov, df')
