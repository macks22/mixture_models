# encoding: utf-8

import numpy as np
import scipy as sp
import scipy.special as spsp
import scipy.stats as stats

cimport cython
cimport numpy as np


def set_cov(np.ndarray[np.double_t, ndim=1] mean,
            np.ndarray[np.double_t, ndim=2] cov):
    cdef unsigned int f, mu_f
    cdef double eps, sum_lndiag
    cdef np.ndarray[np.double_t, ndim=2] adjustment
    cdef np.ndarray[np.double_t, ndim=2] cholesky

    f = cov.shape[0]
    mu_f = mean.shape[0]
    if mu_f != f:
        raise ValueError(
            'incompatible shape: mean.shape[0] != cov.shape[0] '
            '(%d != %d)' % (mu_f, f))

    eps = np.finfo(np.float32).eps
    adjustment = np.eye(f)

    cholesky = sp.linalg.cholesky(cov + adjustment)
    sum_lndiag = np.sum(np.log(np.diag(cholesky)))
    return cholesky, sum_lndiag


def set_df(np.ndarray[np.double_t, ndim=1] mean,
           unsigned int df):
    cdef unsigned int p = mean.shape[0]
    cdef double half_dfp, df_terms

    if df < p:
        df = p

    half_dfp = (df + p) / 2.
    df_terms = (spsp.gammaln(half_dfp)
                - spsp.gammaln(df / 2.)
                - (p / 2.) * np.log(np.pi * df))
    return half_dfp, df_terms


def mvt_logpdf(np.ndarray[np.double_t, ndim=1] x,
               np.ndarray[np.double_t, ndim=1] mean,
               np.ndarray[np.double_t, ndim=2] cov,
               np.ndarray[np.double_t, ndim=2] cholesky,
               double sum_lndiag,
               double df,
               double df_terms,
               double half_dfp):
    """Multivariate Students-t log probability density function."""
    cdef np.ndarray[np.double_t, ndim=1] sols
    cdef double rss, prob

    if df == 0 or np.isinf(df):
        return stats.multivariate_normal.logpdf(x, mean, cov)

    sols = sp.linalg.solve(cholesky, x - mean)
    rss = np.sum(sols ** 2)
    prob = df_terms - sum_lndiag - half_dfp * np.log1p(rss / df)
    return prob

