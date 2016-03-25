# encoding: utf-8

import numpy as np
import scipy as sp
import scipy.special as spsp
import scipy.stats as stats
from scipy.linalg import blas

cimport cython
cimport numpy as np


def mglr_fit_pp(np.ndarray[np.double_t, ndim=2] X,
                np.ndarray[np.double_t, ndim=2] post_V,
                double post_b,
                double post_a,
                np.ndarray[np.double_t, ndim=1] post_mu):

    cdef np.ndarray[np.double_t, ndim=2] pp_cov
    cdef np.ndarray[np.double_t, ndim=1] mean
    cdef double df, b_over_a
    cdef unsigned int f

    mean = np.dot(X, post_mu)  # n x 1
    # mean = blas.dgemv(1., X, post_mu)  # n x 1
    df = post_a * 2
    b_over_a = post_b / post_a
    f = mean.shape[0]
    pp_cov = np.dot(np.dot(X, post_V), X.T)
    # pp_cov = blas.dsymm(1., post_V, X.T)
    # pp_cov = blas.dgemm(1., X, pp_cov)
    pp_cov += b_over_a * np.eye(f)

    return mean, pp_cov, df

