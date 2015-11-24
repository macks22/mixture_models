"""
Gibbs sampling for Gaussian Mixture Model.

"""
import numpy as np
import scipy as sp
import scipy.stats as stats
import scipy.special as spsp
import scipy.cluster.vq as spvq
import matplotlib.pyplot as plt


def precision(X):
    return np.linalg.inv(np.cov(X, rowvar=False))

def mvstdt_pdf(x, mean, cov, df):
    """Multivariate Students-t probability density function.

    From Wikipedia: https://en.wikipedia.org/wiki/Multivariate_t-distribution.

    Args:
        x (np.ndarray): 1 x p data vector observation.
        mean (np.ndarray): 1 x p mean vector of the Students-t.
        cov (np.ndarray): p x p covariance matrix of the Students-t.
        df (int): Scalar degrees of freedom of the Students-t.

    Returns:
        (float): The probability of observing x from a multivariate Students-t
            distribution with the given parameters.
    """
    p = len(x)  # dimension, should match mean and cov
    half_dfp = (df + p) * 0.5
    num = spsp.gamma(half_dfp)

    half_p = p * 0.5
    denom = spsp.gamma(df * 0.5) * ((df * np.pi) ** half_p)
    denom *= np.linalg.det(cov) ** 0.5

    dev = x - mean
    tmp = dev.dot(np.linalg.inv(cov)).dot(dev)
    denom *= (1 + (1. / df) * tmp) ** half_dfp

    return num / denom


def mvstdt_logpdf(x, mean, cov, df):
    """Multivariate Students-t log probability density function.

    From Wikipedia: https://en.wikipedia.org/wiki/Multivariate_t-distribution.

    Args:
        x (np.ndarray): 1 x p data vector observation.
        mean (np.ndarray): 1 x p mean vector of the Students-t.
        cov (np.ndarray): p x p covariance matrix of the Students-t.
        df (int): Scalar degrees of freedom of the Students-t.

    Returns:
        (float): The probability of observing x from a multivariate Students-t
            distribution with the given parameters.
    """
    p = len(x)
    dev = x - mean
    r = dev.dot(np.linalg.inv(cov)).dot(dev)
    half_dfp = (df + p) * 0.5
    half_p = p * 0.5
    logdf = np.log(df)
    return (spsp.gammaln(half_dfp) - spsp.gammaln(half_p)
            - half_p * (logdf + np.log(np.pi))
            - 0.5 * np.log(np.linalg.det(cov))
            - half_dfp * (np.log(r) - logdf))


class Gaussian(object):
    """Multivariate Gaussian distribution; for use as mixture component."""

    def __init__(self, X, instances):
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
        self.instances = instances
        self.X = X

        # Init mu ~ Normal hyperparams.
        self.mu0 = self._xbar
        self.k0 = 1.0

        # Init Sigma ~ Inv-Wishart hyperparams.
        self.nu0 = self.nf + 2
        dev = X - self._xbar
        self.Psi0 = dev.T.dot(dev)

        # The conjugate udpate for S uses a term that only depends on
        # these hyper-parameters. Cache that now for efficiency.
        # self._prior_sqmean = self.k0 * self.mu0[:, None].dot(self.mu0[None, :])
        # self._prior_sqmean += self.Psi0

        # The precision matrix is the inverse of the covariance.
        # Whenever it is asked for, we'll need to get the inverse of
        # the current covariance matrix. We use a hash to avoid
        # recomputation as long as the covariance matrix is the same.
        self._cov_hash = 0

        # Fit params to the data.
        self.fit()

    @property
    def X(self):
        return self._X[self.instances]

    @X.setter
    def X(self, X):
        self._X = X

        # Cache stats used during fitting.
        n = self.n
        # self._S = self.X.T.dot(self.X) # sample sum of squares
        self._xbar = self.X.sum(0) if n else np.zeros(self.nf) # sample mean

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

    def add_instance(self, i):
        """Add an instance to this Gaussian component.
        This is done by setting element i of the `instances` mask to True.
        """
        if self.instances[i]:  # already in component
            return

        # Add sufficient stats to the cached stats.
        # x = self._X[i]
        # self._S += x[:, None].dot(x[None, :])
        # self._xbar = (self._xbar * self.n + x) / (self.n + 1)

        self.instances[i] = True
        self.fit()

    def rm_instance(self, i):
        if not self.instances[i]:
            raise IndexError('index %i not currently in component' % i)

        # Remove sufficient stats from cached stats.
        # x = self._X[i]
        # self._S -= x[:, None].dot(x[None, :])
        # new_n = self.n - 1
        # if new_n == 0:
        #     self._xbar[:] = 0
        # else:
        #     self._xbar = (self._xbar * self.n - x) / new_n

        self.instances[i] = False  # remove instance
        self.fit()  # handles empty component case

    def fit(self):
        """Perform conjugate updates of the GIW prior parameters to compute the
        posterior mean and convaraince. We also compute the posterior predictive
        mean and covariance for use in the students-t posterior predictive pdf.

        Eqs. (8) and (15) in Kamper's guide.
        """
        # Conjugate hyper-parameter updates.
        n = self.n
        kappa = self.k0 + n
        nu = self.nu0 + n

        self._xbar = self.X.sum(0)
        mu = (self.k0 * self.mu0 + n * self._xbar) / kappa

        dev = self.X - self._xbar
        S = dev.T.dot(dev)
        dev = self._xbar - self.mu0
        Psi = (self.Psi0 + S
                + ((self.k0 * n) / kappa)
                  * dev[:, None].dot(dev[None, :]))

        # Psi = (self._S + self._prior_sqmean
        #         - kappa * mu[:, None].dot(mu[None, :]))

        # Posterior parameter calculations.
        self.cov = stats.invwishart.rvs(nu, Psi)
        self.mean = stats.multivariate_normal.rvs(mu, self.cov / kappa)

        # Posterior predictive parameter calculations.
        # Eq. (15) from Kamper's guide.
        self.pp_mean = mu
        self.pp_df = nu - self.nf + 1
        self.pp_cov = (Psi * (kappa + 1)) / (kappa * self.pp_df)

        # Save conjugate updated hyper-params for marginal likelihood calcs.
        self._kappa = kappa
        self._nu = nu
        self._Psi = Psi

    # def pdf(self, x):
    #     """Multivariate normal probability density function."""
    #     return stats.multivariate_normal.pdf(x, self.mean, self.cov)

    def likelihood(self):
        """Compute marginal likelihood of data given the observed
        data instances assigned to this component.

        Eq. (266) from Murphy (2007).
        """
        half_d = self.nf * 0.5
        nu_prior = self.nu0 * 0.5
        nu_post = self._nu * 0.5

        lik = (np.pi ** -(self.n * half_d)
                * (spsp.gamma(nu_post) / spsp.gamma(nu_prior))
                * ((np.linalg.det(self.Psi0) ** nu_prior)
                    / (np.linalg.det(self._Psi) ** nu_post))
                * ((self.k0 / self._kappa) ** half_d))
        return lik

    def llikelihood(self):
        """Compute marginal log likelihood of data given the observed
        data instances assigned to this component.

        Eq. (266) from Murphy (2007).
        """
        half_d = self.nf * 0.5
        nu_prior = self.nu0 * 0.5
        nu_post = self._nu * 0.5

        return (spsp.gammaln(nu_post) - spsp.gammaln(nu_prior)
                + nu_prior * np.log(np.linalg.det(self.Psi0))
                - nu_post * np.log(np.linalg.det(self._Psi))
                + half_d * (np.log(self.k0) - np.log(self._kappa))
                - (self.n * half_d) * np.log(np.pi))

    def pp_pdf(self, x):
        """Students-t posterior predictive density function."""
        return mvstdt_pdf(x, self.pp_mean, self.pp_cov, self.pp_df)

    def pp_logpdf(self, x):
        """Students-t posterior predictive log density function."""
        return mvstdt_logpdf(x, self.pp_mean, self.pp_cov, self.pp_df)

    def pp_rvs(self, size):
        """Samples from Students-t posterior predictive distribution."""
        raise NotImplemented(
            'random variables from posterior predictive not implemented')


class GMM(object):
    """Finite Gaussian Mixture Model."""

    def __init__(self, K, nsamples=220, burnin=20, thin_step=2):
        """Initialize top-level parameters for Gaussian Mixture Model.

        Args:
            K (int): Fixed number of components.
            nsamples (int): Number of Gibbs samples to draw.
            burnin (int): Number of Gibbs samples to discard.
            thin_step (int): Stepsize for thinning to reduce autocorrelation.
        """
        self.K = K
        self.nsamples = nsamples
        self.burnin = burnin
        self.thin_step = thin_step

        # These are the parameters that will be fit.
        self.comps = None
        self.z = None

    @property
    def n(self):
        if self.comps is None:
            return 0
        else:
            return sum(comp.n for comp in self.comps.values())

    @property
    def nf(self):
        if self.comps is None:
            return 0
        else:
            return self.comps.values()[0].nf

    def fit(self, X, alpha=0.0, init_method='kmeans', iters=100):
        """Fit the parameters of the model using the data X.

        Args:
            X (np.ndarray): Data matrix with instances as rows.
            alpha (float): Dirichlet hyper-parameter alpha.
            init_method (str): Method to use for initialization. One of:
                'kmeans': initialize using k-means clustering with K clusters.
                'random': randomly assign instances to components.
                'load':   load parameters from previously learned model.
            iters (int): Number of iterations to use for k-means
                initialization if init_method is 'kmeans'.
        """
        supported = ['kmeans', 'random', 'load']
        if init_method not in supported:
            raise ValueError(
                '%s is not a supported init method; must be one of: %s' % (
                    init_method, ', '.join(supported)))

        not_implemented = ['load']
        if init_method in not_implemented:
            raise NotImplemented(
                '%s initialization not yet implemented' % init_method)

        # Choose an initial assignment of instances to components using
        # the specified init_method. This also initializes the component
        # parameters based on the assigned data instances.
        n, f = X.shape  # number of instances, number of features
        K = self.K
        comp_labels = np.arange(K)

        if init_method == 'kmeans':
            centroids, self.z = spvq.kmeans2(X, K, minit='points', iter=iters)
            self.comps = {k: Gaussian(X, self.z == k) for k in comp_labels}
        elif init_method == 'random':
            self.z = np.random.randint(0, K, n)
            self.comps = {k: Gaussian(X, self.z == k) for k in comp_labels}
        elif init_method == 'load':
            pass

        # Set alpha to K by default if not given.
        # Setting to K makes the Dirichlet uniform over the components.
        self.alpha = alpha if alpha else float(K)
        alpha = self.alpha
        alpha_k = self.alpha / K

        # We'll use this for our conditional multinomial probs.
        Pk = np.ndarray(K)
        denom = float(n + alpha - 1)

        # Init trace vars for parameters.
        keeping = self.nsamples - self.burnin
        store = int(keeping / 2)
        pi = np.zeros((store, K))
        mu = np.zeros((store, K, f))
        Sigma = np.zeros((store, K, f, f))
        ll = np.zeros(store)

        print('initial log-likelihood: %.3f' % self.llikelihood())

        # Run collapsed Gibbs sampling to fit the model.
        indices = np.arange(n)
        for iternum in range(self.nsamples):
            for i in np.random.permutation(indices):
                x = X[i]

                # Remove X[i]'s stats from component z[i].
                k = self.z[i]
                self.comps[k].rm_instance(i)

                # Calculate probability of instance belonging to each comp.
                for k in comp_labels:
                    comp = self.comps[k]

                    # Calculate P(z[i] = k | z[-i], alpha)
                    weight = (comp.n + alpha_k) / denom

                    # Calculate P(X[i] | X[-i], pi, mu, sigma)
                    prob = comp.pp_pdf(x)  # posterior predictive pdf
                    # prob = comp.pp_logpdf(x)

                    # print('k: %d, w=%.3e, p=%.3e' % (k, weight, prob))

                    # Calculate P(z[i] = k | z[-i], X, alpha, pi, mu, Sigma)
                    # Pk[k] = prob * weight
                    Pk[k] = prob + np.log(weight)

                # Normalize probabilities.
                # Pk[np.isnan(Pk)] = 0
                # Pk = abs(Pk)  # TEMP
                Pk = Pk / Pk.sum()
                # print(Pk)

                # Sample new component for X[i] using normalized probs.
                k = np.random.choice(comp_labels, p=Pk)

                # Add X[i] to selected component. Sufficient stats are updated.
                self.comps[k].add_instance(i)
                self.z[i] = k


            if iternum % self.thin_step == 0:
                llik = self.llikelihood()
                print('sample %d, log-likelihood: %.3f' % (iternum, llik))

                if iternum >= (self.burnin - 1):
                    i = (iternum - self.burnin) / self.thin_step
                    pi[i] = Pk
                    comps = [comp for comp in self.comps.values()]
                    mu[i] = np.r_[[comp.mean for comp in comps]]
                    Sigma[i] = np.r_[[comp.cov for comp in comps]]
                    ll[i] = llik

        return ll, pi, mu, Sigma

    def label_likelihood(self):
        """Calculate P(z | alpha), the marginal likelihood of the component
        instance assignments.

        Eq. (22) from Kamper's guide.
        """
        alpha = self.alpha
        alpha_k = alpha / self.K
        lik = spsp.gamma(alpha) / spsp.gamma(self.n + alpha)

        gam_alpha_k = spsp.gamma(alpha_k)
        numerators = np.array([comp.n for comp in self.comps.values()])\
                        .astype(np.float) + alpha_k
        lik *= np.product(spsp.gamma(numerators))
        return lik

    def label_llikelihood(self):
        """Calculate ln(P(z | alpha)), the marginal log-likelihood of the
        component instance assignments.

        Eq. (22) from Kamper's guide.
        """
        alpha = self.alpha
        alpha_k = alpha / self.K

        llik = spsp.gammaln(alpha) - spsp.gammaln(self.n + alpha)
        counts = np.array([comp.n for comp in self.comps.values()])\
                   .astype(np.float) + alpha_k
        llik += (spsp.gammaln(counts) - spsp.gammaln(alpha_k)).sum()
        return llik

    def likelihood(self):
        """Calculate P(X, z | alpha, pi, mu, Sigma), the marginal likelihood
        of the data and the component instance assignments given the parameters
        and hyper-parameters.

        This can be used as a convergence metric. We expect to see this
        likelihood increase as sampling proceeds.

        Eq. (29) from Kamper's guide.
        """
        lik = self.label_likelihood()
        lik *= np.product([
            comp.likelihood() for comp in self.comps.values()])
        return lik

    def llikelihood(self):
        """Calculate ln(P(X, z | alpha, pi, mu, Sigma)), the marginal
        log-likelihood of the data and the component instance assignments given
        the parameters and hyper-parameters.

        This can be used as a convergence metric. We expect to see an
        increase as sampling proceeds.

        Eq. (29) from Kamper's guide.
        """
        llik = self.label_llikelihood()
        llik += np.sum([comp.llikelihood() for comp in self.comps.values()])
        return llik


if __name__ == "__main__":
    np.random.seed(1234)

    M = 50           # number of samples per component
    K = 2            # initial guess for number of clusters
    method = 'kmeans'  # parameter initialization method

    # Generate two 2D Gaussians
    X = np.r_[
        stats.multivariate_normal.rvs([-5, -7], 2, M),
        stats.multivariate_normal.rvs([5, 7], 4, M)
    ]
    true_z = np.concatenate([[k] * M for k in range(K)])
    n, nf = X.shape  # number of instances, number of features

    gmm = GMM(K)
    ll, pi, mu, Sigma = gmm.fit(X, init_method=method)
