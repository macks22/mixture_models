"""
Gibbs sampling for Finite Gaussian Mixture Model.

"""
import numpy as np
import scipy as sp
import scipy.stats as stats
import scipy.special as spsp
import scipy.cluster.vq as spvq
import matplotlib.pyplot as plt

from component import GaussianComponent


class GMM(object):
    """Finite Gaussian Mixture Model."""

    """Initialization methods supported & not yet implemented."""
    _supported = ('kmeans', 'random', 'load')
    _not_implemented = ('load',)

    @classmethod
    def supported_init_method(cls, init_method):
        return init_method in cls._supported

    @classmethod
    def implemented_init_method(cls, init_method):
        return init_method not in cls._not_implemented

    def __init__(self):
        """Initialize top-level parameters for Gaussian Mixture Model."""
        # These are the parameters that will be fit.
        self.comps = []
        self.z = []

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

    @property
    def means(self):
        return np.array([comp.mean for comp in self])

    @property
    def covs(self):
        return np.array([comp.cov for comp in self])

    def _init_comps(self, X, init_method='kmeans', iters=100):
        """Choose an initial assignment of instances to components using
        the specified init_method. This also initializes the component
        parameters based on the assigned data instances.
        """
        n, f = X.shape  # number of instances, number of features
        self.labels = np.arange(self.K)

        if init_method == 'kmeans':
            centroids, self.z = spvq.kmeans2(X, self.K, minit='points', iter=iters)
            self.comps = [GaussianComponent(X, self.z == k) for k in self.labels]
        elif init_method == 'random':
            self.z = np.random.randint(0, self.K, n)
            self.comps = [GaussianComponent(X, self.z == k) for k in self.labels]
        elif init_method == 'load':
            pass

    def init_comps(self, X, init_method='kmeans', iters=100):
        """Initialize mixture components.

        Args:
            X (np.ndarray): Data matrix with instances as rows.
            init_method (str): Method to use for initialization. One of:
                'kmeans': initialize using k-means clustering with K clusters.
                'random': randomly assign instances to components.
                'load':   load parameters from previously learned model.
            iters (int): Number of iterations to use for k-means
                initialization if init_method is 'kmeans'.
        """
        if init_method not in self._supported:
            raise ValueError(
                '%s is not a supported init method; must be one of: %s' % (
                    init_method, ', '.join(supported)))

        if init_method in self._not_implemented:
            raise NotImplemented(
                '%s initialization not yet implemented' % init_method)

        return self._init_comps(X, init_method, iters)

    def fit(self, X, K, alpha=0.0, init_method='kmeans', iters=100,
            nsamples=220, burnin=20, thin_step=2):
        """Fit the parameters of the model using the data X.
        See `init_comps` method for info on parameters not listed here.

        Args:
            X (np.ndarray): Data matrix with instances as rows.
            K (int): Fixed number of components.
            alpha (float): Dirichlet hyper-parameter alpha; defaults to K.
            nsamples (int): Number of Gibbs samples to draw.
            burnin (int): Number of Gibbs samples to discard.
            thin_step (int): Stepsize for thinning to reduce autocorrelation.
        """
        self.K = K
        self.init_comps(X, init_method, iters)

        n, f = X.shape
        self.alpha = alpha if alpha else float(K)

        self.nsamples = nsamples
        self.burnin = burnin
        self.thin_step = thin_step

        # Set alpha to K by default if not given.
        # Setting to K makes the Dirichlet uniform over the components.
        K = self.K
        alpha = self.alpha
        alpha_k = self.alpha / K

        # We'll use this for our conditional multinomial probs.
        Pk = np.ndarray(K)
        denom = float(n + alpha - 1)

        # Init trace vars for parameters.
        keeping = self.nsamples - self.burnin
        store = int(keeping / 2)

        pi = np.zeros((store, K))  # mixture weights
        H_ik = np.zeros((store, n, K))  # posterior mixture responsibilities
        mu = np.zeros((store, K, f))  # component means
        Sigma = np.zeros((store, K, f, f))  # component covariance matrices
        ll = np.zeros(store)  # log-likelihood at each step

        print('initial log-likelihood: %.3f' % self.llikelihood())

        # Run collapsed Gibbs sampling to fit the model.
        indices = np.arange(n)
        for iternum in range(self.nsamples):
            logging_iter = iternum % self.thin_step == 0
            saving_sample = logging_iter and (iternum >= self.burnin - 1)
            if saving_sample:
                idx = (iternum - self.burnin) / self.thin_step

            for i in np.random.permutation(indices):
                x = X[i]

                # Remove X[i]'s stats from component z[i].
                old_k = self.z[i]
                self.comps[old_k].rm_instance(i)

                # Calculate probability of instance belonging to each comp.
                # Calculate P(z[i] = k | z[-i], alpha)
                weights = (self.counts + alpha_k) / denom

                # Calculate P(X[i] | X[-i], pi, mu, sigma)
                probs = np.array([comp.pp.pdf(x) for comp in self.comps])

                # Calculate P(z[i] = k | z[-i], X, alpha, pi, mu, Sigma)
                Pk[:] = probs * weights

                # Normalize probabilities.
                Pk = Pk / Pk.sum()

                # Sample new component for X[i] using normalized probs.
                new_k = np.nonzero(np.random.multinomial(1, Pk))[0][0]

                # Add X[i] to selected component. Sufficient stats are updated.
                self.comps[new_k].add_instance(i)
                self.z[i] = new_k

                # save posterior responsibility each component takes for
                # explaining data instance i
                if saving_sample:
                    H_ik[idx, i] = Pk

            if logging_iter:
                llik = self.llikelihood()
                print('sample %d, log-likelihood: %.3f' % (iternum, llik))

                if saving_sample:
                    pi[idx] = Pk
                    stats = [comp.posterior.rvs() for comp in self.comps]
                    mu[idx] = np.r_[[stat[0] for stat in stats]]
                    Sigma[idx] = np.r_[[stat[1] for stat in stats]]
                    ll[idx] = llik

        return ll, H_ik, pi, mu, Sigma

    def label_llikelihood(self):
        """Calculate ln(P(z | alpha)), the marginal log-likelihood of the
        component instance assignments.

        Eq. (22) from Kamper's guide.
        """
        alpha = self.alpha
        alpha_k = alpha / self.K

        llik = spsp.gammaln(alpha) - spsp.gammaln(self.n + alpha)
        counts = self.counts.astype(np.float) + alpha_k
        llik += (spsp.gammaln(counts) - spsp.gammaln(alpha_k)).sum()
        return llik

    def label_likelihood(self):
        """Calculate P(z | alpha), the marginal likelihood of the component
        instance assignments.
        """
        return np.exp(self.label_llikelihood())

    def llikelihood(self):
        """Calculate ln(P(X, z | alpha, pi, mu, Sigma)), the marginal
        log-likelihood of the data and the component instance assignments given
        the parameters and hyper-parameters.

        This can be used as a convergence metric. We expect to see an
        increase as sampling proceeds.

        Eq. (29) from Kamper's guide.
        """
        llik = self.label_llikelihood()
        llik += np.sum([comp.llikelihood() for comp in self])
        return llik

    def likelihood(self):
        """Calculate P(X, z | alpha, pi, mu, Sigma), the marginal likelihood
        of the data and the component instance assignments given the parameters
        and hyper-parameters.
        """
        return np.exp(self.llikelihood())


if __name__ == "__main__":
    np.random.seed(1234)
    np.seterr('raise')

    M = 200            # number of samples per component
    K = 2              # initial guess for number of clusters
    method = 'kmeans'  # parameter initialization method

    # Generate two 2D Gaussians
    X = np.r_[
        stats.multivariate_normal.rvs([-5, -7], 2, M),
        stats.multivariate_normal.rvs([2, 4], 4, M)
    ]
    true_z = np.concatenate([[k] * M for k in range(K)])

    # Initialize and fit Gaussian Mixture Model.
    gmm = GMM()
    ll, H_ik, pi, mu, Sigma = gmm.fit(
        X, K, init_method=method, nsamples=100, burnin=10)

    # Calculate expectations using Monte Carlo estimates.
    E = {
        'H_ik': H_ik.mean(0),
        'pi': pi.mean(0),
        'mu': mu.mean(0),
        'Sigma': Sigma.mean(0)
    }
