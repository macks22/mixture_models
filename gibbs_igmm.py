"""
Gibbs sampling for Infinite Gaussian Mixture Model.

"""
import numpy as np
import scipy as sp
import scipy.stats as stats
import scipy.special as spsp
import scipy.cluster.vq as spvq
import matplotlib.pyplot as plt

from gibbs_gmm import GMM
from component import GaussianComponent
from distributions import AlphaGammaPrior


class IGMM(GMM):
    """Infinite Gaussian Mixture Model using Dirichlet Process prior."""

    def __init__(self):
        """Initialize top-level parameters for Gaussian Mixture Model."""
        # These are the parameters that will be fit.
        self.alpha_prior = None
        self.K = None

        self.labels = []
        self.comps = {}
        self.z = np.array([])

    def __iter__(self):
        return (comp for comp in self.comps.values())

    @property
    def alpha(self):
        return None if self.alpha_prior is None else self.alpha_prior.alpha

    def _next_k(self):
        """Return the label of the next empty component."""
        labels = self.comps.keys()
        for k, i in zip(labels, xrange(self.K)):
            if i != k:
                return i
        return k + 1

    def _add_component(self, comp, new_k):
        self.comps[new_k] = comp
        self.K += 1
        self.labels = self.comps.keys()

    def add_component(self, comp):
        """Place the new component in an empty space in the components array and 
        increment K. We assume the assignments z has already been updated.
        """
        new_k = self._next_k()
        self._add_component(comp, new_k)
        return new_k

    def add_component_with_instance(self, X, i):
        """Add new component containing X[i]."""
        new_k = self._next_k()
        self.z[i] = new_k
        self._add_component(GaussianComponent(X, self.z == new_k), new_k)
        return new_k

    def rm_component(self, k):
        """Remove component k from components being tracked."""
        try:
            del self.comps[k]
            self.K -= 1
            self.labels = self.comps.keys()
        except KeyError:
            raise KeyError(
                "trying to remove component %d but it's not being tracked" % k)

    def _init_comps(self, X, init_method='kmeans', iters=100):
        """Choose an initial assignment of instances to components using
        the specified init_method. This also initializes the component
        parameters based on the assigned data instances.
        """
        n, f = X.shape  # number of instances, number of features
        self.labels = np.arange(self.K)

        if init_method == 'kmeans':
            centroids, self.z = \
                spvq.kmeans2(X, self.K, minit='points', iter=iters)
            self.comps = {
                k: GaussianComponent(X, self.z == k) for k in self.labels}

        elif init_method == 'random':
            self.z = np.random.randint(0, self.K, n)
            self.comps = {
                k: GaussianComponent(X, self.z == k) for k in self.labels}

        elif init_method == 'load':
            pass

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
        self.alpha_prior = AlphaGammaPrior(n, alpha if alpha else float(K))

        self.nsamples = nsamples
        self.burnin = burnin
        self.thin_step = thin_step

        # Set alpha to K by default if not given.
        # Setting to K makes the Dirichlet uniform over the components.
        alpha = self.alpha
        alpha_k = self.alpha / self.K
        base_distrib = GaussianComponent(X, np.zeros(n).astype(bool))

        # We'll use this for our conditional multinomial probs.
        Pk = np.zeros(self.K * 2)
        denom = float(n + alpha - 1)

        # Init trace vars for parameters.
        keeping = self.nsamples - self.burnin
        store = int(keeping / 2)
        alphas = np.ndarray(store, float)  # concentration parameter
        pi = np.ndarray(store, object)     # mixture weights
        H_ik = np.ndarray((store, n), object)  # posterior mixture responsibilities
        mu = np.ndarray(store, object)     # component means
        Sigma = np.ndarray(store, object)  # component covariance matrices
        ll = np.ndarray(store, float)      # log-likelihood at each step

        print('initial log-likelihood: %.3f' % self.llikelihood())

        # Run collapsed Gibbs sampling to fit the model.
        indices = np.arange(n)
        for iternum in range(self.nsamples):
            logging_iter = iternum % self.thin_step == 0
            saving_sample = logging_iter and (iternum >= self.burnin - 1)
            if saving_sample:
                idx = (iternum - self.burnin) / self.thin_step

            # sample alpha
            alpha = self.alpha_prior.draw(self.K)
            alpha_k = alpha / self.K

            for i in np.random.permutation(indices):
                x = X[i]

                # Remove X[i]'s stats from component z[i].
                old_k = self.z[i]
                old_comp = self.comps[old_k]
                old_comp.rm_instance(i)

                # The component may now be empty, but we keep it since it has
                # cached stats. It will have same probability as base
                # distribution, so we may be able to reuse it if the base
                # distribution is selected.
                if old_comp.is_empty:
                    self.rm_component(old_k)

                # Calculate probability of instance belonging to each
                # currently non-empty component.
                # Calculate P(z[i] = k | z[-i], alpha)
                weights = self.counts / denom

                # Calculate P(X[i] | X[-i], pi, mu, sigma)
                probs = np.array([
                    comp.pp.pdf(x) for comp in self.comps.values()])

                # Calculate P(z[i] = k | z[-i], X, alpha, pi, mu, Sigma)
                Pk[:self.K] = probs * weights

                # Calculate probability of instance belonging to new
                # component k*.
                prior_predictive = base_distrib.pp.pdf(x)
                weight = alpha / denom
                Pk[self.K] = prior_predictive * weight

                # Normalize probabilities.
                next_k = self.K + 1
                Pk[:next_k] = Pk[:next_k] / Pk[:next_k].sum()

                # Sample new component for X[i] using normalized probs.
                _new_k = np.nonzero(np.random.multinomial(1, Pk[:next_k]))[0][0]

                # If new component was selected, add to tracked components.
                if _new_k == self.K:
                    # reuse old component if it was removed.
                    if old_comp.is_empty:
                        old_comp.add_instance(i)
                        new_k = self.add_component(old_comp)
                        self.z[i] = new_k
                    else:
                        self.add_component_with_instance(X, i)

                    # Resize Pk if necessary for new component
                    if self.K >= Pk.shape[0]:
                        Pk = np.resize(Pk, Pk.shape[0] * 2)

                else:  # Old component selected, add data instance to it.
                    new_k = self.labels[_new_k]  # non-contiguous labels
                    self.comps[new_k].add_instance(i)
                    self.z[i] = new_k

                # save posterior responsibility each component takes for
                # explaining data instance i
                if saving_sample:
                    H_ik[idx, i] = Pk

            if logging_iter:
                llik = self.llikelihood()
                print('sample %d, %d comps, log-likelihood: %.3f' % (
                    iternum, self.K, llik))

                if saving_sample:
                    alphas[idx] = alpha
                    pi[idx] = Pk[:next_k].copy()
                    stats = \
                        [comp.posterior.rvs() for comp in self.comps.values()]
                    mu[idx] = np.r_[[stat[0] for stat in stats]]
                    Sigma[idx] = np.r_[[stat[1] for stat in stats]]
                    ll[idx] = llik

        print('sample %d, %d comps, log-likelihood: %.3f' % (
            iternum, self.K, llik))

        return ll, alphas, H_ik, pi, mu, Sigma

    def label_llikelihood(self):
        """Calculate ln(P(z | alpha)), the marginal log-likelihood of the
        component instance assignments.

        Eq. (34) from Kamper's guide.
        """
        return (self.K * np.log(self.alpha)
                + spsp.gammaln(self.counts).sum()
                - np.log(np.arange(self.n) + self.alpha).sum())


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

    igmm = IGMM()
    ll, alphas, H_ik, pi, mu, Sigma = igmm.fit(
        X, K, alpha=1.0, init_method=method, nsamples=100, burnin=10)
