"""
Gibbs sampling for Infinite Gaussian Mixture Model.

"""
import logging

import numpy as np
import scipy as sp
import scipy.stats as stats
import scipy.special as spsp
import scipy.cluster.vq as spvq
import matplotlib.pyplot as plt

import gibbs_gmm
from component import GaussianComponent
from distributions import AlphaGammaPrior


class IGMMTrace(object):
    __slots__ = ['ll', 'alpha', 'pi', 'mu', 'Sigma', 'H_ik']

    def __init__(self, nsamples, n, f):
        """Initialize empty storage for n samples of each parameter.
        Each parameter is stored in an ndarray with the first index being for
        the sample number.

        Args:
            nsamples (int): Number of Gibbs samples to store for each parameter.
            n (int): Number of data instances being learned on.
            f (int): Number features the model is being learned on.

        """
        self.ll = np.ndarray((nsamples,), float)      # log-likelihood at each step
        self.alpha = np.ndarray((nsamples,), float)  # concentration parameter
        self.pi = np.ndarray((nsamples,), object)     # mixture weights
        self.mu = np.ndarray((nsamples,), object)     # component means
        self.Sigma = np.ndarray((nsamples,), object)  # component covariance matrices
        self.H_ik = np.ndarray((nsamples, n), object) # posterior mixture memberships

    def expectation(self):
        labels = set(key for d in self.mu for key in d.keys())
        max_k = len(labels)
        lmap = dict(zip(np.arange(max_k), labels))

        nsamples = self.ll.shape[0]
        n = self.H_ik.shape[1]
        f = self.mu[0].values()[0].shape[0]

        E = {
            'alpha': self.alpha.mean(),
            'H_ik':  np.ndarray((n, max_k)),
            'pi':    np.ndarray((max_k,)),
            'mu':    np.ndarray((max_k, f)),
            'Sigma': np.ndarray((max_k, f, f))
        }

        indices = np.arange(nsamples)
        mu_zeros = np.zeros(f)
        Sigma_zeros = np.zeros((f, f))
        for i, k in lmap.items():
            E['pi'][i] = \
                sum(self.pi[idx].get(k, 0.) for idx in indices) / nsamples
            E['mu'][i] = \
                sum(self.mu[idx].get(k, mu_zeros) for idx in indices) / nsamples
            E['Sigma'][i] = \
                sum(self.Sigma[idx].get(k, Sigma_zeros) for idx in indices)\
                / nsamples

            for _i in range(n):
                E['H_ik'][_i, i] = \
                    sum(self.H_ik[idx, _i].get(k, 0.) for idx in indices)\
                    / nsamples
        return E


class IGMM(gibbs_gmm.GMM):
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
        store = int(keeping / self.thin_step)

        trace = IGMMTrace(store, n, f)
        logging.info('initial log-likelihood: %.3f' % self.llikelihood())

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
                    trace.H_ik[idx, i] = dict(zip(self.labels, Pk[:next_k]))

            if logging_iter:
                llik = self.llikelihood()
                logging.info('sample %d, %d comps, log-likelihood: %.3f' % (
                    iternum, self.K, llik))

                if saving_sample:
                    trace.alpha[idx] = alpha
                    trace.pi[idx] = dict(zip(self.labels, Pk[:next_k]))
                    mus, Sigmas = self.posterior_rvs()
                    trace.mu[idx] = dict(zip(self.labels, mus))
                    trace.Sigma[idx] = dict(zip(self.labels, Sigmas))
                    trace.ll[idx] = llik

        logging.info('sample %d, %d comps, log-likelihood: %.3f' % (
            iternum, self.K, llik))

        return trace

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

    args = gibbs_gmm.parse_args(
        'Infer igmm model parameters for generated data.')

    # Generate two 2D Gaussians
    M = args.samples_per_comp  # number of samples per component
    K = args.K  # initial guess for number of clusters
    X = np.r_[
        stats.multivariate_normal.rvs([-5, -7], 2, M),
        stats.multivariate_normal.rvs([4, 8], 4, M),
        stats.multivariate_normal.rvs([0, 0], np.diag([1, 2]), M)
    ]
    true_z = np.concatenate([[k] * M for k in range(K)])

    igmm = IGMM()
    trace = igmm.fit(
        X, K, alpha=1.0, init_method=args.init_method, nsamples=args.nsamples,
        burnin=args.burnin, thin_step=args.thin_step)

    # Calculate expectations
    E = trace.expectation()
