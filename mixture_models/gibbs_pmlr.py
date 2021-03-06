"""
Gibbs sampler for the PMLR (Profiling Mixture of Linear Regressions) model
without bias terms.

"""
import argparse
import logging

import numpy as np
import scipy as sp
import scipy.cluster.vq as spvq
import scipy.special as spsp

from mixture import MixtureModel, MGLRComponent, AlphaGammaPrior
from mixture import cli, gendata


class PMLRTrace(object):
    __slots__ = ['ll', 'pi', 'W', 'sigma', 'H']

    def __init__(self, nsamples, n, f, K):
        """Initialize empty storage for Gibbs samples of each parameter.
        Each parameter is stored in an ndarray with the first index being for
        the sample number.

        Args:
            nsamples (int): Number of Gibbs samples to store for each parameter.
            n (int): Number of unique primary entities.
            f (int): Number features the model is being learned on.
            K (int): Number of clusters.

        """
        self.ll = np.zeros((nsamples,))        # log-likelihood at each step
        self.pi = np.zeros((nsamples, K))      # mixture weights
        self.W = np.zeros((nsamples, K, f))   # component reg coefficients
        self.sigma = np.zeros((nsamples, K))   # component reg variances
        self.H = np.zeros((nsamples, n, K)) # posterior mixture memberships

    def expectation(self):
        return {name: getattr(self, name).mean(0) for name in self.__slots__}


class PMLR(MixtureModel):
    """Profiling Mixture of Linear Regressions with GIG prior."""

    """Initialization methods supported & not yet implemented."""
    _supported = ('kmeans', 'em', 'random', 'load')
    _not_implemented = ('em', 'load')

    def __init__(self):
        """Initialize top-level parameters for PMLR."""
        self.alpha_prior = None
        self.K = None

        self.comps = []
        self.z = []

    @property
    def alpha(self):
        return None if self.alpha_prior is None else self.alpha_prior.alpha

    @property
    def alpha_k(self):
        return self.alpha / self.K

    @property
    def W(self):
        return np.array([comp.mean for comp in self])

    @property
    def sigma(self):
        return np.array([comp.cov for comp in self])

    def posterior_rvs(self):
        stats = [comp.posterior.rvs() for comp in self]
        W = np.r_[[stat[0] for stat in stats]]
        sigma = np.r_[[stat[1] for stat in stats]]
        return W, sigma

    def n_assigned(self):
        return np.bincount(self.z)

    def init_comps(self, X, y, pids, init_method='kmeans', iters=100):
        """Initialize mixture components.

        Args:
            X (np.ndarray): Matrix with feature vector for each sample as rows.
            y (np.ndarray): Observed target for each instance.
            pids (np.ndarray): Primary entity ID for each instance.
            init_method (str): Method to use for initialization. One of:
                'kmeans': initialize using k-means clustering with K clusters.
                'random': randomly assign instances to components.
                'load':   load parameters from previously learned model.
            iters (int): Number of iterations to use for k-means
                initialization if init_method is 'kmeans'.
        """
        self.validate_init_method(init_method)
        return self._init_comps(X, y, pids, init_method, iters)

    def _init_comps(self, X, y, pids, init_method='kmeans', iters=100):
        """Choose an initial assignment of instances to components using
        the specified init_method. This also initializes the component
        parameters based on the assigned data instances.
        """
        N, f = X.shape
        uniq_pids = np.unique(pids)  # array of unique primary entity ids
        n_pe = len(uniq_pids)  # number of primary entities
        self.labels = np.arange(self.K)

        if init_method == 'random':
            self.z = np.random.randint(0, self.K, n_pe)
            self.Z = self.z[pids]
            self.comps = [MGLRComponent(X, y, self.Z == k) for k in self.labels]
        # Currently broken for this model.
        if init_method == 'kmeans':
            centroids, self.z = spvq.kmeans2(X, self.K, minit='points', iter=iters)
            self.Z = self.z[pids]
            self.comps = [MGLRComponent(X, y, self.Z == k) for k in self.labels]
        elif init_method == 'em':
            pass
        elif init_method == 'load':
            pass

    def fit(self, X, y, I, K, alpha=0.0, init_method='kmeans', iters=100,
            nsamples=220, burnin=20, thin_step=2):
        """Fit the parameters of the model using the data X.
        See `init_comps` method for info on parameters not listed here.

        Args:
            X (np.ndarray): Data matrix with primary entity as first index,
                secondary entity as second index, and feature vectors as third.
            y (np.ndarray): Observation vector corresponding to X.
            I (np.ndarray): Entity ids for each observation; first column
                corresponds to the primary entity (often users).
            K (int): Fixed number of components.
            alpha (float): Dirichlet hyper-parameter alpha; defaults to K.
            nsamples (int): Number of Gibbs samples to draw.
            burnin (int): Number of Gibbs samples to discard.
            thin_step (int): Stepsize for thinning to reduce autocorrelation.
        """
        self.K = K
        N, f = X.shape

        # TODO: update to use I instead of the row-wise X and y.
        pids = I[:, 0].astype(np.int)  # primary entity ids
        uniq_pids = np.unique(pids)  # array of unique primary entity ids
        self.n_pe = n_pe = len(uniq_pids)  # number of primary entities

        self.init_comps(X, y, pids, init_method, iters)
        self.alpha_prior = AlphaGammaPrior(alpha if alpha else float(K))

        self.nsamples = nsamples
        self.burnin = burnin
        self.thin_step = thin_step

        # Set alpha to K by default if not given.
        # Setting to K makes the Dirichlet uniform over the components.
        K = self.K
        alpha = self.alpha
        alpha_k = alpha / K

        # We'll use this for our conditional multinomial probs.
        Pk = np.ndarray(K)
        probs = np.ndarray(K)
        denom = float(n_pe + alpha - 1)

        # Init trace vars for parameters.
        keeping = self.nsamples - self.burnin
        store = int(keeping / self.thin_step)
        idx = -1  # initial trace index.

        trace = PMLRTrace(store, n_pe, f, K)
        logging.info('initial log-likelihood: %.3f' % self.llikelihood())

        # Construct masks for each user.
        masks = [(pids == i).nonzero()[0] for i in range(n_pe)]

        # Run collapsed Gibbs sampling to fit the model.
        indices = np.arange(n_pe)
        for iternum in range(self.nsamples):
            # Will we log this iteration?
            logging_iter = iternum % self.thin_step == 0

            # If we're saving the sample, calculate trace index.
            # We do not save samples discarded by chain burn in or thinning.
            saving_sample = logging_iter and (iternum >= self.burnin - 1)
            if saving_sample:
                idx = (iternum - self.burnin) / self.thin_step

            # Randomly permute the user IDs and iterate over the permutation.
            while True:
                try:
                    self._sample_once(
                        indices, masks, X, y, alpha_k, denom, probs, Pk,
                        trace, idx, saving_sample)
                    break
                except sp.linalg.LinAlgError:  # pp cov became non-PSD
                    self.init_comps(X, y, pids, init_method, iters)
                    logging.info('pp cov became non-PSD; random restart')

            # Log the likelihood and consider saving the sample in the trace.
            if logging_iter:
                llik = self.llikelihood()
                logging.info('sample %03d, log-likelihood: %.3f' % (iternum, llik))

                if saving_sample:
                    trace.pi[idx] = Pk
                    trace.W[idx], trace.sigma[idx] = self.posterior_rvs()
                    trace.ll[idx] = llik

        self.Z = self.z[pids]
        return trace

    def _sample_once(self, indices, masks, X, y, alpha_k, denom, probs, Pk,
                     trace, idx, saving_sample):
        """Draw samples for a single iteration, optionally storing in trace.

        Raises:
            sp.linalg.LinAlgError: if a posterior predictive covariance matrix
                estimated during Z sampling becomes non-PSD (positive
                semi-definite).
        """
        # Randomly permute the user IDs and iterate over the permutation.
        for i in np.random.permutation(indices):
            # Filter to observations for this user.
            _is = masks[i]
            xs = X[_is]
            ys = y[_is]

            # Remove X[i]'s stats from component z[i].
            old_k = self.z[i]
            self.comps[old_k].rm_instances(_is)

            # Calculate probability of instance belonging to each comp.
            # Calculate P(z[i] = k | z[-i], alpha)
            weights = (self.counts + alpha_k) / denom

            # Calculate P(X[i] | X[-i], pi, mu, sigma)
            for k, comp in enumerate(self.comps):
                comp.fit_pp(xs)
                probs[k] = comp.pp.pdf(ys)

            # Calculate P(z[i] = k | z[-i], X, alpha, pi, mu, Sigma)
            Pk[:] = probs * weights

            # Normalize probabilities.
            Pk[:] = Pk / Pk.sum()

            # Sample new component for X[i] using normalized probs.
            new_k = np.nonzero(np.random.multinomial(1, Pk))[0][0]

            # Add X[i] to selected component. Sufficient stats are updated.
            self.comps[new_k].add_instances(_is)
            self.z[i] = new_k

            # save posterior responsibility each component takes for
            # explaining data instance i
            if saving_sample:
                trace.H[idx, i] = Pk

    def label_llikelihood(self):
        """Calculate ln(P(z | alpha)), the marginal log-likelihood of the
        component instance assignments.

        Eq. (22) from Kamper's guide.
        """
        alpha = self.alpha
        alpha_k = alpha / self.K

        llik = spsp.gammaln(alpha) - spsp.gammaln(self.n_pe + alpha)
        counts = self.n_assigned().astype(np.float) + alpha_k
        llik += (spsp.gammaln(counts) - spsp.gammaln(alpha_k)).sum()
        return llik

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


def make_parser():
    parser = argparse.ArgumentParser(
        'Personalized Mixture of Gaussian Linear Regressions on synthetic data')
    cli.add_gibbs_args(parser)
    cli.add_mixture_args(parser)
    parser.add_argument(
        '-nu', '--nusers',
        type=int, default=4,
        help='number of users')
    parser.add_argument(
        '-nsg', '--nsamples-to-generate',
        type=int, default=20,
        help='number of samples')
    parser.add_argument(
        '-nf', '--nfeatures',
        type=int, default=2,
        help='number of features')
    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = cli.parse_and_setup(parser)

    nusers = args.nusers
    nsamples = args.nsamples_to_generate
    K = args.K
    F = args.nfeatures

    logging.info('number of users: %d' % nusers)
    logging.info('number of samples: %d' % nsamples)
    logging.info('number of clusters: %d' % K)
    logging.info('number of features: %d' % F)

    data, params = gendata.gen_prmix_data(nusers, nsamples, F, K)
    X = data['X']
    y = data['y']
    I = data['I']

    pmlr = PMLR()
    trace = pmlr.fit(
        X, y, I, K, init_method=args.init_method, nsamples=args.nsamples,
        burnin=args.burnin, thin_step=args.thin_step)

    # Calculate expectations using Monte Carlo estimates.
    E = trace.expectation()
