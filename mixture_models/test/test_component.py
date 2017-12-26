"""
Tests for the `mixture.component` module.

"""
import unittest

import mock
import numpy as np

from ..mixture.component import GaussianComponent, MGLRComponent, GIG


class TestGaussianComponent(unittest.TestCase):
    pass


class TestMGLRComponent(unittest.TestCase):
    """Unittests for the MGLRComponent class."""

    def setUp(self):
        dims = (20, 3)  # arbitrary data dimensions
        n, f = dims

        # Randomly select 1/4 of the indices and make mask.
        self.indices = np.arange(n)
        sample_size = n / 4
        self.instances = np.random.choice(
            self.indices, size=sample_size, replace=False)

        # Build boolean mask from index array.
        self.mask = np.ones(n)
        self.mask[self.instances] = 0
        self.mask = (self.mask == 0)

        # Randomly generate Xs and ys.
        self.X = np.random.randn(n, f)
        self.y = np.random.randn(n)

        # Mock the GIG prior so we can ignore its workings when needed.
        # self.prior_class = mock.create_autospec(GIG)
        self.prior_params = (np.zeros(f), np.eye(f), 1.0, 1.0)
        self.prior = GIG(*self.prior_params)
        # self.prior_class.copy = \
        #     lambda inner_self: self.prior_class(*self.prior_params)
        # self.mock_prior = self.prior_class(*self.prior_params)

    def tearDown(self):
        del self.indices
        del self.instances
        del self.mask

        del self.X
        del self.y

        # del self.prior_class
        del self.prior_params
        del self.prior
        # del self.mock_prior

    def test_init_passing_prior(self):
        """Should set prior and posterior correctly when passing prior."""
        comp = MGLRComponent(self.X, self.y, self.mask, self.prior)

        # Let's make sure prior was actually assigned.
        self.assertEqual(comp.prior, self.prior)

        # What's important to guarantee here?
        # Testing actual conjugate updates is overkill here.

        # Prior and posterior should be same distribution => Conjugacy.
        self.assertEqual(type(comp.prior), type(comp.posterior))

        # Prior and posterior parameters should differ.
        for param_name in comp.prior.param_names:
            prior_val = getattr(comp.prior, param_name)
            post_val = getattr(comp.posterior, param_name)
            if hasattr(prior_val, 'shape'):  # numpy variable
                self.assertFalse((prior_val == post_val).all())
            else:
                self.assertNotEqual(prior_val, post_val)

    def test_init_default_prior(self):
        """Should set prior and posterior correctly with default."""
        with mock.patch.object(
            MGLRComponent, 'default_prior', return_value=self.prior) \
                as mock_default_prior:

            comp = MGLRComponent(self.X, self.y, self.mask)
            mock_default_prior.assert_called_once_with()

        # Let's make sure prior was actually assigned.
        self.assertEqual(comp.prior, self.prior)

        # What's important to guarantee here?
        # Testing actual conjugate updates is overkill here.

        # Prior and posterior should be same distribution => Conjugacy.
        self.assertEqual(type(comp.prior), type(comp.posterior))

        # Prior and posterior parameters should differ.
        for param_name in comp.prior.param_names:
            prior_val = getattr(comp.prior, param_name)
            post_val = getattr(comp.posterior, param_name)
            if hasattr(prior_val, 'shape'):  # numpy variable
                self.assertFalse((prior_val == post_val).all())
            else:
                self.assertNotEqual(prior_val, post_val)

    def test_init_sets_X_to_subset(self):
        """Should set instance mask on X using initialization instance mask."""
        comp = MGLRComponent(self.X, self.y, self.mask, self.prior)
        X_subset = self.X[self.mask]
        self.assertTrue((comp.X == X_subset).all())

    def test_init_sets_y_to_subset(self):
        """Should set instance mask on y using initialization instance mask."""
        comp = MGLRComponent(self.X, self.y, self.mask, self.prior)
        y_subset = self.y[self.mask]
        self.assertTrue((comp.y == y_subset).all())

    def test_init_fits_posterior(self):
        """The posterior distribution should be fit on initialization."""
        with mock.patch.object(
                MGLRComponent, 'fit', return_value=None) as mock_fit:

            comp = MGLRComponent(self.X, self.y, self.mask, self.prior)
            mock_fit.assert_called_once_with()

    def test_sufficient_stats_are_correct_after_init(self):
        """Sufficient stats should be accurate after initialization."""
        comp = MGLRComponent(self.X, self.y, self.mask, self.prior)

        # Calculate sufficient stats manually
        X_subset = self.X[self.mask]
        y_subset = self.y[self.mask]
        n = X_subset.shape[0]
        x_ssq = X_subset.T.dot(X_subset)
        y_ssq = y_subset.dot(y_subset)
        Xy = X_subset.T.dot(y_subset)
        stats = (n, x_ssq, y_ssq, Xy)

        for correct, under_test in zip(stats, comp.sufficient_stats()):
            if hasattr(correct, 'shape'):
                self.assertTrue((correct == under_test).all())
            else:
                self.assertEqual(correct, under_test)

    def test_rm_instances_invalid_indices_bool_mask(self):
        """Should raise IndexError when trying to remove non-member indices."""
        comp = MGLRComponent(self.X, self.y, self.mask, self.prior)
        anti_mask = ~self.mask
        with self.assertRaises(IndexError):
            comp.rm_instances(anti_mask)

    def test_rm_instances_invalid_indices_int_index(self):
        """Should raise IndexError when trying to remove non-member indices."""
        comp = MGLRComponent(self.X, self.y, self.mask, self.prior)
        anti_mask = ~self.mask
        indices = anti_mask.nonzero()[0]
        with self.assertRaises(IndexError):
            comp.rm_instances(indices)

    def test_rm_all_instances_removes_rows_bool_mask(self):
        """Using boolean mask, remove all instances so comp has 0 members."""
        comp = MGLRComponent(self.X, self.y, self.mask, self.prior)
        comp.rm_instances(self.mask)

        self.assertEqual(comp.X.shape[0], 0)
        self.assertEqual(comp.y.shape[0], 0)
        self.assertEqual(comp.n, 0)

    def test_rm_all_instances_removes_rows_int_index(self):
        """Using int index arr, remove all instances so comp has 0 members."""
        comp = MGLRComponent(self.X, self.y, self.mask, self.prior)
        indices = self.mask.nonzero()[0]
        comp.rm_instances(indices)

        self.assertEqual(comp.X.shape[0], 0)
        self.assertEqual(comp.y.shape[0], 0)
        self.assertEqual(comp.n, 0)

    def test_rm_instances_removes_rows_bool_mask(self):
        """Using boolean mask, should remove the associated X,y values."""
        comp = MGLRComponent(self.X, self.y, self.mask, self.prior)

        # Make copies of X and y to compare against after removing.
        X_before = comp.X.copy()
        y_before = comp.y.copy()

        # Remove the first few assigned members.
        n_removed = 2  # must be < 5 (see self.setUp)
        to_rm = self.mask.nonzero()[0][:n_removed]  # first n_removed
        self.mask[:] = False
        self.mask[to_rm] = True
        comp.rm_instances(self.mask)

        # Compare the saved X and y to the new X and y.
        self.assertEqual(comp.X.shape, X_before[n_removed:].shape)
        self.assertEqual(comp.y.shape, y_before[n_removed:].shape)

        self.assertTrue((comp.X == X_before[n_removed:]).all())
        self.assertTrue((comp.y == y_before[n_removed:]).all())

    def test_rm_instances_removes_rows_int_index(self):
        """Using int index array, should remove the associated X,y values."""
        comp = MGLRComponent(self.X, self.y, self.mask, self.prior)

        # Make copies of X and y to compare against after removing.
        X_before = comp.X.copy()
        y_before = comp.y.copy()

        # Remove the first few assigned members.
        n_removed = 2  # must be < 5 (see self.setUp)
        to_rm = self.mask.nonzero()[0][:n_removed]  # first n_removed
        comp.rm_instances(to_rm)

        # Compare the saved X and y to the new X and y.
        self.assertEqual(comp.X.shape, X_before[n_removed:].shape)
        self.assertEqual(comp.y.shape, y_before[n_removed:].shape)

        self.assertTrue((comp.X == X_before[n_removed:]).all())
        self.assertTrue((comp.y == y_before[n_removed:]).all())

    def test_rm_instances_updates_sufficient_stats(self):
        """Should remove stats for removed indices."""
        comp = MGLRComponent(self.X, self.y, self.mask, self.prior)

        n_before, X_ssq_before, y_ssq_before, Xy_before = \
            comp.sufficient_stats()

        # Remove the first few assigned members.
        n_removed = 2  # must be < 5 (see self.setUp)
        to_rm = self.mask.nonzero()[0][:n_removed]  # first n_removed
        comp.rm_instances(to_rm)

        # Calculate sufficient stats removed.
        X_removed = self.X[to_rm]
        y_removed = self.y[to_rm]
        X_ssq_removed = X_removed.T.dot(X_removed)
        y_ssq_removed = y_removed.dot(y_removed)
        Xy_removed = X_removed.T.dot(y_removed)

        # Now get new stats and compare to old.
        n_after, X_ssq_after, y_ssq_after, Xy_after = \
            comp.sufficient_stats()

        self.assertEqual(n_after, n_before - n_removed)
        self.assertTrue((X_ssq_after == (X_ssq_before - X_ssq_removed)).all())
        self.assertEqual(y_ssq_after, (y_ssq_before - y_ssq_removed))
        self.assertTrue((Xy_after == (Xy_before - Xy_removed)).all())

    def test_rm_all_instances_zeros_sufficient_stats(self):
        """Should set all stats to 0 when removing all members."""
        comp = MGLRComponent(self.X, self.y, self.mask, self.prior)
        comp.rm_instances(self.mask)
        for stat in comp.sufficient_stats():
            if hasattr(stat, 'shape'):
                self.assertTrue((stat == 0).all())
            else:
                self.assertEqual(stat, 0)

    def test_add_instances_all_present(self):
        """Should do nothing when all instances are already members."""
        pass

    def test_add_instances_adds_rows_bool_mask(self):
        """Using boolean mask, should add associated values of X and y."""
        pass

    def test_add_instances_adds_rows_int_index(self):
        """Using int index arr, should add associated values of X and y."""
        pass

    def test_add_instances_updates_sufficient_stats(self):
        """Should add stats of new instances to sufficient stats."""
        pass

    def test_add_instances_fits_when_not_cached(self):
        """Should call fit when not loading from cache."""
        pass

    def test_add_instances_just_removed_loads_from_cache(self):
        """Should load from the cache if adding just-removed instances."""
        pass


if __name__ == "__main__":
    unittest.main()
