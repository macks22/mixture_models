"""
Tests for the `mixture.mixture` module.

"""
import unittest

import mock
import numpy as np

from ..mixture.mixture import MixtureModel, MixtureComponent


class TestMixtureModel(unittest.TestCase):

    def test_init_method_validation_unsupported(self):
        """Should raise ValueError for unsupported method."""
        name = '__this_is_an_unsupported_method_name__'
        with self.assertRaises(ValueError):
            MixtureModel.validate_init_method(name)

    def test_init_method_validation_not_implemented(self):
        """Should raise NotImplementedError if supported but not implemented."""
        # Backup reference tuples used for validation and then replace them.
        backup_supported = MixtureModel._supported
        backup_not_implemented = MixtureModel._not_implemented

        name = 'c'
        MixtureModel._supported = ('a', 'b', 'c')
        MixtureModel._not_implemented = (name,)

        # Run the test.
        with self.assertRaises(NotImplementedError):
            MixtureModel.validate_init_method(name)

        # Restore original reference tuples.
        MixtureModel._supported = backup_supported
        MixtureModel._not_implemented = backup_not_implemented

    def test_iter_returns_generator_of_comps(self):
        """Should iterate over `comps` instance variable."""
        model = MixtureModel()
        test_values = range(10)
        model.comps = test_values
        iterator = iter(model)
        for comp, test_value in zip(iterator, test_values):
            self.assertEqual(comp, test_value)

    def setup_for_count_related_tests(self):
        model = MixtureModel()

        # Generate some simple objects with an `n` attribute.
        n_comps = 5
        counts = range(n_comps)
        test_values = [type('comp', tuple(), {'n': counts[k]}) for k in counts]

        # Assign those to the `comps` instance variable and run the test.
        model.comps = test_values
        return model, counts

    def test_counts(self):
        """Should return `n` instance variables for each component."""
        model, counts = self.setup_for_count_related_tests()
        for count, test_count in zip(model.counts, counts):
            self.assertEqual(count, test_count)

    def test_number_of_instances_assigned_property(self):
        """The instance variable `n` should return the sum of counts."""
        model, counts = self.setup_for_count_related_tests()
        self.assertEqual(model.n, sum(counts))

    def test_number_of_features_no_comps(self):
        """Number of features should be 0 when no components are assigned."""
        model = MixtureModel()
        model.comps = []  # may be set to this already, but let's be sure
        self.assertEqual(model.nf, 0)

    def test_number_of_features_with_comps(self):
        """Number of features should be same as first mixture component."""
        model = MixtureModel()
        nf = 10
        model.comps = [type('c', tuple(), {'nf': nf})]
        self.assertEqual(model.nf, nf)

    def test_label_likelihood(self):
        """Should return exp{label_llikelihood}."""
        # We monkey-patch the missing label log likelihood method.
        backup = MixtureModel.label_llikelihood
        return_value = 1.0
        MixtureModel.label_llikelihood = lambda self: return_value

        # Instantiate an object with the patched method and run the test.
        model = MixtureModel()
        should_be = np.exp(return_value)
        self.assertEqual(model.label_likelihood(), should_be)

        # Now restore the old label log likelihood method.
        MixtureModel.label_llikelihood = backup

    def test_likelihood(self):
        """Should return exp{llikelihood}."""
        # We monkey-patch the missing log likelihood method.
        backup = MixtureModel.llikelihood
        return_value = 1.0
        MixtureModel.llikelihood = lambda self: return_value

        # Instantiate an object with the patched method and run the test.
        model = MixtureModel()
        should_be = np.exp(return_value)
        self.assertEqual(model.likelihood(), should_be)

        # Now restore the old label log likelihood method.
        MixtureModel.llikelihood = backup


class DummyPrior(object):
    """Dummy class to use as prior for MixtureComponent testing."""

    def copy(self):
        return DummyPrior()


class TestMixtureComponent(unittest.TestCase):

    def setUp(self):
        dims = (20, 3)  # arbitrary data dimensions
        self.X = np.ndarray(dims)
        self.dummy_prior = DummyPrior()

    def tearDown(self):
        del self.X
        del self.dummy_prior

    def create_comp_with_instances(self, instances):
        mask = np.ones(self.X.shape[0])
        mask[instances] = 0
        mask = (mask == 0)
        with mock.patch.object(
                MixtureComponent, '_populate_cache', return_value=None):
            comp = MixtureComponent(self.X, mask, self.dummy_prior)

        return mask, comp

    def test_init_assigns_correct_instances(self):
        """Init should assign only specified data instances to component."""
        first5 = np.arange(self.X.shape[0])[:5]  # indices of first 5 instances

        with mock.patch.object(MixtureComponent,
                        '_populate_cache',
                        return_value=None) as mock_populate_cache:
            comp = MixtureComponent(self.X, first5, self.dummy_prior)
            mock_populate_cache.assert_called_once_with()

            self.assertTrue((comp.X == self.X[first5]).all())
            self.assertEqual(comp.n, 5)
            self.assertEqual(comp.nf, self.X.shape[1])

    def test_init_with_default_prior(self):
        """Initializer should set prior to default_prior if none is passed."""
        first5 = np.arange(self.X.shape[0])[:5]  # indices of first 5 instances
        dummy = DummyPrior()

        with mock.patch.object(
                MixtureComponent, '_populate_cache', return_value=None) \
                    as mock_populate_cache,\
             mock.patch.object(
                MixtureComponent, 'default_prior', return_value=dummy) \
                    as mock_default_prior:

            comp = MixtureComponent(self.X, first5)
            mock_default_prior.assert_called_once_with()
            mock_populate_cache.assert_called_once_with()

            self.assertEqual(comp.prior, dummy)
            self.assertTrue(isinstance(comp.posterior, DummyPrior))

    def test_is_empty_when_empty(self):
        """Should return True when no instances are assigned."""
        first0 = np.arange(self.X.shape[0])[:0]  # empty array
        mask, comp = self.create_comp_with_instances(first0)
        self.assertTrue(comp.is_empty)

    def test_is_empty_when_not_empty(self):
        """Should return False when some instances are assigned."""
        first5 = np.arange(self.X.shape[0])[:5]  # indices of first 5 instances
        mask, comp = self.create_comp_with_instances(first5)
        self.assertFalse(comp.is_empty)

    def test_add_instance_in_component(self):
        """Should do nothing when instance is already member."""
        first5 = np.arange(self.X.shape[0])[:5]  # indices of first 5 instances
        mask, comp = self.create_comp_with_instances(first5)

        with mock.patch.object(
                MixtureComponent, '_cache_add_instance', return_value=None) \
                    as mock_cache_add_instance:

            to_add = 0
            self.assertTrue(comp._instances[to_add])  # should be in first 5
            comp.add_instance(to_add)

            # Should not have changed anything.
            self.assertEqual(mock_cache_add_instance.call_count, 0)
            self.assertTrue(comp._instances[to_add])

    def test_add_instance_not_in_component(self):
        """Should add specified instances to cache and instance mask."""
        first0 = np.arange(self.X.shape[0])[:0]  # empty array
        mask, comp = self.create_comp_with_instances(first0)

        with mock.patch.object(
                MixtureComponent, '_cache_add_instance', return_value=None) \
                    as mock_cache_add_instance,\
             mock.patch.object(
                MixtureComponent, 'fit', return_value=None) \
                    as mock_fit:

            to_add = 0
            self.assertFalse(comp._instances[to_add])  # should not be member
            comp.add_instance(to_add)

            # Should have added instance to cache and
            # set its mask position to True.
            mock_cache_add_instance.assert_called_once_with(to_add)
            self.assertTrue(comp._instances[to_add])

            # Should have called fit.
            mock_fit.assert_called_once_with()

    def test_rm_instance_not_present(self):
        """Should raise IndexError on attempt to remove non-member instance."""
        first0 = np.arange(self.X.shape[0])[:0]  # empty array
        mask, comp = self.create_comp_with_instances(first0)

        with self.assertRaises(IndexError):
            comp.rm_instance(0)

    def test_rm_instance(self):
        """Should remove specified instances from cache and instance mask."""
        first5 = np.arange(self.X.shape[0])[:5]  # empty array
        mask, comp = self.create_comp_with_instances(first5)

        with mock.patch.object(
                MixtureComponent, '_cache_stats', return_value=None) \
                    as mock_cache_stats,\
             mock.patch.object(
                MixtureComponent, '_cache_rm_instance', return_value=None) \
                    as mock_cache_rm_instance,\
             mock.patch.object(
                MixtureComponent, 'fit', return_value=None) \
                    as mock_fit:

            # Get sum of X values before removing to compare with sum after.
            Xsum = comp.X.sum(0)

            to_rm = 0
            self.assertTrue(comp._instances[to_rm])  # should be member
            comp.rm_instance(to_rm)

            # Should have cached stats, removed instance, and refit.
            self.assertFalse(comp._instances[to_rm])
            mock_cache_stats.assert_called_once_with()
            mock_cache_rm_instance.assert_called_once_with(to_rm)
            mock_fit.assert_called_once_with()

            # Sum of X columns should have decreased by the removed instances
            # values.
            removed = self.X[to_rm]
            diff = Xsum - comp.X.sum(0)
            self.assertTrue((diff == removed).all())

    def test_add_instance_just_removed(self):
        """Should restore from cache and avoid refitting."""
        first5 = np.arange(self.X.shape[0])[:5]  # empty array
        mask, comp = self.create_comp_with_instances(first5)

        with mock.patch.object(
                MixtureComponent, '_restore_from_cache', return_value=None) \
                    as mock_restore_from_cache,\
             mock.patch.object(
                MixtureComponent, 'fit', return_value=None) \
                    as mock_fit:

            # Manually remove last i removed to set up test.
            to_rm = 0
            comp._last_i_removed = 0
            comp._instances[to_rm] = False

            # Add the instance and make sure cache reload triggered.
            comp.add_instance(to_rm)
            mock_restore_from_cache.assert_called_once_with()
            self.assertEqual(mock_fit.call_count, 0)


if __name__ == "__main__":
    unittest.main()
