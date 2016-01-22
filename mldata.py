"""
Tools for loading, preprocessing, formatting, and partitioning csv-formatted
datasets for ML applications. These tools can be used by a variety of models to
quickly convert diverse datasets to appropriate formats for learning.
"""
import os
import logging
import argparse

import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from sklearn import preprocessing

from oset import OrderedSet


# Error codes
BOUNDS_FORMAT = 1000
MISSING_ATTRIBUTE = 1001
DIM_MISMATCH = 1002
BAD_FEATURE_CONF = 1003
BAD_FILENAME = 1004
BAD_CMDLINE_ARG = 1005


class BadFeatureConfig(Exception):
    """Raise when bad feature configuration file is found."""
    pass

class FeatureGuide(object):
    """Parse and represent fields of a feature guide."""

    """Guide to feature configuration file field letters."""
    config_guide = {
        't': 'target',
        'i': 'index',
        'k': 'key',
        'e': 'entities',
        'c': 'categoricals',
        'r': 'real_valueds'
    }

    sections = tuple(config_guide.values())
    other_sections = ('target', 'index')
    feature_sections = tuple(set(sections) - set(other_sections))

    # We use slots here not so much for efficiency as for documentation.
    __slots__ = [field_name for field_name in sections]
    __slots__.append('fname')

    @classmethod
    def parse_config(cls, fname):
        """Parse the given configuration file and return a dict of
        {field_letter: list_of_field_names_parsed}.
        """
        with open(fname) as f:
            lines = [l.strip() for l in f.read().split('\n')
                     if not l.startswith('#') and l.strip()]

        # We use a simple state-machine approach to the parsing
        # in order to deal with multi-line sections.
        parsing = False
        keys = cls.config_guide.keys()
        vars = {var: [] for var in keys}
        for line in lines:
            if not parsing:
                k, csv = line.split(':')
            else:
                csv = line

            vars[k].extend([val.strip() for val in csv.split(',')])
            parsing = not line.endswith(';')
            if not parsing:
                vars[k][-1] = vars[k][-1][:-1]  # remove semi-colon

        # Remove whitespace strings. These may have come from something like:
        # c: this, , that;
        for k in keys:
            vars[k] = [val for val in vars[k] if val]  # already stripped

        return vars

    def __init__(self, fname):
        """Read the feature guide and parse out the specification.

        The expected file format is the following:

            t:<target>;
            i:<single index field name, if one exists>;
            k:<comma-separated fields that comprise a unique key>;
            e:<comma-separated categorical entity names>;
            c:<comma-separated categorical variable names>;
            r:<comma-separated real-valued variable names>;

        Whitespace is ignored, as are lines that start with a "#" symbol. Any
        variables not included in one of the three groups is ignored. We assume
        the first two categorical variables are the user and item ids.

        Args:
            fname (str): Path of the file containing the feature guide.

        Stores instance variables for each of the field areas under the names in
        FeatureGuide.config_guide.
        """
        self.fname = os.path.abspath(fname)
        vars = self.parse_config(fname)

        # Sanity checks.
        num_targets = len(vars['t'])
        if num_targets != 1:
            raise BadFeatureConfig(
                'feature config should specify 1 target; got %d; check for'
                'the semi-colon at the end of the t:<target> line' % num_targets)

        num_entity = len(vars['e'])
        if not num_entity:
            raise BadFeatureConfig('0 entity variables given; need at least 1')

        # num_features = len(vars['c']) + len(vars['r'])
        # if not num_features > 0:
        #     raise BadFeatureConfig('no predictors specified')

        # Store extracted field names as instance variables.
        logging.info('read the following feature guide:')
        for k, field_name in self.config_guide.items():
            logging.info('%s: %s' % (field_name, ', '.join(vars[k])))

            # convert to OrderedSet before setting attribute
            setattr(self, field_name, OrderedSet(vars[k]))

        # Extract target variable from its list and store solo.
        self.target = self.target[0]

    def __repr__(self):
        return '%s("%s")' % (self.__class__.__name__, self.fname)

    def __str__(self):
        return '\n'.join([
            'target: %s' % self.target,
            'index: %s' % ', '.join(self.index),
            'key: %s' % ', '.join(self.key),
            'entities: %s' % ', '.join(self.entities),
            'categoricals: %s' % ', '.join(self.categoricals),
            'real-valueds: %s' % ', '.join(self.real_valueds)
        ])

    @property
    def feature_names(self):
        return list(reduce(
                lambda s1, s2: s1 | s2,
                [getattr(self, attr) for attr in self.feature_sections]))

    @property
    def all_names(self):
        return self.feature_names + [self.target] + list(self.index)


"""
A dataset can exist in several different forms. For now, we assume one of two
forms:

    1.  Complete dataset in one file.
    2.  Separate train and test files with identical fields/indices/etc.

We allow the dataset to be initialized from either format and require an
accompanying config file that specifies the fields of the dataset. Since it may
be possible that other forms exist (e.g. with a hold-out set) we use a factory
to load the datasets. By making the first argument to the factory an optional
iterable, we can initialize differently based on the number of filenames
present.
"""
class Dataset(object):
    """Represent a complete dataset existing in one file."""
    pass


class PandasDataset(Dataset):

    @staticmethod
    def index_from_feature_guide(fguide):
        """Return an appropriate index column name based on the feature guide.
        We should use the index as the index if it is present.
        If the index is not present, we should use the key as the index only
        if none of those fields are also used as features.
        If some of those fields are used as features, we should simply make a
        new index and add it to the feature guide; in this case, return None.
        """
        if fguide.index:
            return list(fguide.index)
        elif (fguide.key and
              (fguide.key - fguide.feature_names) == fguide.key):
                return list(fguide.key)
        else:
            return None

    def index_colname(self):
        return self.index_from_feature_guide(self.fguide)

    def __init__(self, fname, config_file):
        """Load the feature configuration and then load the columns present in
        the feature config.
        """
        self.fguide = fguide = FeatureGuide(config_file)
        self.fname = os.path.abspath(fname)

        # We only need to load the columns that show up in the config file.
        index_col = self.index_colname()
        # Note that pandas interprets index_col=None as "make new index."
        self.dataset = pd.read_csv(
            fname, usecols=fguide.all_names, index_col=index_col)


def map_ids(data, key, id_map=None):
    """Map ids to 0-contiguous index. This enables the use of these ids as
    indices into an array (for the bias terms, for instance). This returns the
    number of unique IDs for `key`.
    """
    if id_map is None:
        ids = data[key].unique()
        n = len(ids)
        id_map = dict(zip(ids, range(n)))

    data[key] = data[key].apply(lambda _id: id_map[_id])
    return id_map


def read_train_test(train_file, test_file, conf_file):
    """Read the train and test data according to the feature guide in the
    configuration file. Return the train and test data as (X, y) pairs for the
    train and test data.

    Args:
        train_file (str): Path of the CSV train data file.
        test_file (str): Path of the CSV test data file.
        conf_file (str): Path of the configuration file.

    Returns:
        tuple: (train_eids, train_X, train_y,
                test_eids, test_X, test_y,
                feat_indices, number_of_entities),
               preprocessed train/test data and mappings from features to
               indices in the resulting data matrices. Both train and test are
               accompanied by 0-contiguous primary entity id arrays.

    """
    try:
        target, ents, cats, reals = read_feature_guide(conf_file)
    except IOError:
        raise IOError('invalid feature guide conf file path: %s' % conf_file)

    to_read = [target] + ents + cats + reals
    def read_file(name, fpath):
        try:
            data = pd.read_csv(fpath, usecols=to_read)
        except IOError:
            raise IOError('invalid %s file path: %s' % (name, fpath))
        except ValueError as err:
            attr_name = err.args[0].split("'")[1]
            attr_type = ('entity' if attr_name in ents else
                         'categorical' if attr_name in cats else
                         'real-valued' if attr_name in reals else
                         'target')
            raise BadFeatureConfig('%s attribute %s not found in %s file' % (
                attr_type, attr_name, name))

        return data

    train = read_file('train', train_file)
    test = read_file('test', test_file)
    n_train = train.shape[0]
    n_test = test.shape[0]
    logging.info('number of instances: train=%d, test=%d' % (
        train.shape[0], test.shape[0]))

    return train, test, target, ents, cats, reals


def preprocess(train, test, target, ents, cats, reals):
    """Return preprocessed (X, y, eid) pairs for the train and test sets.

    Preprocessing includes:

    1.  Map primary entity ID (first in ents) to a 0-contiguous range.
    2.  Z-score scale the real-valued features.
    3.  One-hot encode the categorical features (including primary entity ID).

    This function tries to be as general as possible to accomodate learning by
    many models. As such, there are 8 return values. The first three are:

    1.  train_eids: primary entity IDs as a numpy array
    2.  train_X: training X values (first categorical, then real-valued)
    3.  train_y: training y values (unchanged from input)

    The next three values are the same except for the test set. The final two
    values are:

    7.  indices: The indices of each feature in the encoded X matrix.
    8.  nents: The number of categorical features after one-hot encoding.

    """

    # Separate X, y for train/test data.
    n_train = train.shape[0]
    n_test = test.shape[0]
    train_y = train[target].values
    test_y = test[target].values

    # Read out id lists for primary entity.
    all_dat = pd.concat((train, test))
    id_map = map_ids(all_dat, ents[0])

    map_ids(train, ents[0], id_map)
    map_ids(test, ents[0], id_map)
    train_eids = train[ents[0]].values
    test_eids = test[ents[0]].values

    # Z-score scaling of real-valued features.
    if reals:
        scaler = preprocessing.StandardScaler()
        train_reals = scaler.fit_transform(train[reals])
        test_reals = scaler.transform(test[reals])

    # One-hot encoding of entity and categorical features.
    catf = ents + cats
    all_cats = pd.concat((train[catf], test[catf]))
    encoder = preprocessing.OneHotEncoder()
    enc = encoder.fit_transform(all_cats)
    train_cats = enc[:n_train]
    test_cats = enc[n_train:]

    # Create a feature map for decoding one-hot encoding.
    ncats = encoder.active_features_.shape[0]
    nreal = train_reals.shape[1] if reals else 0
    nf = ncats + nreal

    # Count entities.
    logging.info('after one-hot encoding, found # unique values:')
    counts = np.array([
        all_cats[catf[i]].unique().shape[0]
        for i in xrange(len(catf))
    ])
    indices = zip(catf, np.cumsum(counts))
    for attr, n_values in zip(ents, counts):
        logging.info('%s: %d' % (attr, n_values))

    # Add in real-valued feature indices.
    indices += zip(reals, range(indices[-1][1] + 1, nf + 1))

    # How many entity features and categorical features do we have?
    nents = dict(indices)[ents[-1]]
    ncats = ncats - nents
    nf = nents + ncats + nreal

    ent_idx = range(len(ents))
    cat_idx = range(len(ents), len(ents) + len(cats))
    nactive_ents = sum(encoder.n_values_[i] for i in ent_idx)
    nactive_cats = sum(encoder.n_values_[i] for i in cat_idx)

    logging.info('number of active entity features: %d of %d' % (
        nents, nactive_ents))
    logging.info('number of active categorical features: %d of %d' % (
        ncats, nactive_cats))
    logging.info('number of real-valued features: %d' % nreal)

    # Put all features together.
    if reals:
        train_X = sp.sparse.hstack((train_cats, train_reals))
        test_X = sp.sparse.hstack((test_cats, test_reals))
    else:
        train_X = train_cats
        test_X = test_cats

    logging.info('Total of %d features after encoding' % nf)

    return (train_eids, train_X, train_y,
            test_eids, test_X, test_y,
            indices, nents)


class Model(object):
    """General model class with load/save/preprocess functionality."""

    def set_fguide(self, fguidef=''):
        if fguidef:
            self.read_fguide(fguidef)
        else:
            self.reset_fguide()

    def read_fguide(self, fguidef):
        self.target, self.ents, self.cats, self.reals = \
            read_feature_guide(fguidef)

    def reset_fguide(self):
        self.target = ''
        for name in ['ents', 'cats', 'reals']:
            setattr(self, name, [])

    @property
    def fgroups(self):
        return ('ents', 'cats', 'reals')

    @property
    def nf(self):
        return sum(len(group) for group in self.fgroups)

    def check_fguide(self):
        if self.nf <= 0:
            raise ValueError("preprocessing requires feature guide")

    def read_necessary_fguide(self, fguide=''):
        if fguide:
            self.read_fguide(fguide)
        else:
            self.check_fguide()

    def preprocess(self, train, test, fguidef=''):
        self.read_necessary_fguide(fguidef)
        return preprocess(
            train, test, self.target, self.ents, self.cats, self.reals)

    def check_if_learned(self):
        unlearned = [attr for attr, val in self.model.items() if val is None]
        if len(unlearned) > 0 or len(self.model) == 0:
            raise UnlearnedModel("IPR predict with unlearned params", unlearned)

    def save(self, savedir, ow=False):
        save_model_vars(self.model, savedir, ow)

    def load(self, savedir):
        self.model = load_model_vars(savedir)
