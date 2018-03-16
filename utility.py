import theano
import numpy as np
import pandas as pd
import theano.tensor as T

VARIABLE_TYPE_BINARY = 'binary'
VARIABLE_TYPE_REAL = 'real'
VARIABLE_TYPE_CATEGORY = 'category'
VARIABLE_TYPE_INTEGER = 'integer'


class Setup(object):
    def __init__(self):
        pass

    @staticmethod
    def load_variables(filename, x={}, y={}):

        df = pd.read_csv(filename)
        num_rows, num_cols = df.shape

        x['n_person'] = {
            'data': theano.shared(
                np.asarray(df['n_person'].values / 1.37535,
                           dtype=theano.config.floatX).reshape(-1, 1, 1),
                borrow=True
            ),
            'dtype': VARIABLE_TYPE_REAL,
            'shape': (1, 1),
            'label': None,
            'stddev': 1.37535
        }

        x['driver_lic'] = {
            'data': theano.shared(np.asarray(
                    df['driver_lic'].values,
                    dtype=theano.config.floatX).reshape(-1, 1, 1),
                borrow=True
            ),
            'dtype': VARIABLE_TYPE_BINARY,
            'shape': (1, 1),
            'label': None,
            'stddev': None
        }

        x['trip_purp'] = {
            'data': theano.shared(np.eye(
                    df['trip_purp'].values.max() + 1,
                    dtype=theano.config.floatX)[
                        df['trip_purp'].values].reshape(num_rows, 1, -1),
                borrow=True
            ),
            'dtype': VARIABLE_TYPE_CATEGORY,
            'shape': (1, 4),
            'label': None,
            'stddev': None
        }

        y['mode_prime'] = {
            'data': theano.shared(np.eye(
                    df['mode_prime'].values.max() + 1,
                    dtype=theano.config.floatX)[
                        df['mode_prime'].values].reshape(num_rows, 1, -1),
                borrow=True
            ),
            'dtype': VARIABLE_TYPE_CATEGORY,
            'shape': (1, 8),
            'label': T.cast(theano.shared(np.asarray(
                        df['mode_prime'].values,
                        dtype=theano.config.floatX),
                    borrow=True),
                'int32'
            ),
            'stddev': None
        }

        return x, y, num_rows
