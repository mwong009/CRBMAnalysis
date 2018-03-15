import theano
import pandas as pd


class Setup(object):
    def __init__(self):
        pass

    @staticmethod
    def load_variables(filename, x={}, y={}):

        df = pd.read_csv(filename)

        x['n_person'] = {
            'data': theano.shared(df['n_person'].values),
            'dtype': VARIABLE_TYPE_REAL,
            'shape': (1, 1),
            'label': None,
            'stddev': None
        }

        x['driver_lic'] = {
            'data': theano.shared(df['driver_lic'].values),
            'dtype': VARIABLE_TYPE_BINARY,
            'shape': (1, 1),
            'label': None,
            'stddev': None
        }

        x['trip_purp'] = {
            'data': theano.shared(df['trip_purp'].values),
            'dtype': VARIABLE_TYPE_CATEGORY,
            'shape': (1, 6),
            'label': None,
            'stddev': None
        }

        y['mode_prime'] = {
            'data': theano.shared(np.eye(
                df['mode_prime'].values.max() + 1)[df['mode_prime'].values]
            ),
            'dtype': VARIABLE_TYPE_CATEGORY,
            'shape': (1, 9),
            'label': T.cast(theano.shared(df['mode_prime'].values), 'int32'),
            'stddev': None
        }

        return x, y
