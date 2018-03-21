import h5py
import theano
import numpy as np
import pandas as pd
import theano.tensor as T

VARIABLE_TYPE_BINARY = 'binary'
VARIABLE_TYPE_REAL = 'real'
VARIABLE_TYPE_CATEGORY = 'category'
VARIABLE_TYPE_INTEGER = 'integer'
DTYPE_FLOATX = theano.config.floatX


class SetupH5PY(object):
    def __init__(self):
        pass

    @staticmethod
    def load_dataset(filename='data.h5'):
        f = h5py.File(filename, 'r')
        return f

    @staticmethod
    def compile_dataset(filename='datatable.csv'):
        df = pd.read_csv(filename)
        with h5py.File('data.h5', 'w') as f:
            f.attrs['n_rows'] = len(df)
            f.create_dataset(
                name='mode_prime/data',
                data=np.eye(df.mode_prime.values.max()+1)[
                     df.mode_prime.values].reshape(len(df), 1, -1),
                dtype=DTYPE_FLOATX
            )
            f.create_dataset(
                name='mode_prime/label',
                data=df.mode_prime.values,
                dtype=DTYPE_FLOATX
            )
            f['mode_prime'].attrs['dtype'] = 'category'

            f.create_dataset(
                name='n_person/data',
                data=df.n_person.values.reshape(len(df), 1, -1)/1.37535,
                dtype=DTYPE_FLOATX
            )
            f['n_person'].attrs['dtype'] = 'real'
            f['n_person'].attrs['stddev'] = 1.37535

            f.create_dataset(
                name='driver_lic/data',
                data=df.driver_lic.values.reshape(len(df), 1, -1),
                dtype=DTYPE_FLOATX
            )
            f['driver_lic'].attrs['dtype'] = 'binary'

            f.create_dataset(
                name='trip_purp/data',
                data=np.eye(df.trip_purp.values.max()+1)[
                     df.trip_purp.values].reshape(len(df), 1, -1),
                dtype=DTYPE_FLOATX
            )
            f.create_dataset(
                name='trip_purp/label',
                data=df.trip_purp.values,
                dtype=DTYPE_FLOATX
            )
            f['trip_purp'].attrs['dtype'] = 'category'
