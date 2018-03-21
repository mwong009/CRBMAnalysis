import theano
import pickle
import numpy as np
import theano.tensor as T
import matplotlib.pyplot as plt
from collections import OrderedDict
from optimizers import Optimizers
from pylab import rcParams


class Network(object):
    def __init__(self, name, hyper, load_params=False):

        if load_params:
            try:
                with open(name + '.params', 'rb') as f:
                    model_values, hyper, curves = pickle.load(f)
            except IOError as e:
                print("Error opening file: ", e)
        else:
            model_values = {}
            curves = {'CD error': [], 'log likelihood': []}

        # initialize random number generator
        self.np_rng = np.random.RandomState(hyper['seed'])
        self.theano_rng = T.shared_randomstreams.RandomStreams(hyper['seed'])

        self.name = name
        self.model_values = model_values
        self.hyperparameters = hyper
        self.monitoring_curves = curves
        self.model_params = OrderedDict()
        self.model_params_shapes = OrderedDict()
        self.opt = Optimizers()

    def save_params(self, iter):
        """
        save_params func
            Saves model parameter values to a pickle file. To read
            params, unpickle and reshape.

        """
        path = '{0:s}_{1:d}.params'.format(self.name, iter)
        hyper = self.hyperparameters
        curves = self.monitoring_curves
        model_values = {}
        # evaluating tensor shared variable to numpy array
        for param_name, param in self.model_params.items():
            model_values[param_name] = param.eval().reshape(
                self.model_params_shapes[param_name])

        to_file = model_values, hyper, curves
        with open(path, 'wb') as f:
            pickle.dump(to_file, f, protocol=pickle.HIGHEST_PROTOCOL)

    def plot_curves(self, curves=None):

        if curves is None:
            curves = self.monitoring_curves
        rcParams['axes.xmargin'] = 0
        rcParams['axes.ymargin'] = 0
        rcParams['figure.figsize'] = (12.8, 4.8)
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        ax1.set(title='CD loss', xlabel='iterations')
        ax1.plot(
            *zip(*curves['CD error']),
            linewidth=0.5,
            color='C0'
        )
        ax2.set(title='log likelihood loss', xlabel='iterations')
        ax2.plot(
            *zip(*curves['log likelihood']),
            linewidth=0.5
        )
        fig.set
        fig.savefig(self.name)
