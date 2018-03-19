import theano
import os
import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import theano.tensor as T
from collections import OrderedDict
from itertools import chain
from pylab import rcParams
# internal imports
from optimizers import Optimizers
from utility import Setup

VARIABLE_DTYPE_BINARY = 'binary'
VARIABLE_DTYPE_REAL = 'real'
VARIABLE_DTYPE_CATEGORY = 'category'
VARIABLE_DTYPE_INTEGER = 'integer'

net = 'net1', {
    'n_hidden': (4,),
    'seed': 42,
    'batch_size': 32,
    'variable_dtypes': [
        VARIABLE_DTYPE_BINARY,
        VARIABLE_DTYPE_REAL,
        VARIABLE_DTYPE_CATEGORY
    ],
    'noisy_rectifier': True,
    'learning_rate': 1e-3,
    'gibbs_steps': 1,
    'shapes': {},
    'amsgrad': True,
    'alpha': 1.0
}


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
        self.opt = Optimizers()

    def save_params(self, iter):
        """
        save_params func
            Saves model parameter values to a pickle file. To read
            params, unpickle and reshape.

        """

        path = '%s_%d.params'.format(self.name, iter)
        hyper = self.hyperparameters
        curves = self.monitoring_curves
        model_values = {}
        # evaluating tensor shared variable to numpy array
        for param_name, param in self.model_params.items():
            if self.hyperparameters['shapes'][param_name]:
                shape = self.hyperparameters['shapes'][param_name]
                model_values[param_name] = param.eval().reshape(shape)
            else:
                model_values[param_name] = param.eval()

        to_file = model_values, hyper, curves
        with open(path, 'wb') as f:
            pickle.dump(to_file, f, protocol=pickle.HIGHEST_PROTOCOL)

    def plot_curves(self, curves=None):

        if curves is None:
            curves = self.monitoring_curves
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        rcParams['axes.xmargin'] = 0
        rcParams['axes.ymargin'] = 0
        ax1.set(title='CD loss', xlabel='iterations')
        ax1.plot(
            *zip(*curves[0]),
            linewidth=0.5,
            alpha=0.8,
            linestyle='--',
            color='C0'
        )
        ax2.set(title='log likelihood loss', xlabel='iterations')
        ax2.plot(
            *zip(*curves[1]),
            linewidth=0.5,
            alpha=0.8
        )


class RBM(Network):
    ''' define the RBM toplevel '''
    def __init__(self, name, hyperparameters=OrderedDict()):
        Network.__init__(self, name, hyperparameters)
        self.input = []         # list of tensors
        self.input_dtype = []    # list of str dtypes
        self.output = []        # list of tensors
        self.output_dtype = []   # list of str dtypes
        self.label = []         # list of label tensors
        self.hbias = []
        self.W_params = []      # list of ALL the W params
        self.V_params = []      # list of the xWh params
        self.U_params = []      # list of the hWy params
        self.B_params = []      # list of all Bx params
        self.vbias = []
        self.cbias = []
        # flattened version
        self.W_params_flat = []      # list of ALL the W params
        self.V_params_flat = []      # list of the xWh params
        self.U_params_flat = []      # list of the hWy params
        self.B_params_flat = []      # list of the Bx params
        self.vbias_flat = []
        self.cbias_flat = []

    def add_latent(self, name='hbias', shp_hidden=None):
        """
        add_latent func

        Parameters
        ----------
        name : `str`, optional
            Name of hidden node e.g. `'hbias'`
        shp_hidden : `tuple`, optional
            Size of the hidden units

        Updates
        -------
        self.hbias[] : sequence of `theano.shared()`
        self.model_params[name] : OrderedDict of `theano.shared()`
        """
        if shp_hidden is None:
            try:
                shp_hidden = self.hyperparameters['n_hidden']
            except KeyError as e:
                print("hidden unit shape not defined!")
        else:
            self.hyperparameters['n_hidden'] = shp_hidden
        if name in self.model_values.keys():
            hbias = theano.shared(
                value=self.model_values[name],
                name=name,
                borrow=True
            )
        else:
            hbias = theano.shared(
                value=np.zeros(shape=shp_hidden, dtype=theano.config.floatX),
                name='hbias',
                borrow=True
            )

        self.hbias.append(hbias)
        self.model_params[name] = hbias

    def add_node(self, var_dtype, name, shp_visible=None):
        """
        add_node func

        Parameters
        ----------
        var_dtype : `str`
            Type of variables e.g. 'binary', 'category',
            see hyperparameters for more information
        name : `str`
            Name of visible node e.g. 'age'
        shp_visible : `tuple`, optional
            Size of the visible units

        Updates
        -------
        self.input[] : sequence of `T.tensor3()`\n
        self.input_dtype[] : sequence of `str`\n
        self.W_params[] : sequence of `theano.shared()`\n
        self.vbias[] : sequence of `theano.shared()`\n
        self.model_params['x_'+name] : OrderedDict of `theano.shared()`\n
        """
        try:
            var_dtype in self.hyperparameters['variable_dtypes']
        except KeyError as e:
            print("variable dtype {0:s} not implemented!".format(var_dtype))

        if shp_visible is None:
            try:
                shp_visible = self.hyperparameters['shapes'][name]
            except KeyError as e:
                print("input data shape missing!")
        else:
            self.hyperparameters['shapes'][name] = shp_visible

        shp_hidden = self.hyperparameters['n_hidden']
        if len(shp_visible) == 2:
            # input tensor as (rows, items, cats) / (rows, items, values)
            tsr_variable = T.tensor3(name)
        else:
            # input tensor as (rows, values)
            tsr_variable = T.matrix(name)
            print("Warning! inconsistent tensor: ndim=2, recommended ndim=3")

        # Create the tensor shared variables as (items, cats, hiddens)
        if 'W_' + name in self.model_values.keys():
            size = shp_visible + shp_hidden
            W_flat = theano.shared(
                value=self.model_values['W_'+name],
                name='W_'+name,
                borrow=True
            )
            W = T.reshape(W_flat, size)
        else:
            size = shp_visible + shp_hidden
            W_flat = theano.shared(
                # value=np.zeros(
                #     shape=(np.prod(size),),
                #     dtype=theano.config.floatX
                # ),
                value=np.random.normal(
                    loc=0., scale=0.1, size=np.prod(size)),
                name='W_'+name,
                borrow=True
            )
            W = T.reshape(W_flat, size)
        if 'vbias_' + name in self.model_values.keys():
            size = shp_visible
            vbias_flat = theano.shared(
                value=self.model_values['vbias_'+name],
                name='vbias_'+name,
                borrow=True
            )
            vbias = T.reshape(vbias_flat, size)
        else:
            size = shp_visible
            vbias_flat = theano.shared(
                value=np.zeros(
                    shape=np.prod(shp_visible),
                    dtype=theano.config.floatX
                ),
                name='vbias_'+name,
                borrow=True
            )
            vbias = T.reshape(vbias_flat, size)

        self.input.append(tsr_variable)
        self.input_dtype.append(var_dtype)
        self.W_params.append(W)
        self.V_params.append(W)
        self.vbias.append(vbias)
        self.W_params_flat.append(W_flat)
        self.V_params_flat.append(W_flat)
        self.vbias_flat.append(vbias_flat)
        self.model_params['W_' + name] = W_flat
        self.model_params['vbias_' + name] = vbias_flat

    def add_connection_to(self, var_dtype, name, shp_output=None):
        """
        add_connection_to func

        Parameters
        ----------
        var_dtype : `str`
            Type of variables e.g. `'binary'`, `'category'`, see
            hyperparameters for more information
        name : `str`
            Name of visible node e.g. `'mode_prime'`
        shp_output : `tuple`, optional
            Size of the visible units

        Updates
        -------
        self.output[] : sequence of `T.matrix()`
        self.W_params[] : sequence of `theano.shared()`
        self.cbias[] : sequence of `theano.shared()`
        self.B_params[] : sequence of `theano.shared()`
        self.model_params[] : sequence of `theano.shared()`
        """
        try:
            var_dtype in self.hyperparameters['variable_dtypes']
        except KeyError as e:
            print("variable dtype {0:s} not implemented!".format(var_dtype))

        if shp_output is None:
            try:
                shp_output = self.hyperparameters['shapes'][name]
            except KeyError as e:
                print("output data shape missing!")
        else:
            self.hyperparameters['shapes'][name] = shp_output

        shp_hidden = self.hyperparameters['n_hidden']
        if len(shp_output) == 2:
            # input tensor as (rows, items, cats) / (rows, items, values)
            tsr_variable = T.tensor3(name)
        else:
            # input tensor as (rows, values)
            tsr_variable = T.matrix(name)
            print("Warning! inconsistent tensor: ndim=2, recommended ndim=3")
        tsr_label = T.ivector(name + '_label')  # 1D vector of [int] labels

        # Create the tensor shared variables as (items, outs, hiddens)
        if 'W_' + name in self.model_values.keys():
            size = shp_output + shp_hidden
            W_flat = theano.shared(
                value=self.model_values['W_'+name],
                name='W_'+name,
                borrow=True
            )
            W = T.reshape(W_flat, size)
        else:
            size = shp_output + shp_hidden
            W_flat = theano.shared(
                # value=np.zeros(
                #     shape=(np.prod(size),),
                #     dtype=theano.config.floatX
                # ),
                value=np.random.normal(
                    loc=0., scale=0.1, size=np.prod(size)),
                name='W_'+name,
                borrow=True
            )
            W = T.reshape(W_flat, size)
        if 'cbias_' + name in self.model_values.keys():
            size = shp_output
            cbias_flat = theano.shared(
                value=self.model_values['cbias_'+name],
                name='cbias_'+name,
                borrow=True
            )
            cbias = T.reshape(cbias_flat, size)
        else:
            size = shp_output
            cbias_flat = theano.shared(
                value=np.zeros(
                    shape=np.prod(size),
                    dtype=theano.config.floatX
                ),
                name='cbias_'+name,
                borrow=True
            )
            cbias = T.reshape(cbias_flat, size)

        self.output.append(tsr_variable)
        self.output_dtype.append(var_dtype)
        self.label.append(tsr_label)
        self.W_params.append(W)
        self.U_params.append(W)
        self.cbias.append(cbias)
        self.W_params_flat.append(W_flat)
        self.U_params_flat.append(W_flat)
        self.cbias_flat.append(cbias_flat)
        self.model_params['W_' + name] = W_flat
        self.model_params['cbias_' + name] = cbias_flat

        # condtional RBM connection (B weights)
        for node in self.input:
            var_name = node.name
            shp_visible = self.hyperparameters['shapes'][var_name]

            # Create the tensor shared variables as (items, cats, items, outs)
            if 'B_' + var_name in self.model_values.keys():
                size = shp_visible + shp_output
                B_flat = theano.shared(
                    value=self.model_values['B_'+var_name],
                    name='B_'+var_name+'_'+name,
                    borrow=True
                )
                B = T.reshape(B_flat, size)
            else:
                size = shp_visible + shp_output
                B_flat = theano.shared(
                    # value=np.zeros(
                    #     shape=(np.prod(size),),
                    #     dtype=theano.config.floatX
                    # ),
                    value=np.random.normal(
                        loc=0., scale=0.1, size=np.prod(size)),
                    name='B_'+var_name+'_'+name,
                    borrow=True
                )
                B = T.reshape(B_flat, size)

            self.B_params.append(B)
            self.B_params_flat.append(B_flat)
            self.model_params['B_' + var_name + '_' + name] = B_flat

    def free_energy(self, input=None):
        """
        free_energy func

        Parameters
        ----------
        self : RBM class object

        input : `[T.tensors]`, optional
            Used when calculating free energy of gibbs chain sampling

        Returns
        -------
        F(y,x) :
            Scalar value of the generative model free energy

        :math:
        `F(y,x,h) = -(xWh + yWh + vbias*x + hbias*h + cbias*y)`\n
        `    wx_b = xW + yW + hbias`\n
        `  F(y,x) = -{vbias*x + cbias*y + sum_k[ln(1+exp(wx_b))]}`\n

        """
        # amend input if given an input. e.g. free_energy(chain_end)
        if input is None:
            visibles = self.input + self.output
        else:
            visibles = input
        dtypes = self.input_dtype + self.output_dtype
        hbias = self.hbias[0]
        vbiases = self.vbias + self.cbias
        W_params = self.W_params

        # input shapes as (rows, items, cats) or (rows, outs)
        # weight shapes as (items, cats, hiddens) or (outs, hiddens)
        # bias shapes as (items, cats) or (outs,)

        # wx_b = hbias : (hiddens,) broadcast(T,F) --> (rows, hiddens)
        wx_b = hbias
        utility = 0  # (rows,)
        for dtype, v, W, vbias in zip(dtypes, visibles, W_params, vbiases):
            if dtype is VARIABLE_DTYPE_REAL:
                vbias = vbias.dimshuffle('x', 0, 1)
                # utility = sum_{i} 0.5(v-vbias)^2 : (rows,)
                utility += T.sum(T.sqr(v - vbias) / 2., axis=(1, 2))
            else:
                # utility = v.vbias : (rows,)
                utility += T.tensordot(v, vbias, axes=[[1, 2], [0, 1]])

            if W.ndim == 2:
                # wx_b = vW + hbias : (rows, hiddens)
                wx_b += T.dot(v, W)
            else:
                # wx_b = vW + hbias : (rows, hiddens)
                wx_b += T.tensordot(v, W, axes=[[1, 2], [0, 1]])

        # utility --> (rows,)
        # ...axis=1) sums over hidden axis --> (rows,)
        entropy = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return - (utility + entropy)

    def discriminative_free_energy(self, input=None):
        """
        discriminative_free_energy func
            The correct output is p(y|x)

        Parameters
        ----------
        self : RBM class object

        input : `[T.tensors]`, optional
            Used when calculating free energy of gibbs chain sampling

        Returns
        -------
        F(y|x) :
            A `list[]` of vectors of the discriminative model free energy
            for each output node. Negative loglikelihood can be used as the
            objective function.

        Notes
        -----
        The free energy for the discriminative model is computed as:

        :math:
        `F(y,x,h) = -(xWh + yWh + yBx + vbias*x + hbias*h + cbias*y)`\n
        `    wx_b = xW_{ik} + yW_{jk} + hbias`\n
        `  F(y,x) = -{cbias*y + yBx + sum_k[ln(1+exp(wx_b))]}`\n
        `  F(y|x) = -{cbias + Bx + sum_k[ln(1+exp(wx_b)]}`\n
        `  F(y|x) = -{cbias + Bx + hbias + yWh}`\n

        :params: used are W^1, W^2, B, c, h biases

        """
        # amend input if given an input. e.g. free_energy(chain_end)
        if input is None:
            visibles = self.input
        else:
            visibles = input
        hbias = self.hbias[0]
        cbiases = self.cbias
        vbias = self.vbias
        xWh_params = self.V_params
        hWy_params = self.U_params  # (items, outs, hiddens)
        B_params = self.B_params

        # rebroadcast hidden unit biases
        # (hiddens,) broadcast(T, F, T) --> ('x', hiddens, 'x')
        wx_b = hbias.dimshuffle('x', 0, 'x')
        utility = []

        for cbias in cbiases:
            # (items, outs) --> ('x', outs)
            # utility = [cbias,...]  ('x', outs)
            cbias = T.sum(cbias, axis=0)
            u = cbias.dimshuffle('x', 0)
            utility.append(u)

        # loop over all input nodes
        # x : input variables
        # W, B : weights
        # a : input biases
        for x, xWh, B in zip(visibles, xWh_params, B_params):
            # matrix dot product between input variables and hidden units
            # xw = xW_{ik} : (rows, hiddens)
            # wx_b = xW_{ik} + hbias : (rows, hiddens) --> (rows, hids, 'x')
            if xWh.ndim == 2:
                xw = T.dot(x, xWh)
                wx_b += xw.dimshuffle(0, 1, 'x')
            else:
                xw = T.tensordot(x, xWh, axes=[[1, 2], [0, 1]])
                wx_b += xw.dimshuffle(0, 1, 'x')

            # loop over all output nodes
            # hWy : weights (items, outs, hiddens)
            for i, hWy in enumerate(hWy_params):
                # wx_b = W_{jk} + W_{jk} + hbias : (rows, hiddens, outs)
                hWy = T.sum(hWy, axis=0)
                wx_b += hWy.dimshuffle('x', 1, 0)
                # xB : (rows, items, cats) . (items, cats, items, outs)
                # utility[i] = cbias + Bx : (rows, outs)
                # utility[i] = cbias + Bx : (rows, outs)
                utility[i] += T.tensordot(
                    x, T.sum(B, axis=-2),
                    axes=[[1, 2], [0, 1]]
                )

        # sum over hiddens axis
        # sum_k \ln(1+\exp(wx_b)) : (rows, hiddens, outs) -- > (rows, outs)
        entropy = T.sum(T.log(1 + T.exp(wx_b)), axis=1)

        # add entropy to each expected utility term
        # -F(y|x)  (rows, outs)
        energy = []
        for u in utility:
            energy.append(u+entropy)

        return energy

    def sample_h_given_v(self, v0_samples):
        """
        sample_h_given_v func
            Binomial hidden units

        Parameters
        ----------
        v0_samples : `[T.tensors]`
            theano Tensor variable

        Returns
        -------
        h1_preactivation : `scalar` (-inf, inf)
            preactivation function e.g. logit utility func
        h1_means : `scalar` (0, 1)
            sigmoid activation
        h1_samples : `integer` 0 or 1
            binary samples
        """
        # prop up
        W_params = self.W_params
        hbias = self.hbias
        h1_preactivation = self.propup(v0_samples, W_params, hbias[0])

        # h ~ p(h|v0_sample)
        h1_means = T.nnet.sigmoid(h1_preactivation)
        h1_samples = self.theano_rng.binomial(
            size=h1_means.shape,
            p=h1_means,
            dtype=theano.config.floatX
        )

        return h1_preactivation, h1_means, h1_samples

    def propup(self, samples, weights, bias):

        preactivation = bias
        # (rows, items, cats), (items, cats, hiddens)
        # (rows, outs), (outs, hiddens)
        for v, W, in zip(samples, weights):
            if W.ndim == 2:
                preactivation += T.dot(v, W)
            else:
                preactivation += T.tensordot(v, W, axes=[[1, 2], [0, 1]])

        return preactivation

    def sample_v_given_h(self, h0_samples):
        """
        sample_v_given_h func
            Binomial hidden units

        Parameters
        ----------
        h0_samples : `[T.tensors]`
            theano Tensor variable

        Returns
        -------
        v1_preactivation : `[scalar]` (-inf, inf)
            sequence of preactivation function e.g. logit utility func
        v1_means : `[scalar]` (0, 1)
            sequence of sigmoid activation
        v1_samples : `[binary]` or `[integer]` or `[float32]` or `[array[j]]`
            visible unit samples
        """
        # prop down
        W_params = self.W_params
        bias = self.vbias + self.cbias
        v1_preactivation = self.propdown(h0_samples, W_params, bias)

        # v ~ p(v|h0_sample)
        v1_means = []
        v1_samples = []
        dtypes = self.input_dtype + self.output_dtype
        for v1, dtype in zip(v1_preactivation, dtypes):
            if dtype is VARIABLE_DTYPE_BINARY:
                v1_mean = T.nnet.sigmoid(v1)
                v1_sample = self.theano_rng.binomial(
                    size=v1.shape,
                    p=v1_mean,
                    dtype=theano.config.floatX
                )

            elif dtype is VARIABLE_DTYPE_CATEGORY:
                # softmax temperature value \tau (default=1)
                tau = 1. / v1.shape[-1]
                epsilon = 1e-10  # small value to prevent log(0)
                uniform = self.theano_rng.uniform(
                    size=v1.shape,
                    dtype=theano.config.floatX
                )
                gumbel = - (- T.log(uniform + epsilon) + epsilon)
                # reshape softmax tensors to 2D matrix
                if v1.ndim == 3:
                    (d1, d2, d3) = v1.shape
                    logit = (v1 + gumbel).reshape((d1 * d2, d3))
                    v1_mean = T.nnet.softmax(logit / tau)
                    # reshape back into original dimensions
                    v1_mean = v1_mean.reshape((d1, d2, d3))
                else:
                    logit = (v1 + gumbel)
                    v1_mean = T.nnet.softmax(logit/tau)  # (rows, items, cats)
                v1_sample = v1_mean

            elif dtype is VARIABLE_DTYPE_REAL:
                v1_mean = v1
                v1_std = T.nnet.sigmoid(v1)
                normal_sample = self.theano_rng.normal(
                    size=v1_mean.shape,  # (rows, items, cats)
                    avg=v1_mean,
                    std=v1_std,
                    dtype=theano.config.floatX
                )
                v1_sample = T.nnet.relu(normal_sample)

            elif dtype is VARIABLE_DTYPE_INTEGER:
                if self.hyperparameters['noisy_rectifier'] is True:
                    v1_mean = T.nnet.sigmoid(v1)
                    normal_sample = self.theano_rng.normal(
                        size=v1.shape,
                        avg=v1,
                        std=v1_mean,
                        dtype=theano.config.floatX
                    )
                    v1_sample = T.nnet.relu(normal_sample)
                else:
                    # slower implementation of NReLu but more accurate
                    # values and samples exact integers from v1
                    N = 200
                    offset = - np.arange(1, N) + 0.5
                    # (rows, items, cats, Ns)
                    v1 = T.shape_padright(v1) + offset
                    v1_mean = T.nnet.sigmoid(v1)
                    binomial = self.theano_rng.binomial(
                        size=v1.shape,
                        p=v1_mean,
                        dtype=theano.config.floatX
                    )
                    # (rows, items, cats)
                    v1_sample = T.sum(binomial, axis=-1)

            else:
                raise NotImplementedError

            v1_means.append(v1_mean)
            v1_samples.append(v1_sample)

        return v1_preactivation, v1_means, v1_samples

    def propdown(self, samples, weights, bias):

        preactivation = []
        # (rows, hiddens), (items, cats, hiddens) --> dimshuffle(0, 2, 1)
        # (rows, hiddens), (outs, hiddens) --> dimshuffle(1, 0)
        for W, b in zip(weights, bias):
            if W.ndim == 2:
                W = W.dimshuffle(1, 0)
            else:
                W = W.dimshuffle(0, 2, 1)
            preactivation.append(T.dot(samples, W) + b)

        return preactivation

    def gibbs_hvh(self, h0_samples):
        v1_pre, v1_means, v1_samples = self.sample_v_given_h(h0_samples)
        h1_pre, h1_means, h1_samples = self.sample_h_given_v(v1_samples)

        gibbs_scan_list = v1_pre + v1_means + v1_samples \
            + [h1_pre] + [h1_means] + [h1_samples]
        return gibbs_scan_list

    def gibbs_vhv(self, v0_samples):
        h1_pre, h1_means, h1_samples = self.sample_h_given_v(v0_samples)
        v1_pre, v1_means, v1_samples = self.sample_v_given_h(h1_samples)

        gibbs_scan_list = [h1_pre] + [h1_means] + [h1_samples] \
            + v1_pre + v1_means + v1_samples
        return gibbs_scan_list

    def get_generative_cost_updates(self, k=1, lr=1e-3):
        """
        get_generative_cost_updates func
            updates weights for W^(1), W^(2), a, c and d
        """
        logits = self.discriminative_free_energy()
        labels = self.label
        y0_samples = []
        dcost = 0
        for i, (logit, label) in enumerate(zip(logits, labels)):
            # small value for tau to minic argmax but with differentiable
            # gradients
            tau = 1. / logit.shape[-1]
            epsilon = 1e-8  # small value to prevent log(0)
            uniform = self.theano_rng.uniform(
                size=logit.shape,
                dtype=theano.config.floatX
            )
            gumbel = - (- T.log(uniform + epsilon) + epsilon)
            # reshape softmax tensors to 2D matrix
            if logit.ndim == 3:
                (d1, d2, d3) = logit.shape
                y0 = (logit + gumbel).reshape((d1 * d2, d3))
                y0_mean = T.nnet.softmax(y0 / tau)
                p_y_given_x = T.nnet.softmax(logit.reshape((d1 * d2, d3)))
                # reshape back into original dimensions
                y0_mean = y0_mean.reshape((d1, d2, d3))
                p_y_given_x = p_y_given_x.reshape((d1, d3))
            else:
                y0 = (logit + gumbel)
                y0_mean = T.nnet.softmax(y0 / tau)  # (rows, outs)
                p_y_given_x = T.nnet.softmax(logit)
                y0_mean = y0_mean.dimshuffle(0, 'x', 1)
            y0_samples.append(y0_mean)
            dcost += self.get_loglikelihood(p_y_given_x, label)

        # prepare visible samples from x input and y outputs
        # v0_samples = self.input + self.output
        v0_samples = self.input + y0_samples

        # perform positive Gibbs sampling phase
        # one step Gibbs sampling p(h|v1,v2,...) = p(h|v1)+p(h|v2)+...
        h1_pre, h1_means, h1_samples = self.sample_h_given_v(v0_samples)
        # start of Gibbs sampling chain
        # we only want the samples generated from the Gibbs sampling phase
        chain_start = h1_samples
        scan_out = 3 * len(v0_samples) * [None] + [None, None, chain_start]

        # theano scan function to loop over all Gibbs steps k
        # [v1_pre[], v1_means[], v1_samples[], h1_pre, h1_means, h1_samples]
        # outputs are given by outputs_info
        # [[t,t+1,t+2,...], [t,t+1,t+2,...], ], gibbs_updates
        # NOTE: scan returns a dictionary of updates
        gibbs_output, gibbs_updates = theano.scan(
            fn=self.gibbs_hvh,
            outputs_info=scan_out,
            n_steps=k,
            name='gibbs_hvh'
        )

        # note that we only need the visible samples at the end of the chain
        chain_end = []
        for output in gibbs_output:
            chain_end.append(output[-1])

        gibbs_pre = chain_end[:len(v0_samples)]
        gibbs_means = chain_end[len(v0_samples): 2 * len(v0_samples)]
        gibbs_samples = chain_end[2 * len(v0_samples): 3 * len(v0_samples)]
        # calculate the model cost
        initial_cost = T.mean(self.free_energy())
        final_cost = T.mean(self.free_energy(gibbs_samples))
        cost = (initial_cost - final_cost) * self.hyperparameters['alpha']
        cost += dcost

        # calculate the gradients
        params = self.W_params_flat + self.hbias + self.vbias_flat \
            + self.cbias_flat \
            + self.B_params_flat
        grads = T.grad(
            cost=cost,
            wrt=params,
            consider_constant=gibbs_samples,
            disconnected_inputs='ignore'
        )
        updates = self.opt.sgd_updates(
            params=params,
            grads=grads,
            learning_rate=lr
        )
        # update Gibbs chain with update expressions from updates list[]
        for variable, expression in updates:
            gibbs_updates[variable] = expression

        # pseudo loglikelihood to track the quality of the hidden units
        # on input variables ONLY
        monitoring_cost = self.pseudo_loglikelihood(
            inputs=self.input,
            preactivation=gibbs_pre[:len(self.input)])

        return [monitoring_cost], gibbs_updates

    def get_discriminative_cost_updates(self, lr=1e-3):

        # prepare visible samples from x input
        v0_samples = self.input
        labels = self.label
        dtypes = self.output_dtype

        logits = self.discriminative_free_energy()
        cost = []
        updates = OrderedDict()
        for i, (logit, label) in enumerate(zip(logits, labels)):
            p_y_given_x = T.nnet.softmax(logit)
            cost.append(self.get_loglikelihood(p_y_given_x, label))
            # calculate the gradients
            params = [self.B_params_flat[i]] + [self.cbias_flat[i]]
            grads = T.grad(
                cost=cost[i],
                wrt=params,
                disconnected_inputs='ignore'
            )
            # a list of update expressions (variable, update expression)
            update = self.opt.sgd_updates(
                params=params,
                grads=grads,
                learning_rate=lr
            )
            for var, expr in update:
                if var in updates:
                    updates[var] = expr + updates[var]
                else:
                    updates[var] = expr

        return cost, updates

    def pseudo_loglikelihood(self, inputs, preactivation):
        dtypes = self.input_dtype
        v1_post = 0
        for input, v1, dtype in zip(inputs, preactivation, dtypes):
            if dtype is VARIABLE_DTYPE_BINARY:
                v1_post -= T.mean(input * T.log(T.nnet.sigmoid(v1)))

            elif dtype is VARIABLE_DTYPE_CATEGORY:
                tau = 0.5  # softmax temperature value \tau (default=1)
                epsilon = 1e-10  # small value to prevent log(0)
                if v1.ndim == 3:
                    (d1, d2, d3) = v1.shape
                    logit = v1.reshape((d1 * d2, d3))
                    v1_mean = T.nnet.softmax(logit / tau)
                    # reshape back into original dimensions
                    v1_mean = v1_mean.reshape((d1, d2, d3))
                    v1_post -= T.mean(input * T.log(v1_mean))
                else:
                    logit = v1
                    v1_mean = T.nnet.softmax(logit / tau)
                    v1_post -= T.mean(input * T.log(v1_mean))

            elif dtype is VARIABLE_DTYPE_REAL:
                v1_post -= 0

            elif dtype is VARIABLE_DTYPE_INTEGER:
                raise NotImplementedError

            else:
                raise NotImplementedError
        return T.sum(v1_post)

    def get_loglikelihood(self, prob, label):
        """
        get_sample_loglikelihood func
            Approximation to the reconstruction error
        """
        # TODO: loglikelihood loss of the discriminative (output) samples
        # return negative loglikelihood
        neg_ll = -T.mean(T.log(prob)[T.arange(label.shape[0]), label])
        return neg_ll

    def get_t_statistcs(self):

        logits = self.discriminative_free_energy()
        for i, logit in enumerate(logits):
            p_y_given_x = T.nnet.softmax(-logit)
            cost = self.get_loglikelihood(p_y_given_x, self.label[i])
            # calculate the gradients
            params = [self.B_params_flat[i]] + [self.cbias_flat[i]]
            hessians = T.hessian(
                cost=cost,
                wrt=params,
                disconnected_inputs='ignore'
            )
            cramer_rao_bound = [T.diag(-1. / h) for h in hessians]
            sigma = [T.sqrt(cr) for cr in cramer_rao_bound]
            t_stat = [p / s for p, s in zip(params, sigma)]

        return hessians

    def predict(self):

        # prepare visible samples from x input
        v0_samples = self.input
        logits = self.discriminative_free_energy()
        preds = []
        for i, logit in enumerate(logits):
            p_y_given_x = T.nnet.softmax(-logit)
            pred = T.argmax(p_y_given_x, axis=1)
            preds.append(pred)

        return preds

    def build_fn(self, train_set_x, train_set_y, train_set_labels):
        """
        build_fn func
        # TODO

        Parameters
        ----------
        """
        print('building theano computational graphs')
        index = T.lscalar()
        tensor_inputs = self.input + self.output
        data_inputs = train_set_x + train_set_y
        labels = self.label

        lr = self.hyperparameters['learning_rate']
        k = self.hyperparameters['gibbs_steps']
        batch_size = self.hyperparameters['batch_size']

        gibbs_cost, gibbs_updates = self.get_generative_cost_updates(k, lr)
        cost, updates = self.get_discriminative_cost_updates(lr)
        # for var, expression in updates.items():
        #     gibbs_updates[var] = expression
        # preds = self.predict()
        # flatten list of preds in case of out variables > 1
        # preds = list(chain.from_iterable(preds))
        # hessians = self.get_t_statistcs()

        self.train = theano.function(
            inputs=[index],
            outputs=gibbs_cost+cost,
            updates=gibbs_updates,
            givens={
                tsr: data[index * batch_size: (index + 1) * batch_size]
                for tsr, data in zip(tensor_inputs + labels,
                                     data_inputs + train_set_labels)
            },
            name='train',
            allow_input_downcast=True,
            on_unused_input='ignore'
        )


def main(rbm):
    # load variables from csv

    x, y, n_samples = Setup.load_variables('datatable.csv')

    rbm.hyperparameters['n_samples'] = n_samples
    rbm.add_latent(shp_hidden=(16,))

    for var in x:
        rbm.add_node(
            var_dtype=x[var]['dtype'],
            name=var,
            shp_visible=x[var]['shape']
        )
    for var in y:
        rbm.add_connection_to(
            var_dtype=y[var]['dtype'],
            name=var,
            shp_output=y[var]['shape']
        )

    train_set_x = [x[var]['data'] for var in x]
    train_set_y = [y[var]['data'] for var in y]
    train_set_label = [y[var]['label'] for var in y]

    rbm.build_fn(train_set_x, train_set_y, train_set_label)

    epochs = 50
    print('training the model...')
    n_train_batches = (rbm.hyperparameters['n_samples'] //
                       rbm.hyperparameters['batch_size'])
    epoch = 0
    while epoch < epochs:
        epoch += 1
        cost = []
        for minibatch_index in range(n_train_batches):
            cost_items = rbm.train(minibatch_index)
            cost.append(cost_items)
            iter = (epoch - 1) * n_train_batches + minibatch_index
        epoch_cost = np.asarray(cost).sum(axis=0)
        print(
            ("epoch {0:d} batch {1:d}/{2:d} gibbs cost: {3:.3f},"
                " loglikelihood cost {4:.3f}").format(
                epoch, minibatch_index + 1, n_train_batches,
                epoch_cost[0], epoch_cost[1])
        )
        # curves = {'CD error': [], 'log likelihood': []}
        rbm.monitoring_curves['CD error'].append((iter, epoch_cost[0]))
        rbm.monitoring_curves['log likelihood'].append((iter, epoch_cost[1]))
    rbm.save_params(iter)
    rbm.plot_curves()
    print('train complete')

if __name__ == '__main__':
    main(RBM(*net))
