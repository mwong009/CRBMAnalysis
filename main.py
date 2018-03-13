import theano
import pickle
import sys
import os
import numpy as np
import theano.tensor as T
import optimizers as opt

from collections import OrderedDict as odict
from itertools import chain

net = 'net1', {
    'n_hidden': (16,),
    'seed': 42,
    'batch_size': 32,
    'variable_types': ['binary', 'integer', 'real', 'category'],
    'noisy_rectifier': True,
    'learning rate': 1e-3,
    'gibbs steps': 1
}


class Network(object):
    def __init__(self, name, hyperparameters=odict()):

        self.path = name

        seed = hyperparameters['seed']
        self.np_rng = np.random.RandomState(seed)
        self.theano_rng = T.shared_randomstreams.RandomStreams(seed)

        model_values, hyper, curves = self.load_params(name, hyperparameters)
        self.model_values = model_values
        self.hyperparameters = hyper
        self.monitoring_curves = curves
        self.model_params = odict()

    def save_params(self):

        path = self.path+'.params'
        hyper = self.hyperparameters
        curves = self.monitoring_curves
        model_values = {}
        # evaluating tensor shared variable to numpy array
        for param_name, param in self.model_params:
            model_values[param_name] = param.eval()

        to_file = model_values, hyper, curves
        with open(path, 'wb') as f:
            pickle.dump(to_file, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_params(self, name, hyperparameters):
        """
        load_params func

        Parameters
        ----------
        name: `str`
            Name of the hyperparmeter\n
        hyperparameters : `dict{}`
            dictionary of the hyperparameters\n

        Returns
        -------
        model_values : `{key: value}` pair of saved parameter values`\n
        hyperparameters : Updated list of hyperparameters`\n
        curves: monitoring curves\n
        """
        path = name + '.params'
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                model_values, hyper, curves = pickle.load(f)

            # update hyperparameters from init
            for key, value in hyper.items():
                hyperparameters[key] = value

        else:
            model_values = {}
            curves = {
                'CD error': [],
                'log likelihood': []
            }

        return model_values, hyperparameters, curves


class RBM(Network):
    ''' define the RBM toplevel '''
    def __init__(self, name, hyperparameters=odict()):
        Network.__init__(self, name, hyperparameters)
        self.input = []         # list of tensors
        self.input_type = []    # list of str types
        self.output = []        # list of tensors
        self.output_type = []   # list of str types
        self.label = []         # list of label tensors
        self.hbias = []
        self.W_params = []      # list of ALL the W params
        self.V_params = []      # list of the xWh params
        self.U_params = []      # list of the hWy params
        self.B_params = []
        self.vbias = []
        self.cbias = []
        # flatten version
        self.W_params_flat = []      # list of ALL the W params
        self.V_params_flat = []      # list of the xWh params
        self.U_params_flat = []      # list of the hWy params
        self.B_params_flat = []
        self.vbias_flat = []
        self.cbias_flat = []

        self.add_hbias()

    def add_hbias(self, name='hbias'):
        """
        add_hbias func

        Parameters
        ----------
        name: `str`, optional
            Name of hidden node e.g. `'hbias'`

        Updates
        -------
        self.hbias[] : sequence of `theano.shared()`\n
        self.model_params[name] : OrderedDict of `theano.shared()`\n
        """
        shp_hidden = self.hyperparameters['n_hidden']
        if name in self.model_values.keys():
            hbias = theano.shared(
                value=self.model_values[name],
                name=name,
                borrow=True
            )
        else:
            hbias = theano.shared(
                value=np.zeros(
                    shape=shp_hidden,
                    dtype=theano.config.floatX
                ),
                name='hbias',
                borrow=True
            )

        self.hbias.append(hbias)
        self.model_params[name] = hbias

    def add_node(self, var_type, name):
        """
        add_node func

        Parameters
        ----------
        var_type : `str`
            Type of variables e.g. 'binary', 'category',
            see hyperparameters for more information
        name : `str`, optional
            Name of visible node e.g. 'age'

        Updates
        -------
        self.input[] : sequence of `T.tensor3()`\n
        self.input_type[] : sequence of `str`\n
        self.W_params[] : sequence of `theano.shared()`\n
        self.vbias[] : sequence of `theano.shared()`\n
        self.model_params['x_'+name] : OrderedDict of `theano.shared()`\n
        """
        if var_type not in self.hyperparameters['variable_types']:
            print('variable type \'%s\' not implemented!' % t)
            raise NotImplementedError

        shp_hidden = self.hyperparameters['n_hidden']
        shp_visible = self.hyperparameters[name]
        tsr_variable = T.tensor3(name)  # input tensor as (rows, items, cats)

        # Create the tensor shared variables as (items, cats, hiddens)
        if 'W_'+name in self.model_values.keys():
            size = shp_visible+shp_hidden
            W_flat = theano.shared(
                value=self.model_values['W_'+name],
                name='W_'+name,
                borrow=True
            )
            W = W_flat.reshape(size)
        else:
            size = shp_visible+shp_hidden
            W_flat = theano.shared(
                value=np.random.uniform(
                    low=-np.sqrt(6./np.sum(size)),
                    high=np.sqrt(6./np.sum(size)),
                    size=np.prod(size)
                ),
                name='W_'+name,
                borrow=True
            )
            W = W_flat.reshape(size)
        if 'vbias_'+name in self.model_values.keys():
            size = shp_visible
            vbias_flat = theano.shared(
                value=self.model_values['vbias_'+name],
                name='vbias_'+name,
                borrow=True
            )
            vbias = vbias_flat.reshape(size)
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
            vbias = vbias_flat.reshape(size)

        self.input.append(tsr_variable)
        self.input_type.append(var_type)
        self.W_params.append(W)
        self.V_params.append(W)
        self.vbias.append(vbias)
        self.W_params_flat.append(W_flat)
        self.V_params_flat.append(W_flat)
        self.vbias_flat.append(vbias_flat)
        self.model_params['W_'+name] = W_flat
        self.model_params['vbias_'+name] = vbias_flat

    def add_connection_to(self, var_type, name='output'):
        """
        add_connection_to func

        Parameters
        ----------
        var_type : `str`
            Type of variables e.g. `'binary'`, `'category'`,
            see hyperparameters for more information
        name : string, optional
            Name of visible node e.g. `'mode_prime'`

        Updates
        -------
        self.output[] : sequence of `[T.matrix(), str]`\n
        self.W_params[] : sequence of `theano.shared()`\n
        self.cbias[] : sequence of `theano.shared()`\n
        self.B_params[] : sequence of `theano.shared()`\n
        self.model_params[] : sequence of `theano.shared()`\n
        """
        if var_type not in self.hyperparameters['variable_types']:
            print('variable type \'%s\' not implemented!' % t)
            raise NotImplementedError

        shp_hidden = self.hyperparameters['n_hidden']
        shp_output = self.hyperparameters[name]
        tsr_variable = T.matrix(name)  # output tensor as (rows, outs)
        tsr_label = T.ivector(name+'_label')  # 1D vector of [int] labels

        # Create the tensor shared variables as (outs, hiddens)
        if 'W_'+name in self.model_values.keys():
            size = shp_output+shp_hidden
            W_flat = theano.shared(
                value=self.model_values['W_'+name],
                name='W_'+name,
                borrow=True
            )
            W = W_flat.reshape(size)
        else:
            size = shp_output+shp_hidden
            W_flat = theano.shared(
                value=np.random.uniform(
                    low=-np.sqrt(6./np.sum(size)),
                    high=np.sqrt(6./np.sum(size)),
                    size=np.prod(size)
                ),
                name='W_'+name,
                borrow=True
            )
            W = W_flat.reshape(size)
        if 'cbias_'+name in self.model_values.keys():
            size = shp_output
            cbias_flat = theano.shared(
                value=self.model_values['cbias_'+name],
                name='cbias_'+name,
                borrow=True
            )
            cbias = cbias_flat.reshape(size)
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
            cbias = cbias_flat.reshape(size)

        self.output.append(tsr_variable)
        self.output_type.append(var_type)
        self.label.append(tsr_label)
        self.W_params.append(W)
        self.U_params.append(W)
        self.cbias.append(cbias)
        self.W_params_flat.append(W_flat)
        self.U_params_flat.append(W_flat)
        self.cbias_flat.append(cbias_flat)
        self.model_params['W_'+name] = W_flat
        self.model_params['cbias_'+name] = cbias_flat

        # condtional RBM connection (B weights)
        for node in self.input:
            var_name = node.name
            shp_visible = self.hyperparameters[var_name]

            # Create the tensor shared variables as (items, cats, outs)
            if 'B_'+var_name in self.model_values.keys():
                size = shp_visible+shp_output
                B_flat = theano.shared(
                    value=self.model_values['B_'+var_name],
                    name='B_'+var_name,
                    borrow=True
                )
                B = B_flat.reshape(size)
            else:
                B_flat = theano.shared(
                    value=np.random.uniform(
                        low=-np.sqrt(6/np.sum(size)),
                        high=np.sqrt(6/np.sum(size)),
                        size=np.prod(size)
                    ),
                    name='B_'+var_name+'_'+name,
                    borrow=True
                )
                B = B_flat.reshape(size)

            self.B_params.append(B)
            self.B_params_flat.append(B_flat)
            self.model_params['B_'+var_name+'_'+name] = B_flat

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

        Notes
        -----
        The free energy for the generative model is compu
        # loop over all input nodested as:

        :math:\n
        `F(y,x,h) = -(xWh + yWh + vbias*x + hbias*h + cbias*y)`\n
        `    wx_b = xW + yW + hbias`\n
        `  F(y,x) = -{vbias*x + cbias*y + sum_k[ln(1+exp(wx_b))]}`\n

        """
        # amend input if given an input. e.g. free_energy(chain_end)
        if input is None:
            visibles = self.input + self.output
        else:
            visibles = input
        types = self.input_type + self.output_type
        hbias = self.hbias[0]
        vbiases = self.vbias + self.cbias
        W_params = self.W_params

        # input shapes as (rows, items, cats) or (rows, outs)
        # weight shapes as (items, cats, hiddens) or (outs, hiddens)
        # bias shapes as (items, cats) or (outs,)

        # wx_b = hbias : (hiddens,) broadcast(T,F) --> (rows, hiddens)
        wx_b = hbias
        utility = 0  # (rows,)
        for t, v, W, vbias in zip(types, visibles, W_params, vbiases):
            if t is 'real':
                vbias = vbias.dimshuffle('x', 0, 1)
                # utility = sum_{i} 0.5(v-vbias)^2 : (rows,)
                utility += T.sum(T.sqr(v - vbias)/2., axis=(1, 2))
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
        entropy = T.sum(T.log(1+T.exp(wx_b)), axis=1)
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

        :math:\n
        `F(y,x,h) = -(xWh + yWh + yBx + vbias*x + hbias*h + cbias*y)`\n
        `    wx_b = xW_{ik} + yW_{jk} + hbias`\n
        `  F(y,x) = -{vbias*x + cbias*y + yBx + sum_k[ln(1+exp(wx_b))]}`\n
        `  F(y|x) = -{vbias*x + cbias + Bx + sum_k[ln(1+exp(wx_b)]}`

        :params: used are W^1, W^2, B, c, h, v biases

        """
        # amend input if given an input. e.g. free_energy(chain_end)
        if input is None:
            visibles = self.input
        else:
            visibles = input
        types = self.input_type
        hbias = self.hbias[0]
        cbias = self.cbias
        vbias = self.vbias
        xWh_params = self.W_params[:len(visibles)]
        hWy_params = self.W_params[len(visibles):]
        B_params = self.B_params

        # rebroadcast hidden unit biases
        # (hiddens,) broadcast(T, F, T) --> ('x', hiddens, 'x')
        wx_b = hbias.dimshuffle('x', 0, 'x')
        utility = []

        # loop over all input nodes
        # x : input variables
        # W, B : weights
        # a : input biases
        # t : input types
        for x, xWh, B, a, t in zip(visibles, xWh_params, B_params, vbias,
                                   types):
            # matrix dot product between input variables and hidden units
            # xw = xW_{ik} : (rows, hiddens)
            # wx_b = xW_{ik} + hbias : (rows, hiddens, outs)
            if xWh.ndim == 2:
                xw = T.dot(x, xWh)
                wx_b += xw.dimshuffle(0, 1, 'x')
            else:
                xw = T.tensordot(x, xWh, axes=[[1, 2], [0, 1]])
                wx_b += xw.dimshuffle(0, 1, 'x')

            # loop over all output nodes
            # hWy : weights
            for i, (hWy, cbias) in enumurate(zip(hWy_params, cbiases)):
                # wx_b = W_{jk} + W_{jk} + hbias : (rows, hiddens, outs)
                wx_b += hWy.dimshuffle('x', 1, 0)
                # (outs,) --> ('x', outs)
                # utility = [cbias,...]  ('x', outs)
                utility.append(cbias.dimshuffle('x', 0))
                if B_param.ndim == 2:
                    # utility[i] = cbias + Bx : (rows, outs)
                    utility[i] += T.dot(x, B)
                else:
                    # utility[i] = cbias + Bx : (rows, outs)
                    utility[i] += T.tensordot(x, B, axes=[[1, 2], [0, 1]])

                if t is 'real':
                    a = a.dimshuffle('x', 0, 1)
                    # utility = sum_{i} 0.5(v-a)^2 : (rows,)
                    vbias_term = T.sum(T.sqr(x - a)/2., axis=(1, 2))
                else:
                    # utility = v.vbias : (rows,)
                    vbias_term = T.tensordot(x, a, axes=[[1, 2], [0, 1]])

                # vbias : (rows,) --> (rows, 'x')
                # utility = cbias + Bx + vbias
                utility[i] += vbias_term.dimshuffle(0, 'x')

        # sum over hiddens axis
        # sum_k \ln(1+\exp(wx_b)) : (rows, hiddens, outs) -- > (rows, outs)
        entropy = T.sum(T.log(1+T.exp(wx_b)), axis=1)

        # add entropy to each expected utility term
        # -F(y|x)  (rows, outs)
        energy = []
        for u in utility:
            energy.append(-(u+entropy))

        return energy

    def sample_h_given_v(self, v0_samples):
        """
        sample_h_given_v func\n
            Binomial hidden units

        Parameters
        ----------
        v0_samples : `[T.tensors]`
            theano Tensor variable

        Returns
        -------
        h1_preactivation : `scalar` (-inf, inf)
            preactivation function e.g. logit utility func\n
        h1_means : `scalar` (0, 1)
            sigmoid activation\n
        h1_samples : `integer` 0 or 1
            binary samples\n
        """
        # prop up
        W_params = self.params
        hbias = self.hbias
        h1_preactivation = self.propup(v0_samples, W_params, hbias)

        # h ~ p(h|v0_sample)
        h1_means = T.nnet.sigmoid(h1_preactivation)
        h1_samples = self.theano_rng.binomial(
            size=h1_means.shape,
            p=h1_means,
            dtype=theano.config.floatX
        )

        return h1_preactivation, h1_means, h1_samples

    def propup(self, samples, weights, bias):

        preactivation = bias[0]
        # (rows, items, cats), (items, cats, hiddens)
        # (rows, outs), (outs, hiddens)
        for v, W, t in zip(samples, weights):
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
        vbias = self.vbias + self.cbias
        v1_preactivation = propdown(h0_samples, W_params, vbias)

        # v ~ p(v|h0_sample)
        v1_means = []
        v1_samples = []
        types = self.input_type + self.output_type
        for v1, t in zip(v1_preactivation, types):
            if t is 'binary':
                v1_mean = T.nnet.sigmoid(v1)
                v1_sample = self.theano_rng.binomial(
                    size=v1.shape,
                    p=v1_mean,
                    dtype=theano.config.floatX
                )

            elif t is 'category':
                tau = 1.  # softmax temperature value \tau (default=1)
                epsilon = 1e-10  # small value to prevent log(0)
                uniform = self.theano_rng.uniform(
                    size=v1.shape,
                    dtype=theano.config.floatX
                )
                gumbel = -(-T.log(uniform + epsilon) + epsilon)
                # reshape softmax tensors to 2D matrix
                if v1.ndim == 3:
                    (d1, d2, d3) = v1.shape
                    logit = (v1 + gumbel).reshape((d1 * d2, d3))
                    v1_mean = T.nnet.softmax(logit/tau)
                    # reshape back into original dimensions
                    v1_mean = v1_mean.reshape((d1, d2, d3))
                else:
                    logit = (v1 + gumbel_error)
                    v1_mean = T.nnet.softmax(logit/tau)  # (rows, items, cats)
                v1_sample = v1_mean

            elif t is 'real':
                v1_mean = v1
                v1_sample = self.theano_rng.normal(
                    size=v1_mean.shape,
                    avg=v1_mean,
                    dtype=theano.config.floatX
                )  # (rows, items, cats)

            elif t is 'integer':
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
                    offset = -np.arange(1, 200) + 0.5
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
            preactivation.append(T.dot(samples, W) + bias)

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

    def get_discriminative_cost_updates(self, lr=1e-3):

        # prepare visible samples from x input
        v0_samples = self.input
        labels = self.output_labels
        types = self.output_type

        logits = self.discriminative_free_energy()
        cost = []
        updates = odict()
        for i, (logit, label, t) in enumerate(zip(logits, labels, types)):
            if t is 'real':
                mse = T.mean((-logit-label)**2, axis=0)
                cost.append(mse)
            else:
                p_y_given_x = T.nnet.softmax(-logit)
                cost.append(self.get_loglikelihood(p_y_given_x, label))
            # calculate the gradients
            params = self.V_params + self.hbias + self.vbias \
                + self.U_params[i] + self.cbias[i] + self.B_params[i]
            grads = T.grad(cost=cost[i], wrt=params)
            # a list of update expressions (variable, update expression)
            update = opt.adam_updates(params, grads, lr, amsgrad=True)
            for var, expr in update:
                if var in updates:
                    updates[var] = (expr + updates[var]) / (i + 1.)
                else:
                    updates[var] = expr

        return cost, updates

    def predict(self):

        # prepare visible samples from x input
        v0_samples = self.input
        labels = self.output_labels

        logits = self.discriminative_free_energy()
        # s is to scale the learning rate to the number of output nodes
        s = len(logits)
        preds = []
        for i, (logit, label) in enumerate(zip(logits, labels)):
            p_y_given_x = T.nnet.softmax(-logit)
            pred = T.argmax(p_y_given_x, axis=1)
            preds.append(pred)

        return preds

    def get_generative_cost_updates(self, k=1, lr=1e-3):
        """
        get_generative_cost_updates func
            updates weights for W^(1), W^(2), a, c and d
        """

        # prepare visible samples from x input and y outputs
        v0_samples = self.input + self.output

        # perform positive Gibbs sampling phase
        # one step Gibbs sampling p(h|v1,v2,...) = p(h|v1)+p(h|v2)+...
        h1_pre, h1_means, h1_samples = self.sample_h_given_v(v0_samples)
        # start of Gibbs sampling chain
        # we only want the samples generated from the Gibbs sampling phase
        chain_start = h1_samples

        # theano scan function to loop over all Gibbs steps k
        # [v1_means[], v1_means[], h1_means, h1_samples]
        # outputs are given by outputs_info
        # [[t,t+1,t+2,...], [t,t+1,t+2,...], ], gibbs_updates
        # NOTE: scan returns a dictionary of updates
        gibbs_output, gibbs_updates = theano.scan(
            fn=self.gibbs_hvh,
            outputs_info=3*len(v0_samples)*[None]+[None, None, chain_start],
            n_steps=k,
            name='gibbs_hvh'
        )

        # note that we only need the visible samples at the end of the chain
        chain_end = []
        for output in gibbs_output:
            chain_end.append(output[-1])

        gibbs_pre = chain_end[:len(v0_samples)]
        gibbs_means = chain_end[len(v0_samples):2*len(v0_samples)]
        gibbs_samples = chain_end[2*len(v0_samples):3*len(v0_samples)]

        # calculate the model cost
        initial_cost = T.mean(self.free_energy())
        final_cost = T.mean(self.free_energy(gibbs_samples))
        cost = initial_cost - final_cost

        # calculate the gradients
        params = self.W_params + self.hbias + self.vbias + self.cbias
        grads = T.grad(
            cost=cost,
            wrt=params,
            consider_constant=gibbs_samples
        )

        updates = opt.adam_updates(params, grads, lr, amsgrad=True)

        # update Gibbs chain with update expressions
        for variable, expression in updates:
            gibbs_updates[variable] = expression

        return cost, gibbs_updates

    def get_loglikelihood(self, prob, label):
        """
        get_sample_loglikelihood func
            Approximation to the reconstruction error
        """
        # TODO: loglikelihood loss of the discriminative (output) samples
        # return negative loglikelihood
        neg_ll = -T.mean(log(prob)[T.arange(label.shape[0], label)])
        return neg_ll

    def get_t_statistcs(self):

        labels = self.output_labels
        logits = self.discriminative_free_energy()
        # s is to scale the learning rate to the number of output nodes
        s = len(logits)
        for i, (logit, label) in enumerate(zip(logits, labels)):
            p_y_given_x = T.nnet.softmax(-logit)
            cost.append(self.get_loglikelihood(p_y_given_x, label))
            # calculate the gradients
            params = self.hbias + self.V_params_flat + self.vbias_flat \
                + self.U_params_flat[i] + self.B_params_flat[i] \
                + self.cbias_flat[i]
            hessians = T.hessian(cost=cost[i], wrt=params)
            cr_bound = T.diag(-1/hessians)
            sigma = T.sqrt(cr_bound)
            t_stat = params / sigma
        return hessians

    def build_fn(self, train_set_x, train_set_y, train_set_labels):
        """
        build_fn func

        Parameters
        ----------
        """
        print('building theano computational graphs')
        index = T.lscalar()
        tensor_inputs = self.input + self.output
        data_inputs = train_set_x + train_set_y
        d_inputs = self.input
        d_data_inputs = train_set_x
        d_labels = self.label
        d_out_labels = train_set_labels

        lr = self.hyperparameters['learning rate']
        k = self.hyperparameters['gibbs steps']
        batch_size = self.hyperparameters['batch size']

        cost, gibbs_updates = self.get_generative_cost_updates(k, lr)
        d_cost, d_updates = self.get_discriminative_cost_updates(lr)
        preds = self.predict()
        # flatten list of preds in case of out variables > 1
        preds = list(chain.from_iterable(preds))
        hessians = self.get_t_statistcs()

        self.generate = theano.function(
            inputs=[index],
            outputs=cost,
            updates=gibbs_updates,
            givens={
                x: visible[index*batch_size:(index+1)*batch_size]
                for x, visible in zip(tensor_inputs, data_inputs)
            },
            name='generate',
            allow_input_downcast=True,
            on_unused_input='warn'
        )

        self.discriminate = theano.function(
            inputs=[index],
            outputs=d_cost,
            updates=d_updates,
            givens={
                tsr: data[index*batch_size:(index+1)*batch_size]
                for tsr, data in zip(d_inputs + d_labels,
                                     d_data_inputs + d_out_labels)
            },
            name='discriminate',
            allow_input_downcast=True,
            on_unused_input='warn'
        )

        self.predict = theano.function(
            inputs=[index],
            outputs=preds,
            updates=None,
            givens={
                tsr: data[index*batch_size:(index+1)*batch_size]
                for tsr, data in zip(d_inputs + d_labels,
                                     d_data_inputs + d_out_labels)
            },
            name='predict',
            allow_input_downcast=True,
            on_unused_input='warn'
        )

        self.statistics = theano.function(
            inputs=[index],
            outputs=hessians,
            updates=None,
            givens={
                tsr: data[:index*batch_size]
                for tsr, data in zip(d_inputs + d_labels,
                                     d_data_inputs + d_out_labels)
            }
        )


def main(rbm):
    rbm.build_fn()
    pass

if __name__ == '__main__':
    main(RBM(*net))
