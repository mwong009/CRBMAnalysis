import theano
import pickle
import sys
import numpy as np
import theano.tensor as T

from collections import OrderedDict as odict

net = 'net1', {
    'n_hidden': (16,),
    'seed': 42,
}


class Network(object):
    def __init__(self, name, hyperparameters=odict()):

        self.path = name
        self.hyperparameters = hyperparameters

        seed = hyperparameters['seed']
        self.np_rng = np.random.RandomState(seed)
        self.theano_rng = T.shared_randomstreams.RandomStreams(seed)

        self.input = []
        self.output = []
        self.model_params = odict()
        self.hbias = []
        self.W_params = []
        self.vbias = []
        self.B_params = []
        self.cbias = []

        self.monitoring_curves = {
            'CD error': [],
            'log likelihood': []
        }

    def save_params(self):
        model_values = {}
        for param in self.model_params:
            model_values[param.name] = param.get_value()

        to_file = model_values, self.hyperparameters, self.monitoring_curves
        with open(self.path+'.params', 'wb') as f:
            pickle.dump(to_file, f, protocol=pickle.HIGHEST_PROTOCOL)

    def __load_params(self, hyper):
        if os.path.isfile(self.path+'.params'):
            with open(self.path, 'rb') as f:
                model_values, hyperparameters, curves = pickle.load(f)
            # update hyperparameters
            for key, value in hyper.items():
                hyperparameters[key] = value

        else:
            pass

        return pass


class RBM(Network):
    ''' define the RBM toplevel '''
    def __init__(self, name, hyperparameters=odict()):
        Network.__init__(name, hyperparameters)
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
        if name in self.model_params.keys():
            hbias = theano.shared(
                value=self.model_params[name],
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

    def add_node(self, name='visible'):
        """
        add_node func

        Parameters
        ----------
        name : string, optional
            Name of visible node e.g. `'age'`

        Updates
        -------
        self.input[] : sequence of `T.tensor3()`\n
        self.W_params[] : sequence of `theano.shared()`\n
        self.vbias[] : sequence of `theano.shared()`\n
        self.model_params['x_'+name] : OrderedDict of `theano.shared()`\n
        """
        shp_hidden = self.hyperparameters['n_hidden']
        shp_visible = self.hyperparameters[name]
        tsr_variable = T.tensor3(name)  # input tensor as (rows, items, cats)

        # Create the tensor shared variables as (items, cats, hiddens)
        if 'W_'+name in self.model_params.keys():
            W = theano.shared(
                value=self.model_params['W_'+name],
                name='W_'+name,
                borrow=True
            )
        else:
            W = theano.shared(
                value=np.random.uniform(
                    low=-np.sqrt(6./np.sum(shp_visible+shp_hidden)),
                    high=np.sqrt(6./np.sum(shp_visible+shp_hidden)),
                    size=shp_visible+shp_hidden
                ),
                name='W_'+name,
                borrow=True
            )
        if 'vbias_'+name in self.model_params.keys():
            vbias = theano.shared(
                value=self.model_params['vbias_'+name],
                name='vbias_'+name,
                borrow=True
            )
        else:
            vbias = theano.shared(
                value=np.zeros(
                    shape=shp_visible,
                    dtype=theano.config.floatX
                ),
                name='vbias_'+name,
                borrow=True
            )

        self.input.append(tsr_variable)
        self.W_params.append(W)
        self.vbias.append(vbias)
        self.model_params['W_'+name] = W
        self.model_params['vbias_'+name] = vbias

    def add_connection_to(self, name='output'):
        """
        add_connection_to func

        Parameters
        ----------
        name : string, optional
            Name of visible node e.g. `'mode_prime'`

        Updates
        -------
        self.output[] : sequence of `T.matrix()`\n
        self.W_params[] : sequence of `theano.shared()`\n
        self.cbias[] : sequence of `theano.shared()`\n
        self.B_params[] : sequence of `theano.shared()`\n
        self.model_params[] : sequence of `theano.shared()`\n
        """
        shp_hidden = self.hyperparameters['n_hidden']
        shp_output = self.hyperparameters[name]
        tsr_variable = T.matrix(name)  # output tensor as (rows, outs)

        # Create the tensor shared variables as (outs, hiddens)
        if 'W_'+name in self.model_params.keys():
            W = theano.shared(
                value=self.model_params['W_'+name],
                name='W_'+name,
                borrow=True
            )
        else:
            W = theano.shared(
                value=np.random.uniform(
                    low=-np.sqrt(6./np.sum(shp_output+shp_hidden)),
                    high=np.sqrt(6./np.sum(shp_output+shp_hidden)),
                    size=shp_output+shp_hidden
                ),
                name='W_'+name,
                borrow=True
            )
        if 'cbias_'+name in self.model_params.keys():
            cbias = theano.shared(
                value=self.model_params['cbias_'+name],
                name='cbias_'+name,
                borrow=True
            )
        else:
            cbias = theano.shared(
                value=np.zeros(
                    shape=shp_output,
                    dtype=theano.config.floatX
                ),
                name='cbias_'+name,
                borrow=True
            )

        self.output.append(tsr_variable)
        self.W_params.append(W)
        self.cbias.append(cbias)
        self.model_params['W_'+name] = W
        self.model_params['cbias_'+name] = cbias

        # condtional RBM connection (B weights)
        for node in self.input:
            name = node.name
            shp_visible = self.hyperparameters[name]

            # Create the tensor shared variables as (items, cats, outs)
            if 'B_'+name in self.model_params.keys():
                B = theano.shared(
                    value=self.model_params['B_'+name],
                    name='B_'+name,
                    borrow=True
                )
            else:
                B = theano.shared(
                    value=np.random.uniform(
                        low=-np.sqrt(6/np.sum(shp_visible+shp_output)),
                        high=np.sqrt(6/np.sum(shp_visible+shp_output)),
                        size=shp_visible+shp_output
                    ),
                    name='B_'+name,
                    borrow=True
                )

            self.B_params.append(B)
            self.model_params['B_'+name] = B

    def free_energy(self):
        """
        free_energy func

        Parameters
        ----------
        self : RBM class object

        Returns
        -------
        F(y,x) :
            Scalar value of the generative model free energy

        Notes
        -----
        The free energy for the generative model is computed as:

        :math:\n
        `F(y,x,h) = -(xWh + yWh + vbias*x + hbias*h + cbias*y)`\n
        `    wx_b = xW + yW + hbias`\n
        `  F(y,x) = -{vbias*x + cbias*y + sum_k[ln(1+exp(wx_b))]}`

        """
        visibles = self.input + self.output
        hbias = self.hbias[0]
        vbiases = self.vbias + self.cbias
        W_params = self.W_params

        # input shapes as (rows, items, cats)
        # output shapes as (rows, outs)
        # weight shapes as (items, cats, hiddens), (outs, hiddens)
        wx_b = hbias  # (hiddens,) broadcast(T,F) --> (rows, hiddens)
        utility = 0  # (rows,)
        for visible, W_param, vbias in zip(visibles, W_params, vbiases):
            utility += T.tensordot(visible, vbias, axes=[[1, 2], [0, 1]])
            if W_param.ndim == 2:
                wx_b += T.dot(visible, W_param)
            else:
                wx_b += T.tensordot(visible, W_param, axes=[[1, 2], [0, 1]])

        # utility --> (rows,)
        # ...axis=1) sums over hidden axis --> (rows,)
        entropy = T.sum(T.log(1+T.exp(wx_b)), axis=1)
        return - (utility + entropy)

    def discriminative_free_energy(self):
        """
        discriminative_free_energy func

        Parameters
        ----------
        self : RBM class object

        Returns
        -------
        F(y|x,h) :
            A `list[]` of vectors of the discriminative model free energy
            for each output node. Negative loglikelihood can be used as the
            objective function.

        Notes
        -----
        The free energy for the discriminative model is computed as:

        :math:\n
        `F(y,x,h) = -(xWh + yWh + vbias*x + hbias*h + cbias*y)`\n
        `    wx_b = xW + W_j + hbias`\n
        `F(y|x,h) = -{cbias + Bx + sum_k[ln(1+(exp(wx_b))]}`

        :params: used are W^1, W^2, B, c, h, v biases

        """
        visibles = self.input
        hbias = self.hbias[0]
        vbiases = self.vbias
        cbiases = self.cbias
        xWh_params = self.W_params[:len(visibles)]
        hWy_params = self.W_params[len(visibles):]
        B_params = self.B_params

        # (hiddens,) broadcast(T, F, T) --> ('x', hiddens, 'x')
        wx_b = hbias.dimshuffle('x', 0, 'x')
        utility = []

        # loop over all input nodes
        for visible, W_param, B_param in zip(visibles, xWh_params, B_params):

            # loop over all output nodes
            for i, (hWy_param, cbias) in enumurate(zip(hWy_params, cbiases)):
                utility.append(cbias)  # (outs,) --> ('x', outs)
                wx_b += hWy_param.dimshuffle('x', 1, 0)
                if B_param.ndim == 2:
                    utility[i] += T.dot(visible, B_param)  # (rows, outs)
                else:
                    utility[i] += T.tensordot(visible, B_param,
                                              axes=[[1, 2], [0, 1]])

            if W_param.ndim == 2:
                wx = T.dot(visible, W_param)
                wx_b += wx.dimshuffle(0, 1, 'x')  # (rows, hiddens, 'x')
            else:
                wx = T.tensordot(visible, W_param, axes=[[1, 2], [0, 1]])
                wx_b += wx.dimshuffle(0, 1, 'x')  # (rows, hiddens, 'x')

        # sum over hiddens axis (rows, hiddens, 'x') --> (rows, 'x')
        entropy = T.sum(T.log(1+T.exp(wx_b)), axis=1)

        # add entropy to each expected utility term
        energy = []
        for u in utility:
            energy.append(- (u + entropy))

        return energy


def main(rbm):
    pass

if __name__ == '__main__':
    main(RBM(*net))
