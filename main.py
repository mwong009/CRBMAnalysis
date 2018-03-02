import theano
import numpy as np
import theano.tensor as T


class RBM(object):
    ''' define the RBM toplevel '''
    def __init__(self, seed, n_hidden):
        self.seed = seed
        self.input = []
        self.output = []
        self.shp_visible = []
        self.shp_output = []
        self.model_params = []
        self.hbias = []
        self.W_params = []
        self.vbias = []
        self.B_params = []
        self.cbias = []

        self.add_hbias((n_hidden,))

    def add_hbias(self, shp_hidden, name='hbias'):
        """
        add_hbias func

        Parameters
        ----------
        shp_hidden : sequence of `int`
            Shape of hidden unit inputs e.g. `(2,)`.

        name: `str`, optional
            Name of hidden node e.g. `'hbias'`

        Updates
        -------
        self.hbias[] : sequence of `theano.shared()`

        self.shp_hidden : sequence of `int`

        self.model_params[] : sequence of `theano.shared()`

        """
        if len(self.hbias) == 0:
            hbias = theano.shared(
                value=np.zeros(
                    shape=shp_hidden,
                    dtype=theano.config.floatX
                ),
                name='hbias',
                borrow=True
            )

            self.hbias.append(hbias)
            self.shp_hidden = shp_hidden
            self.model_params.append(hbias)

    def add_node(self, shp_visible, name='visible'):
        """
        add_node func

        Parameters
        ----------
        shp_visible : sequence of `int`
            Shape of visible unit inputs e.g. `(2, 1)` or `(2,)`.
            Default use `(items, cats)` or `(1, cats)` or `(items, 1)`

        name : string, optional
            Name of visible node e.g. `'age'`

        Updates
        -------
        self.input[] : sequence of `T.tensor3()`

        self.shp_visible[] : sequence of `dict{}`

        self.W_params[] : sequence of `theano.shared()`

        self.vbias[] : sequence of `theano.shared()`

        self.model_params[] : sequence of `theano.shared()`

        """
        shp_hidden = self.shp_hidden
        tsr_variable = T.tensor3(name)  # input tensor as (rows, items, cats)

        # Create the tensor shared variables as (items, cats, hiddens)
        W = theano.shared(
            value=np.random.uniform(
                low=-np.sqrt(6/np.sum(shp_visible+shp_hidden)),
                high=np.sqrt(6/np.sum(shp_visible+shp_hidden)),
                size=shp_visible+shp_hidden
            ),
            name=name,
            borrow=True
        )
        vbias = theano.shared(
            value=np.zeros(
                shape=shp_visible,
                dtype=theano.config.floatX
            ),
            name=name,
            borrow=True
        )

        self.input.append(tsr_variable)
        self.shp_visible.append({'name': name, 'shape': shp_visible})
        self.W_params.append(W)
        self.vbias.append(vbias)
        self.model_params.extend([W, vbias])

    def add_connection_to(self, shp_output, name='output'):
        """
        add_connection_to func

        Parameters
        ----------
        shp_output : sequence of `int`
            Shape of visible unit outputs e.g. `(2,)`. Default use `(cats,)`

        name : string, optional
            Name of visible node e.g. `'mode_prime'`

        Updates
        -------
        self.output[] : sequence of `T.matrix()`

        self.shp_output[] : sequence of `dict{}`

        self.W_params[] : sequence of `theano.shared()`

        self.cbias[] : sequence of `theano.shared()`

        self.B_params[] : sequence of `theano.shared()`

        self.model_params[] : sequence of `theano.shared()`

        """
        shp_hidden = self.shp_hidden
        tsr_variable = T.matrix(name)  # output tensor as (rows, outs)

        # Create the tensor shared variables as (outs, hiddens)
        W = theano.shared(
            value=np.random.uniform(
                low=-np.sqrt(6/np.sum(shp_output+shp_hidden)),
                high=np.sqrt(6/np.sum(shp_output+shp_hidden)),
                size=shp_output+shp_hidden
            ),
            name=name,
            borrow=True
        )
        cbias = theano.shared(
            value=np.zeros(
                shape=shp_output,
                dtype=theano.config.floatX
            ),
            name=name,
            borrow=True
        )

        self.output.append(tsr_variable)
        self.shp_output.append({'name': name, 'shape': shp_output})
        self.W_params.append(W)
        self.cbias.append(cbias)
        self.model_params.extend([W, cbias])

        # condtional RBM connection (B weights)
        for dict_item in self.shp_visible:
            name = dict_item['name']
            shp_visible = dict_item['shape']

            # Create the tensor shared variables as (items, cats, outs)
            B = theano.shared(
                value=np.random.uniform(
                    low=-np.sqrt(6/np.sum(shp_visible+shp_output)),
                    high=np.sqrt(6/np.sum(shp_visible+shp_output)),
                    size=shp_visible+shp_output
                ),
                name=name,
                borrow=True
            )

            self.B_params.append(B)
            self.model_params.append(B)

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
        hbias = self.hbias[0]
        vbiases = self.vbias + self.cbias
        visibles = self.input + self.output
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


def main():
    rbm = RBM(111, 16)


if __name__ == '__main__':
    main()
