import theano
import numpy

import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool


def relu(x):
    return T.switch(x<0, 0, x)

class LogisticRegression(object):
    def __init__(self, input, n_input, n_output):
        self.W = theano.shared(
            value=numpy.zeros(
                (n_input, n_output),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )

        self.b = theano.shared(
            value=numpy.zeros(
                (n_output, ),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]
        self.input = input

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class HiddenLayer(object):
    def __init__(self, rng, input, n_input, n_output, W = None, b = None, activation = T.tanh):
        self.input = input
        if W is None:
            W_value = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_input + n_output)),
                    high=numpy.sqrt(6. / (n_input + n_output)),
                    size=(n_input, n_output)
                ),
                dtype=theano.config.floatX
            )

            if activation == T.nnet.sigmoid:
                W_value = 4 * W_value

            W = theano.shared(value=W_value, name='W', borrow=True)

        if b is None:
            b = theano.shared(
                numpy.zeros(
                    (n_output, ),
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )
        self.b = b
        self.W = W

        lin_output = T.dot(input, self.W) + self.b

        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        self.params = [self.W, self.b]

class MLP(object):
    def __init__(self, rng, input, n_input, n_hidden, n_output):

        self.hiddenlayer = HiddenLayer(
            rng=rng,
            input=input,
            n_input=n_input,
            n_output=n_hidden,
            activation=T.tanh
        )

        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenlayer.output,
            n_input=n_hidden,
            n_output=n_output
        )

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            abs(self.hiddenLayer.W).sum() +
            abs(self.logRegressionLayer.W).sum()
        )
        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            abs(self.hiddenLayer.W ** 2).sum() +
            abs(self.logRegressionLayer.W ** 2).sum()
        )

        self.negtive_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )

        self.error = self.logRegressionLayer.errors
        self.params = self.hiddenlayer.params + self.logRegressionLayer.params

        self.input = self.input


class LeNetConvLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape, W=None, b=None, activation=T.tanh):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        fan_in = numpy.prod(filter_shape[1:])
        fan_out = filter_shape[0] * numpy.prod(filter_shape[2:])

        if W is None:
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            W = theano.shared(
                numpy.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX),
                borrow=True)
        if b is None:
            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, borrow=True)
        self.W = W
        self.b = b

        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape)
        conv_out = (conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output = conv_out if activation is None else activation(conv_out)
        self.params = [self.W, self.b]


class PoolLayer(object):
    def __init__(self, input, poolsize=(2, 2)):
        pooled_out = pool.pool_2d(
            input=input,
            ds=poolsize,
            ignore_border=True)
        self.output = pooled_out

class LeNetConvPoolLayer(object):

    def __init__(self, rng, input, filter_shape, image_shape, poolsize = (2, 2), activation=T.tanh):
        assert image_shape[1] == filter_shape[1]
        self.input = input

        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) // numpy.prod(poolsize))

        W_bound = numpy.sqrt(6. / (fan_in + fan_out))

        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(
                    low=-W_bound,
                    high=W_bound,
                    size=filter_shape
                ), dtype=theano.config.floatX
            ), borrow=True
        )

        self.b = theano.shared(
            numpy.zeros(
                (filter_shape[0], ),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )
        conv_out = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')

        if not activation is None:
            conv_out = activation(conv_out)

        pool_out = pool.pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # self.output = T.tanh(pool_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.output = pool_out

        self.params = [self.W, self.b]

        self.input = input

