import timeit
import sys

from convNet.loadData import *
from convNet.layer import *

import numpy
import theano
import theano.tensor as T


# def deepID_test(learning_rate=0.12, n_epochs=200, dataset='mnist.pkl.gz', nkerns=[20, 40, 60, 80], batch_size=500,
#                 acti_func=relu):
# def deepID_test(learning_rate=0.12, n_epochs=200, dataset='mnist.pkl.gz', nkerns=[10, 20, 30, 40], batch_size=500,
#                 acti_func=relu):
# def deepID_test(learning_rate=0.115, n_epochs=200, dataset='mnist.pkl.gz', nkerns=[30, 45, 60, 85], batch_size=500,
#                 acti_func=relu):
# def deepID_test(learning_rate=0.11, n_epochs=200, dataset='mnist.pkl.gz', nkerns=[10, 15, 20, 29], batch_size=500,
#                 acti_func=relu):
def deepID_test(learning_rate=0.11, n_epochs=200, dataset='mnist.pkl.gz', nkerns=[5, 10, 12, 20], batch_size=500,
                acti_func=relu):
    src_channel = 1
    layer1_image_shape = (500, src_channel, 28, 28)
    layer1_filter_shape = (nkerns[0], src_channel, 5, 5)
    layer2_image_shape = (500, nkerns[0], 12, 12)
    layer2_filter_shape = (nkerns[1], nkerns[0], 3, 3)
    layer3_image_shape = (500, nkerns[1], 5, 5)
    layer3_filter_shape = (nkerns[2], nkerns[1], 2, 2)
    layer4_image_shape = (500, nkerns[2], 2, 2)
    layer4_filter_shape = (nkerns[3], nkerns[2], 2, 2)
    result_image_shape = (500, nkerns[3], 1, 1)

    # layer1_image_shape = (500, src_channel, 28, 28)
    # layer1_filter_shape = (nkerns[0], src_channel, 3, 3)
    # layer2_image_shape = (500, nkerns[0], 13, 13)
    # layer2_filter_shape = (nkerns[1], nkerns[0], 4, 4)
    # layer3_image_shape = (500, nkerns[1], 5, 5)
    # layer3_filter_shape = (nkerns[2], nkerns[1], 2, 2)
    # layer4_image_shape = (500, nkerns[2], 2, 2)
    # layer4_filter_shape = (nkerns[3], nkerns[2], 2, 2)
    # result_image_shape = (500, nkerns[3], 1, 1)

    rng = numpy.random.RandomState(23455)

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
    # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    layer1_input = x.reshape(layer1_image_shape)

    layer1 = LeNetConvPoolLayer(rng=rng,
                                input=layer1_input,
                                image_shape=layer1_image_shape,
                                filter_shape=layer1_filter_shape,
                                poolsize=(2, 2),
                                activation=acti_func)

    layer2 = LeNetConvPoolLayer(rng,
                                input=layer1.output,
                                image_shape=layer2_image_shape,
                                filter_shape=layer2_filter_shape,
                                poolsize=(2, 2),
                                activation=acti_func)

    layer3 = LeNetConvPoolLayer(rng,
                                input=layer2.output,
                                filter_shape=layer3_filter_shape,
                                image_shape=layer3_image_shape,
                                poolsize=(2, 2),
                                activation=acti_func
                                )

    layer4 = LeNetConvLayer(rng,
                            input=layer3.output,
                            image_shape=layer4_image_shape,
                            filter_shape=layer4_filter_shape,
                            activation=acti_func)

    # deepid_input = layer4.output.flatten(2)

    layer3_output_flatten = layer3.output.flatten(2)
    layer4_output_flatten = layer4.output.flatten(2)
    deepid_input = T.concatenate([layer3_output_flatten, layer4_output_flatten], axis=1)

    deepid_layer = HiddenLayer(rng,
                               input=deepid_input,
                               n_input=numpy.prod(result_image_shape[1:]) + numpy.prod(
                                   layer4_image_shape[1:]),
                               # n_in  = numpy.prod( result_image_shape[1:] ),
                               # n_output=160,
                               n_output=21,
                               activation=acti_func)
    softmax_layer = LogisticRegression(
        input=deepid_layer.output,
        # n_input=160,
        # n_input=80,
        n_input=21,
        # n_output=1595)
        n_output=30)
        # n_output=797)

    cost = softmax_layer.negative_log_likelihood(y)
# create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        softmax_layer.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        softmax_layer.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    # params = layer3.params + layer2.params + layer1.params + layer0.params
    params = softmax_layer.params + deepid_layer.params + layer4.params + layer3.params + layer2.params + layer1.params
    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

def sgd_optimization_mnist(learning_rate = 0.13, n_epochs = 1000,
                           dataset = 'mnist.pkl.gz',
                           batch_size = 600):
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    print('...building the model')

    index = T.iscalar()

    x = T.matrix('x')
    y = T.ivector('y')

    classifier = LogisticRegression(x, 28 * 28, 10)
    cost = classifier.negative_log_likelihood(y)

    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # Compute the gradient of cost with respect to theta = (W, b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # update the parameters of the models
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    print('...training the model...')

    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    # save the best model
                    with open('best_model.pkl', 'wb') as f:
                        pickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)


if __name__ == '__main__':
    deepID_test()
    # sgd_optimization_mnist(learning_rate=0.2, n_epochs=1000, dataset='mnist.pkl.gz', batch_size=600)
    # sgd_optimization_mnist()