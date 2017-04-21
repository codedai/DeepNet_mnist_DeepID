import _pickle as cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,                             #卷积函数，用W卷积不加偏置
                filter_shape=filter_shape, image_shape=image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,                             #pooling，用max不用mean，不重叠
                                            ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))          #卷积层池化后加上偏置用tanh输出，dimshuffle()将向量整形为矩阵，具体不懂

        # store parameters of this layer
        self.params = [self.W, self.b]                                                  #卷积核+偏置并为参数

  #学习率=0.1， 学习次数=200， nkerns=[20,50]表示第一层20个核，第二层50个核； 补丁大小：500？？？？
def evaluate_lenet5(learning_rate=0.1, n_epochs=200,
                    dataset='../data/mnist.pkl.gz',
                    nkerns=[20, 50], batch_size=500):
    """ Demonstrates lenet on MNIST datasets

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)                                               #随机数做种

    datasets = load_data(dataset)                                                       #读入数据

    train_set_x, train_set_y = datasets[0]                                              #传递三部分数据（解包）
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing                #表示数据可以借用提高GPU运算速率，shape[0],作用为止
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size                                                       #样本总数量
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch                                       #当前batch的下标
    x = T.matrix('x')   # the data is presented as rasterized images                    #当前batch
    y = T.ivector('y')  # the labels are presented as 1D vector of                      #当前batch的标签
                        # [int] labels

    ishape = (28, 28)  # this is the size of MNIST images

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print ('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size,28*28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 1, 28, 28))                                   #input是reshape的x

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
    # maxpooling reduces this further to (24/2,24/2) = (12,12)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
    #初始化第一个卷积池化layer，input = layer0_input
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
            image_shape=(batch_size, 1, 28, 28),
            filter_shape=(nkerns[0], 1, 5, 5), poolsize=(2, 2))

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
    # maxpooling reduces this further to (8/2,8/2) = (4,4)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
    #初始化第二个卷积池化layer , input = layer0_output
    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
            image_shape=(batch_size, nkerns[0], 12, 12),
            filter_shape=(nkerns[1], nkerns[0], 5, 5), poolsize=(2, 2))

    # the TanhLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,32*4*4) = (20,512)
    #layer2是第一层全连接层，拉平后的池化层作为输入
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    # 用隐藏层的类表示
    layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[1] * 4 * 4,
                         n_out=500, activation=T.tanh)

    # classify the values of the fully-connected sigmoidal layer
    # 输出是逻辑回归层
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)

    # the cost we minimize during training is the NLL of the model
    # 代价函数值用negative_log_likelihood来算，（自带的？）
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    # 定义一个函数，计算输出层的误差，用givens来覆盖全局变量
    test_model = theano.function([index], layer3.errors(y),
             givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    ## 同上定义一个函数，计算输出层的误差，用givens来覆盖全局变量
    validate_model = theano.function([index], layer3.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]})

    # create a list of all model parameters to be fit by gradient descent
    # 各层参数合并
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    # 利用自带的函数计算各参数的偏导
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i],grads[i]) pairs.
    # 更新参数十分麻烦， 创建一个叫做updates的list来自动更新（？为什么要用for，这样不会很慢吗？——坟蛋这不是matlab！）
    updates = []
    for param_i, grad_i in zip(params, grads):
        updates.append((param_i, param_i - learning_rate * grad_i))

    # 定义训练函数，输出cost并用update 的方法更新参数
    train_model = theano.function([index], cost, updates=updates,
          givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is  如果训练误差良好的话训练的次数变为两倍
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is 如果误差小于上一次误差的0.995，patience increase
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)  #评价训练效果的频率，这个数值为什么这么取我不清楚
                                  # go through this manually
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):                        #总体样本训练次数
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):                     #逐个样本训练

            iter = (epoch - 1) * n_train_batches + minibatch_index          #到目前为止总的训练次数

            if iter % 100 == 0:                                             #每训练100次输出一个提示，提示训练次数
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)                          #训练一次

            if (iter + 1) % validation_frequency == 0:                      #到达需要进行一次评价的次数，对学习结果进行评价

                # compute zero-one loss on validation set                   #利用for循环和validation_modle(index)返回所有评价样本的误差值并构造一个表
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)        #当前误差值=当前平均
                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                      (epoch, minibatch_index + 1, n_train_batches, \
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:             #如果当 前平均误差<(最好误差*阀值)，证明参数还有很大的优化空间，加倍训练次数

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]  #用测试样本对模型参数进行评价
                    test_score = numpy.mean(test_losses)                           #这里有个tip：应为参数使用train集合训练使用validation集合进行评价；
                    print(('     epoch %i, minibatch %i/%i, test error of best '   #所以参数的拟合是会偏向那两个集合的特征的，所以要是用全新的集合来得到参数的客观表现
                           'model %f %%') %                                        #在各种训练中，样本都要分为训练样本、评价(拟合)样本和测试样本进行使用，比例大概是6:2:2，这里是 5:1:1
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:                                               #如果没耐性了（到达最大训练次数），就停止训练
                done_looping = True
                break
    #下面就是计时啊评价啊什么什么的
    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i,'\
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)