import  tensorflow as tf
import math
import convNet_tensorflow.tensorflowvisu as tensorflowvisu
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

tf.set_random_seed(0)

mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32, [None, 10])
lr = tf.placeholder(tf.float32)
pkeep = tf.placeholder(tf.float32)

L = 6
M = 12
N = 24
O = 200

W1 = tf.Variable(tf.truncated_normal([6, 6, 1, L], stddev=0.1))
B1 = tf.Variable(tf.ones([L])/10)
W2 = tf.Variable(tf.truncated_normal([5, 5, L, M], stddev=0.1))
B2 = tf.Variable(tf.ones([M])/10)
W3 = tf.Variable(tf.truncated_normal([4, 4, M, N], stddev=0.1))
B3 = tf.Variable(tf.ones([N])/10)

W4 = tf.Variable(tf.truncated_normal([7 * 7 * N, O], stddev=0.1))
B4 = tf.Variable(tf.ones([O])/10)
W5 = tf.Variable(tf.truncated_normal([O, 10], stddev=0.1))
B5 = tf.Variable(tf.zeros([10])/10)

# the model
# XX = tf.reshape(X, [-1, 784])

stride = 1
Y1cnv = tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME')
Y1 = tf.nn.relu(Y1cnv + B1)
stride = 2
Y2cnv = tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME')
Y2 = tf.nn.relu(Y2cnv + B2)
stride = 2
Y3cnv = tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME')
Y3 = tf.nn.relu(Y3cnv + B3)

YY = tf.reshape(Y3, shape=[-1, 7 * 7 * N])

Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
Y4_drop = tf.nn.dropout(Y4, pkeep)
Ylogits = tf.matmul(Y4_drop, W5) + B5
Y = tf.nn.softmax(Ylogits)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

allweights = tf.concat([tf.reshape(W1, [-1]),
                        tf.reshape(W2, [-1]),
                        tf.reshape(W3, [-1]),
                        tf.reshape(W4, [-1]),
                        tf.reshape(W5, [-1])], 0)

allbiases = tf.concat([tf.reshape(B1, [-1]),
                      tf.reshape(B2, [-1]),
                      tf.reshape(B3, [-1]),
                      tf.reshape(B4, [-1]),
                      tf.reshape(B5, [-1])], 0)

# learning rate decaying
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

I = tensorflowvisu.tf_format_mnist_images(X, Y, Y_)  # assembles 10x10 images by default
It = tensorflowvisu.tf_format_mnist_images(X, Y, Y_, 1000, lines=25)  # 1000 images on 25 lines
datavis = tensorflowvisu.MnistDataVis()

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

def training_step(i, update_test_data, update_train_data):

    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)

    # learning rate decay
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    dacay_speed = 2000
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/dacay_speed)

    # compute training values for visualisation
    if update_train_data:
        a, c, im, w, b = sess.run([accuracy, cross_entropy, I, allweights, allbiases], {X: batch_X, Y_: batch_Y, pkeep:1.0})
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")
        datavis.append_training_curves_data(i, a, c)
        datavis.update_image1(im)
        datavis.append_data_histograms(i, w, b)

    # compute test values for visualisation
    if update_test_data:
        a, c, im = sess.run([accuracy, cross_entropy, It], {X: mnist.test.images, Y_: mnist.test.labels, pkeep:1.0})
        print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))
        datavis.append_test_curves_data(i, a, c)
        datavis.update_image2(im)

    # the backpropagation training step
    sess.run(train_step, {X: batch_X, Y_: batch_Y, pkeep: 0.75, lr: learning_rate})

datavis.animate(training_step, iterations=13000+1, train_data_update_freq=20, test_data_update_freq=100, more_tests_at_start=True)

# to save the animation as a movie, add save_movie=True as an argument to datavis.animate
# to disable the visualisation use the following line instead of the datavis.animate line
# for i in range(10000+1): training_step(i, i % 100 == 0, i % 20 == 0)

print("max test accuracy: " + str(datavis.get_max_test_accuracy()))

# 9100: ********* epoch 16 ********* test accuracy:0.9918 test loss: 3.62353
# 9120: accuracy:1.0 loss: 0.00480763 (lr:0.00013033997093593773)
# 9140: accuracy:1.0 loss: 0.0282832 (lr:0.00013003808318107972)
# 9160: accuracy:1.0 loss: 0.0463847 (lr:0.00012973919925957168)
# 9180: accuracy:1.0 loss: 0.00143553 (lr:0.00012944328928277232)
# 9200: accuracy:1.0 loss: 0.0213146 (lr:0.0001291503236594374)
# 9200: ********* epoch 16 ********* test accuracy:0.9918 test loss: 3.58907
# 9220: accuracy:1.0 loss: 0.00298557 (lr:0.00012886027309276044)
# 9240: accuracy:1.0 loss: 0.0317459 (lr:0.00012857310857744306)
# 9260: accuracy:1.0 loss: 0.0321407 (lr:0.00012828880139679441)
# 9280: accuracy:1.0 loss: 0.0299012 (lr:0.00012800732311985956)
# 9300: accuracy:1.0 loss: 0.00156396 (lr:0.00012772864559857617)

# 12900: ********* epoch 22 ********* test accuracy:0.9909 test loss: 3.78907
# 12920: accuracy:1.0 loss: 0.00320282 (lr:0.00010453790756014309)
# 12940: accuracy:1.0 loss: 0.0509803 (lr:0.00010449275462548875)
# 12960: accuracy:1.0 loss: 0.00545407 (lr:0.00010444805097004095)
# 12980: accuracy:0.99 loss: 1.11253 (lr:0.00010440379212339686)
# 13001: accuracy:1.0 loss: 0.025557 (lr:0.00010435779421771104)
# 13001: ********* epoch 22 ********* test accuracy:0.9912 test loss: 3.73501