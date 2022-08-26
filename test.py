# -*- coding: UTF-8 -*-
# softmax plus log-likelihood cost is more common in modern image classification networks.
from network3 import Network, ConvLayer, PoolLayer, FullyConnectedLayer, SoftmaxLayer
from network3 import ReLU
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import numpy as np
import network3


def testTheano():
    from theano import function, config, shared, sandbox
    import theano.tensor as T
    import numpy
    import time
    print("Testing Theano library...")
    vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
    iters = 1000

    rng = numpy.random.RandomState(22)
    x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
    f = function([], T.exp(x))
    print(f.maker.fgraph.toposort())
    t0 = time.time()
    for i in range(iters):
        r = f()
    t1 = time.time()
    print("Looping %d times took %f seconds" % (iters, t1 - t0))
    print("Result is %s" % (r,))
    if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
        print('Used the cpu')
    else:
        print('Used the gpu')
# Perform check:
#testTheano()


# ----------------------
# - network3.py example:

#for test
train_accuracylist = []
test_accuracylist = []
cost_list = []
epoch_index = 10
epoch_indexs = np.arange(0, epoch_index, 1, dtype=int)

# read data:
training_data, validation_data, test_data = network3.load_data_shared()
# mini-batch size:
mini_batch_size = 10
# 8 X 8
#number of nuerons in Conv1
CHNLIn = 1
CHNL1 = 2  # change to 2 to keep columns in constraints
CHNL2 = 8
CHNL3 = 8
CHNL4 = 8
CHNL5 = 8

pool_scale = 2
conv_scale = 3
image_scale = 32
# image width, image height, filter maps number, input maps number
i_f_map = ((image_scale, image_scale, CHNL1, CHNLIn),  # same padding
           (32, 32, CHNL1, CHNL1),  # Pool
           (16, 16, CHNL2, CHNL1),  # valid
           (14, 14, CHNL2, CHNL2),  # Pool
           (7, 7, CHNL5, CHNL4),  # MLP1 8*9*8????
           (11, 10)  # 7 * 7 * 6
           )

#Conv1
image_shape1 = (mini_batch_size, CHNLIn,
                i_f_map[0][0], i_f_map[0][1])  # C = 28*4/4 = 28
filter_shape1 = (i_f_map[0][2], i_f_map[0][3],
                 conv_scale, conv_scale)  # 4,4,1,3,3

#Pool1
image_shape2 = (mini_batch_size, i_f_map[0][2], i_f_map[1][0], i_f_map[1][1])
filter_shape2 = (i_f_map[1][2], i_f_map[1][3], conv_scale, conv_scale)


#Conv2
image_shape3 = (mini_batch_size, i_f_map[1][2], i_f_map[2][0], i_f_map[2][1])
filter_shape3 = (i_f_map[2][2], i_f_map[2][3], conv_scale, conv_scale)

#Pool2
image_shape4 = (mini_batch_size, i_f_map[2][2], i_f_map[3][0], i_f_map[3][1])
filter_shape4 = (i_f_map[3][2], i_f_map[3][3], conv_scale, conv_scale)

# #Conv3
# image_shape5 = (mini_batch_size, i_f_map[3][2], i_f_map[4][0], i_f_map[4][1])
# filter_shape5 = (i_f_map[4][2], i_f_map[4][3], conv_scale, conv_scale)

# #Pool3
# image_shape6 = (mini_batch_size, i_f_map[4][2], i_f_map[5][0], i_f_map[5][1])
# filter_shape6 = (i_f_map[5][2], i_f_map[5][3], conv_scale, conv_scale)

# image_shape7 = (mini_batch_size, i_f_map[5][2], i_f_map[6][0], i_f_map[6][1])
# filter_shape7 = (i_f_map[6][2], i_f_map[6][3], conv_scale, conv_scale)


def create_and_test():

    Conv1 = ConvLayer(image_shape=image_shape1,
                      filter_shape=filter_shape1,
                      poolsize=(pool_scale, pool_scale),
                      activation_fn=ReLU, border_mode='half')

    Pool1 = PoolLayer(image_shape=image_shape2,
                      filter_shape=filter_shape2,
                      poolsize=(pool_scale, pool_scale),
                      activation_fn=ReLU, _params=Conv1.params)

    Conv2 = ConvLayer(image_shape=image_shape3,
                      filter_shape=filter_shape3,
                      poolsize=(pool_scale, pool_scale),
                      activation_fn=ReLU, border_mode='valid')

    Pool2 = PoolLayer(image_shape=image_shape4,
                      filter_shape=filter_shape4,
                      poolsize=(pool_scale, pool_scale),
                      activation_fn=ReLU, _params=Conv2.params)

    # Conv3 = ConvLayer(image_shape=image_shape5,
    #                   filter_shape=filter_shape5,
    #                   poolsize=(pool_scale, pool_scale),
    #                   activation_fn=ReLU, border_mode='half')

    # Pool3 = PoolLayer(image_shape=image_shape6,
    #                   filter_shape=filter_shape6,
    #                   poolsize=(pool_scale, pool_scale),
    #                   activation_fn=ReLU, _params=Conv3.params)

    MLP1 = FullyConnectedLayer(n_in=(i_f_map[4][2], i_f_map[4][0],
                                     i_f_map[4][1]), n_out=i_f_map[5][0]*1*1,
                               activation_fn=ReLU, p_dropout=0.5)

    SMLayer = SoftmaxLayer(n_in=i_f_map[5][0],
                           n_out=i_f_map[5][1])

    net = Network([
        Conv1,
        Pool1,
        Conv2,
        Pool2,
        # Conv3,
        # Pool3,
        MLP1,
        SMLayer
    ], mini_batch_size)

    accuracy_trained, accuracy_test, cost = net.SGD(training_data=training_data, epochs=10, mini_batch_size=mini_batch_size,
                                                    eta=0.03, validation_data=validation_data, test_data=test_data, lmbda=0.1)
    return accuracy_trained, accuracy_test, cost


def plot_n(indexlists, valuelists, labellist):
    if len(indexlists) == len(valuelists) == len(labellist):
        fig, (ax1, ax2, ax3) = plt.subplots(len(indexlists), 1)
        ax1.plot(indexlists[0], valuelists[0], "-.",
                 label=labellist[0])  # Plot the chart
        ax1.set_title(label=labellist[0])
        ax1.label_outer()
        ax2.plot(indexlists[1], valuelists[1], "-.",
                 label=labellist[1])  # Plot the chart
        ax2.set_title(label=labellist[1])
        ax2.label_outer()
        ax3.plot(indexlists[2], valuelists[2], "-.",
                 label=labellist[2])  # Plot the chart
        ax3.set_title(label=labellist[2])
        ax3.label_outer()
        plt.show()  # display


for i in range(epoch_index):
    accuracy_trained, accuracy_test, cost = create_and_test()
    train_accuracylist.append(accuracy_trained)
    test_accuracylist.append(accuracy_test)
    cost_list.append(cost)
    print("Now i = ", i)

print("FortestConvModel:")
print("train_accuracylist= ", ['{:.2%}'.format(i) for i in train_accuracylist])
print("test_accuracylist= ", ['{:.2%}'.format(i) for i in test_accuracylist])
print("cost_list= ", cost_list)

plot_n([epoch_indexs, epoch_indexs, epoch_indexs], [train_accuracylist,
       test_accuracylist, cost_list], ["trained accuracy", "test accuracy", "cost in training"])
