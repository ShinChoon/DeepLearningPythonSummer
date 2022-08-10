# -*- coding: UTF-8 -*-
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
import network3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from network3 import ReLU
from network3 import Network, ConvLayer, PoolLayer, FullyConnectedLayer, SoftmaxLayer # softmax plus log-likelihood cost is more common in modern image classification networks.

#for test
accuracy_list = []
test_accuracylist = []
cost_list = []
epoch_index = 10
epoch_indexs = np.arange(0, epoch_index, 1, dtype=int)

# read data:
training_data, validation_data, test_data = network3.load_data_shared()
# mini-batch size:
mini_batch_size = 10

#number of nuerons in Conv1
CHNLIn = 1
CHNL1 = 2
CHNL2 = 2
CHNL3 = 2 
CHNL4 = 2
CHNL5 = 2

pool_scale = 2
conv_scale = 3
image_scale = 28
# image width, image height, filter maps number, input maps number
i_f_map = ((image_scale, image_scale, CHNL1, CHNLIn),  # full
            (30,30,CHNL1,CHNL1), #pooling
            (15,15,CHNL3,CHNL1),  # full
            (17,17,CHNL3,CHNL3),#pooling
            (8,8,CHNL4,CHNL3), #MLP1 excluded
            (8,8,CHNL5,CHNL4),#MLP2
            (18,10) # 7 * 7 * 6
        )

#Conv1
image_shape1 = (mini_batch_size, CHNLIn,
                i_f_map[0][0], i_f_map[0][1])  # C = 28*4/4 = 28
# (batch size, input channels, input depth, input rows, input columns).
filter_shape1=(i_f_map[0][2], i_f_map[0][3], conv_scale, conv_scale) # 4,4,1,3,3
# 3, 1 , 1
#Conv2
# C = 26*4/4 = 26
image_shape2 = (mini_batch_size, i_f_map[0][2], i_f_map[1][0], i_f_map[1][1])
# (10, 4, 1, 26, 26)
filter_shape2=(i_f_map[1][2], i_f_map[1][3], conv_scale, conv_scale)
# 2, 3,1, 3 ,3 
# (output channels, input channels, filter depth, filter rows, filter columns)


#Pool3
image_shape3 = (mini_batch_size, i_f_map[1][2], i_f_map[2][0], i_f_map[2][1])
# (10, 2, 1, 24, 24)
filter_shape3=(i_f_map[2][2], i_f_map[2][3], conv_scale, conv_scale)
# (10, 4, 1, 24, 24)
#Conv4
image_shape4=(mini_batch_size, i_f_map[2][2], i_f_map[3][0], i_f_map[3][1]) # C = 12*4/4 = 12 
filter_shape4=(i_f_map[3][2], i_f_map[3][3], conv_scale, conv_scale)

#Conv5
image_shape5=(mini_batch_size, i_f_map[3][2], i_f_map[4][0], i_f_map[4][1])  
filter_shape5=(i_f_map[4][2], i_f_map[4][3], conv_scale, conv_scale)


# image_shape7 = (mini_batch_size, i_f_map[5][2], i_f_map[6][0], i_f_map[6][1])
# filter_shape7 = (i_f_map[6][2], i_f_map[6][3], conv_scale, conv_scale)

def create_and_test():

    Conv1 = ConvLayer(image_shape=image_shape1,
                      filter_shape=filter_shape1,
                      poolsize=(pool_scale, pool_scale),
                      activation_fn=ReLU,border_mode='full') ## become 28 - 3 + 1 = 26
    
    
    
    Pool2 = PoolLayer(image_shape=image_shape2,
                      filter_shape=filter_shape2,
                      poolsize=(pool_scale, pool_scale),
                      activation_fn=ReLU, _params=Conv1.params)
    
    Conv3 = ConvLayer(image_shape=image_shape3,
                      filter_shape=filter_shape3,
                      poolsize=(pool_scale, pool_scale),
                      activation_fn=ReLU, border_mode='full')

    Pool4 = PoolLayer(image_shape=image_shape4,
                      filter_shape=filter_shape4,
                      poolsize=(pool_scale, pool_scale),
                      activation_fn=ReLU, _params=Conv1.params)

    # MLP1 = FullyConnectedLayer(n_in=(i_f_map[4][2], i_f_map[4][0],
                                    #  i_f_map[4][1]), n_out=i_f_map[5][2]*i_f_map[5][1]*i_f_map[5][0],
                                # activation_fn=ReLU, p_dropout=0.5)

    MLP2 = FullyConnectedLayer(n_in=(i_f_map[5][2], i_f_map[5][0],
                                    i_f_map[5][1]), n_out=i_f_map[6][0]*1*1,
                              activation_fn=ReLU, p_dropout=0.5)

    SMLayer = SoftmaxLayer(n_in=i_f_map[6][0], 
                            n_out=i_f_map[6][1])
    
    
    net = Network([
        Conv1,
        Pool2,
        Conv3,
        Pool4,
        # MLP1,
        MLP2,
        SMLayer
        ], mini_batch_size)
    
    
   
    accuracy_trained, accuracy_test, cost = net.SGD(training_data=training_data, epochs=60, mini_batch_size=mini_batch_size,
                                        eta=0.03, validation_data=validation_data, test_data=test_data, lmbda=1)
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
    accuracy_list.append(accuracy_trained)
    test_accuracylist.append(accuracy_test)
    cost_list.append(cost)

    print("FortestConvModel:")
    print("accuracy_list= ", accuracy_list)
    print("test_accuracylist= ", test_accuracylist)
    print("cost_list= ", cost_list)
    print("Now i = ", i)

plot_n([epoch_indexs, epoch_indexs, epoch_indexs], [accuracy_list,
       test_accuracylist, cost_list], ["trained", "test", "cost"])
