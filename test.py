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
from network3 import ReLU
from network3 import Network, ConvLayer, PoolLayer, FullyConnectedLayer, SoftmaxLayer # softmax plus log-likelihood cost is more common in modern image classification networks.

# read data:
training_data, validation_data, test_data = network3.load_data_shared()
# mini-batch size:
mini_batch_size = 10

#number of nuerons in Conv1
CHNL1 = 2 
CHNL2 = 4
CHNL3 = 4 
CHNL4 = 4 
CHNL5 = 4 
CHNL6 = 8 
CHNL7 = 8 

pool_scale = 2
conv_scale = 3
image_scale = 28
# image width, image height, filter maps number, input maps number
i_f_map = ((28,28,CHNL1,1), #C = 28*1/4 = 7
            (28,28,CHNL2,CHNL1), #C = 28*2/4 = 14
            (28,28,CHNL3,CHNL2), #C = 28*4/4 = 28
            (14,14,CHNL4,CHNL3), #C = 14*4/4 = 14
            (12,12,CHNL5,CHNL4), #C = 12*4/4 = 12
            (6,6,CHNL6,CHNL5), #C = 6*4/4 = 6
            (4,4,CHNL7,CHNL6),
            (16,10)
        )

#Conv1
image_shape1=(mini_batch_size, 1, i_f_map[0][0], i_f_map[0][1]) # C = 28*4/4 = 28
filter_shape1=(i_f_map[0][2], i_f_map[0][3], conv_scale, conv_scale)
#Conv1_2
image_shape2=(mini_batch_size, i_f_map[0][2], i_f_map[1][0], i_f_map[1][1]) # C = 26*4/4 = 26
filter_shape2=(i_f_map[1][2], i_f_map[1][3], conv_scale, conv_scale)

#Pool1
image_shape3=(mini_batch_size, i_f_map[1][2], i_f_map[2][0], i_f_map[2][1])  
filter_shape3=(i_f_map[2][2], i_f_map[2][3], conv_scale, conv_scale)

#Conv2
image_shape4=(mini_batch_size, i_f_map[2][2], i_f_map[3][0], i_f_map[3][1]) # C = 12*4/4 = 12 
filter_shape4=(i_f_map[3][2], i_f_map[3][3], conv_scale, conv_scale)

#Pool1
image_shape5=(mini_batch_size, i_f_map[3][2], i_f_map[4][0], i_f_map[4][1])  
filter_shape5=(i_f_map[4][2], i_f_map[4][3], conv_scale, conv_scale)

#Conv3
image_shape6=(mini_batch_size, i_f_map[4][2], i_f_map[5][0], i_f_map[5][1]) # C = 10*4/4 = 10
filter_shape6=(i_f_map[5][2], i_f_map[5][3], conv_scale, conv_scale)



Conv1 = ConvLayer(image_shape=image_shape1,
                  filter_shape=filter_shape1,
                  poolsize=(pool_scale, pool_scale),
                  activation_fn=ReLU,border_mode='half') ## become 28 - 3 + 1 = 26

Conv1_2 = ConvLayer(image_shape=image_shape2, 
                  filter_shape=filter_shape2,
                  poolsize=(pool_scale, pool_scale),
                  activation_fn=ReLU,border_mode='half') ## become 26 - 3 + 1 = 24

Pool1 = PoolLayer(image_shape=image_shape3,
                  filter_shape=filter_shape3,
                  poolsize=(pool_scale, pool_scale),
                  activation_fn=ReLU, _params=Conv1.params) ## become 24 / 2 = 12

Conv2 = ConvLayer(image_shape=image_shape4,
                  filter_shape=filter_shape4,
                  poolsize=(pool_scale, pool_scale),
                  activation_fn=ReLU,border_mode='valid') ## become 12 - 3 + 1 = 10

Pool2 = PoolLayer(image_shape=image_shape5,
                  filter_shape=filter_shape5,
                  poolsize=(pool_scale, pool_scale),
                  activation_fn=ReLU, _params=Conv1.params) ## become 24 / 2 = 12
            
Conv3 = ConvLayer(image_shape=image_shape6,
                  filter_shape=filter_shape6,
                  poolsize=(pool_scale, pool_scale),
                  activation_fn=ReLU,border_mode='valid') ## become 10 - 3 + 1 = 8

MLP1 = FullyConnectedLayer(n_in=i_f_map[6][2]*i_f_map[6][0]*i_f_map[6][1], n_out=i_f_map[7][0], activation_fn=ReLU, p_dropout=0.5)

SMLayer = SoftmaxLayer(n_in=i_f_map[7][0], n_out=i_f_map[7][1])


net = Network([
    Conv1,
    Conv1_2,
    Pool1,
    Conv2,
    Pool2,
    Conv3, 
    MLP1,
    SMLayer
    ], mini_batch_size)
net.SGD(training_data=training_data, epochs=2, mini_batch_size=mini_batch_size, eta=0.05, validation_data=validation_data, test_data=test_data, lmbda=25)