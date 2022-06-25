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
from network3 import Network, ConvLayer, PoolLayer, FullyConnectedLayer, SoftmaxLayer # softmax plus log-likelihood cost is more common in modern image classification networks.

# read data:
training_data, validation_data, test_data = network3.load_data_shared()
# mini-batch size:
mini_batch_size = 10

#number of nuerons in Conv1
neuron_number_layer1 = 32
neuron_number_layer2 = 32
pool_number = 2

            


from network3 import ReLU
Conv1 = ConvLayer(image_shape=(mini_batch_size, 1, 28, 28),
                  filter_shape=(neuron_number_layer1, 1, 5, 5),
                  poolsize=(pool_number, pool_number),
                  activation_fn=ReLU)

Pool1 = PoolLayer(image_shape=(mini_batch_size, neuron_number_layer1, 24, 24),
                  filter_shape=(neuron_number_layer1, neuron_number_layer1, 5, 5),
                  poolsize=(pool_number, pool_number),
                  activation_fn=ReLU, _params=Conv1.params)

Conv2 = ConvLayer(image_shape=(mini_batch_size, neuron_number_layer1, 12, 12),
                  filter_shape=(neuron_number_layer2, neuron_number_layer1, 5, 5),
                  poolsize=(pool_number, pool_number),
                  activation_fn=ReLU)


Pool2 = PoolLayer(image_shape=(mini_batch_size, neuron_number_layer2, 8, 8),
                  filter_shape=(neuron_number_layer2, neuron_number_layer2, 5, 5),
                  poolsize=(pool_number, pool_number),
                  activation_fn=ReLU, _params=Conv1.params)

net = Network([
    Conv1,
    # PoolLayer(image_shape=(mini_batch_size, neuron_number_layer1, 24, 24),
    #               filter_shape=(neuron_number_layer1, neuron_number_layer1, 5, 5),
    #               poolsize=(pool_number, pool_number),
    #               activation_fn=ReLU),  
    #
    Pool1, 
    Conv2,
    Pool2,
    # PoolLayer(image_shape=(mini_batch_size, neuron_number_layer2, 8, 8),
    #               filter_shape=(neuron_number_layer1, neuron_number_layer2, 5, 5),
    #               poolsize=(pool_number, pool_number),
    #               activation_fn=ReLU),        
    FullyConnectedLayer(n_in=neuron_number_layer2*4*4, n_out=300, activation_fn=ReLU, p_dropout=0.5),
    FullyConnectedLayer(n_in=300, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
    SoftmaxLayer(n_in=1000, n_out=10)], mini_batch_size)
net.SGD(training_data=training_data, epochs=2, mini_batch_size=mini_batch_size, eta=50, validation_data=validation_data, test_data=test_data, lmbda=0.1)