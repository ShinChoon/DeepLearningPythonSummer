# -*- coding: UTF-8 -*-
# softmax plus log-likelihood cost is more common in modern image classification networks.
from network3 import Network, ConvLayer, PoolLayer, FullyConnectedLayer,  SoftmaxLayer, save_data_shared
from network3 import relu, sigmoid, tanh
from inferenceNetwork import Inference_Network
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import numpy as np
import network3
from draw_IMC import Draw_IMC


# ----------------------
# - network3.py example:

#for test
train_accuracylist = []
quantized_test_accuracylist = []
full_test_accuracylist = []
cost_list = []
epoch_index = 12
epoch_indexs = np.arange(0, epoch_index, 1, dtype=int)

# read data:
training_data, validation_data, test_data = network3.load_data_shared()
# mini-batch size:
mini_batch_size = 10
# bits start value
bits_start = 2
# 8 X 8
# number of nuerons in Conv1
CHNLIn = 1
CHNL1 = 4  # change to 2 to keep columns in constraints
CHNL2 = 8
CHNL3 = 8

usage_ratio = []

pool_scale = 2
conv_scale = 3
image_scale = 28 # padding twice, in hardware padding by 2 directly
# image width, image height, filter maps number, input maps number
i_f_map = ((image_scale, image_scale, CHNL1, CHNLIn),  # same padding
           (28, 28, CHNL1, CHNL1),  # Pool
           (14, 14, CHNL2, CHNL1),  # valid
           (12, 12, CHNL2, CHNL2),  # Pool
           (6, 6, CHNL3, CHNL2),    # Convol # excluded
           (6, 6, CHNL3, 32), # MLP 6*6*8*32
           (1, 1, 1, 10) # SoftMax # replaced by FullyConnected in test
           )

#Conv1
image_shape1 = (mini_batch_size, 
                CHNLIn, # input channel
                i_f_map[0][0], # scale
                i_f_map[0][1] # scale
                )
filter_shape1 = (i_f_map[0][2], # output channel
                i_f_map[0][3], # input channel
                 conv_scale, conv_scale)

#Pool1
image_shape2 = (mini_batch_size, 
                i_f_map[0][2], # input channel
                i_f_map[1][0], i_f_map[1][1])
filter_shape2 = (i_f_map[1][2],  # output channel
                i_f_map[1][3],  # input channel
                conv_scale, conv_scale)


#Conv2
image_shape3 = (mini_batch_size, i_f_map[1][2], i_f_map[2][0], i_f_map[2][1])
filter_shape3 = (i_f_map[2][2], i_f_map[2][3], conv_scale, conv_scale)

#Pool2
image_shape4 = (mini_batch_size, i_f_map[2][2], i_f_map[3][0], i_f_map[3][1])
filter_shape4 = (i_f_map[3][2], i_f_map[3][3], conv_scale, conv_scale)

#Conv3

image_shape5 = (mini_batch_size, i_f_map[3][3], i_f_map[4][0], i_f_map[4][1])
filter_shape5 = (i_f_map[4][2], i_f_map[4][3], conv_scale, conv_scale)

#MLP
image_shape6 = (mini_batch_size, i_f_map[4][3], i_f_map[5][0], i_f_map[5][1])
filter_shape6 = (i_f_map[5][2], i_f_map[5][3], conv_scale, conv_scale)

image_shape7 = (mini_batch_size, i_f_map[5][3], i_f_map[6][0], i_f_map[6][1])
filter_shape7 = (i_f_map[6][2], i_f_map[6][3], conv_scale, conv_scale)

def training_network():

    """
    create models for training network
    """

    Conv1 = ConvLayer(plt, plt_enable=False, image_shape=image_shape1,
                      filter_shape=filter_shape1,
                      poolsize=(pool_scale, pool_scale),
                      activation_fn=relu, border_mode='half')

    Pool1 = PoolLayer(image_shape=image_shape2,
                      filter_shape=filter_shape2,
                      poolsize=(pool_scale, pool_scale),
                      activation_fn=relu, _params=Conv1.params)

    Conv2 = ConvLayer(plt, plt_enable=False, image_shape=image_shape3,
                      filter_shape=filter_shape3,
                      poolsize=(pool_scale, pool_scale),
                      activation_fn=relu, border_mode='valid')

    Pool2 = PoolLayer(image_shape=image_shape4,
                      filter_shape=filter_shape4,
                      poolsize=(pool_scale, pool_scale),
                      activation_fn=relu, _params=Conv2.params)

    Conv3 = ConvLayer(plt, plt_enable=False, image_shape=image_shape5,
                      filter_shape=filter_shape5,
                      poolsize=(pool_scale, pool_scale),
                      activation_fn=relu, border_mode='half')

    MLP1 = FullyConnectedLayer(plt, plt_enable=False, image_shape=image_shape6, n_out=i_f_map[5][-1],
                               activation_fn=relu, p_dropout=0.5)

    MLP2 = FullyConnectedLayer(plt, plt_enable=False, image_shape=image_shape7,
                                n_out=i_f_map[6][3], activation_fn=relu, p_dropout=0.5)

    SMLayer = SoftmaxLayer(plt, plt_enable=False, image_shape=image_shape7,
                           n_out=i_f_map[6][3])

    net = Network(plt, plt_enable=False,
                            layers=[
                    Conv1,
                    Pool1,
                    Conv2,
                    Pool2,
                    MLP1,
                    SMLayer
                    ], 
                mini_batch_size=mini_batch_size)
            

    accuracy_trained, cost, _params, columns = net.SGD(training_data=training_data, epochs=10, 
                                                mini_batch_size=mini_batch_size, 
                                                eta=0.03, validation_data=validation_data, 
                                                test_data=test_data, lmbda=0.1)
    


    return accuracy_trained, cost, _params, columns




def full_quantized_test_network(bits, _params, columns):
    """
    params bits: for changing bits for quantization of output by layers 
    params _params: weights and bias by layers 
    """
    Conv1 = ConvLayer(plt, plt_enable=False, image_shape=image_shape1,
                        filter_shape=filter_shape1,
                        poolsize=(pool_scale, pool_scale),
                        activation_fn=relu, border_mode='half')

    Pool1 = PoolLayer(image_shape=image_shape2,
                        filter_shape=filter_shape2,
                        poolsize=(pool_scale, pool_scale),
                        activation_fn=relu, _params=Conv1.params)

    Conv2 = ConvLayer(plt, plt_enable=False, image_shape=image_shape3,
                        filter_shape=filter_shape3,
                        poolsize=(pool_scale, pool_scale),
                        activation_fn=relu, border_mode='valid')

    Pool2 = PoolLayer(image_shape=image_shape4,
                        filter_shape=filter_shape4,
                        poolsize=(pool_scale, pool_scale),
                        activation_fn=relu, _params=Conv2.params)

    MLP1 = FullyConnectedLayer(plt, plt_enable=False, image_shape=image_shape6, n_out=i_f_map[5][-1],
                                activation_fn=relu, p_dropout=0.5)

    MLP2 = FullyConnectedLayer(plt, plt_enable=False, image_shape=image_shape7,
                                n_out=i_f_map[6][3], activation_fn=relu, p_dropout=0.5)
    net2 = Inference_Network(
                            plt,
                            plt_enable=True,
                            layers=[
                            Conv1,
                            Pool1,
                            Conv2,
                            Pool2,
                            MLP1,
                            MLP2  # FC
                            ], 
                            mini_batch_size=mini_batch_size)

    params_list = [param.get_value() for param in _params]
    net2.reset_params(params_list, scan_range=len(_params)+4)
    full_test_accuracy = net2.normal_test_network(test_data, mini_batch_size)
    print("@@@@@@full_test_accuracy from net2: {:.2%}".format(full_test_accuracy))
    
    decoded_params = save_data_shared("params.csv", _params, columns) 
    
    net2.reset_params(decoded_params, scan_range=len(_params)+4)
    quantized_test_accuracy = net2.test_network(test_data,mini_batch_size,bits)
    print("quantized_test_accuracy from net2: {:.2%}".format(quantized_test_accuracy))
    usage_ratio = net2.occupation_list

    return full_test_accuracy, quantized_test_accuracy, usage_ratio
 

def plot_n(indexlists, valuelists, labellist):
    """
    the length of params should be the same
    params indexlists: list of test indexs 
    params valuelists: list of values should be ploted
    params labellist: list of titles for each plot
    """
    fig, axes = plt.subplots(2, 1)
    ## for accuracy plot
    for ind in range(len(valuelists)-1):
        axes[0].plot(indexlists[0], valuelists[ind],
                     'o-', label=labellist[ind])
        axes[0].legend()
        for i, j in zip(indexlists[0], valuelists[ind]):
            axes[0].annotate('{:.2%}'.format(j), xy=(i, j))
    axes[0].set_title("accuracy show 2-13 bits")
    axes[0].set_xticks(np.arange(min(indexlists[0]), max(indexlists[0])+1, 1))

    ## for cost plot
    axes[1].plot(indexlists[0], valuelists[-1], "-.",
                 label=labellist[-1])  # Plot the chart
    axes[1].legend()
    for i, j in zip(indexlists[0], valuelists[-1]):
        axes[1].annotate('{:.3}'.format(j), xy=(i, j))
    axes[1].set_title("cost show at 2-13 bits")
    axes[1].set_xticks(
        np.arange(min(indexlists[0]), max(indexlists[0])+1, 1))
    # axes[1].label_outer()
    fig.savefig("result_relu_theano_2_13_3.png")
    plt.show()  # display


if __name__ == '__main__':

    for i in range(epoch_index):
        accuracy_trained, cost, _params, columns = training_network()
        full_test_accuracy,quantized_test_accuracy, usage_ratio = full_quantized_test_network(5, _params, columns)
        # quantized_test_accuracy, usage_ratio = quantized_test_network(
            # bits=5, _params=_params, columns=colu
            # mns)
        train_accuracylist.append(accuracy_trained)
        full_test_accuracylist.append(full_test_accuracy)
        quantized_test_accuracylist.append(quantized_test_accuracy)
        cost_list.append(cost)
        print("Now i = ", i)

    print("DeepLearning:")
    print("train_accuracylist= ", ['{:.2%}'.format(i) 
        for i in train_accuracylist])
    print("full_test_accuracylist= ", ['{:.2%}'.format(i) 
        for i in full_test_accuracylist])
    print("quantized_test_accuracylist= ", ['{:.2%}'.format(i) 
        for i in quantized_test_accuracylist])

    print("cost_list= ", cost_list)


    plot_n([epoch_indexs+bits_start, epoch_indexs], [train_accuracylist, full_test_accuracylist, quantized_test_accuracylist,
           cost_list], ["trained accuracy","full test accuracy", "quantized test accuracy", "cost in training"])

    # model = Draw_IMC(total_channels=[1, 4, 8], input_sizes=[30, 14],
    #                  MLP_ports=[i_f_map[-2], i_f_map[-1]])
    # model.draw_picture()
