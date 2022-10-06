# -*- coding: UTF-8 -*-
# softmax plus log-likelihood cost is more common in modern image classification networks.
from network3 import Network, ConvLayer, PoolLayer, FullyConnectedLayer, SoftmaxLayer
from network3 import ReLU
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
test_accuracylist = []
cost_list = []
epoch_index = 1
epoch_indexs = np.arange(0, epoch_index, 1, dtype=int)

# read data:
training_data, validation_data, test_data = network3.load_data_shared()
# mini-batch size:
mini_batch_size = 10
# 8 X 8
#number of nuerons in Conv1
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
           (6, 6, CHNL3, CHNL2),    #Convol # excluded
           (6, 6, CHNL3, 32), #MLP 6*6*8*32
           (1, 1, 1, 10) #SoftMax # replaced by FullyConnected in test
           )

#Conv1
image_shape1 = (mini_batch_size, 
                CHNLIn, #input channel
                i_f_map[0][0], #scale
                i_f_map[0][1] # scale
                )
filter_shape1 = (i_f_map[0][2], #output channel
                i_f_map[0][3], #input channel
                 conv_scale, conv_scale)

#Pool1
image_shape2 = (mini_batch_size, 
                i_f_map[0][2], #input channel
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

def create_and_test():

    Conv1 = ConvLayer(plt, plt_enable=False, image_shape=image_shape1,
                      filter_shape=filter_shape1,
                      poolsize=(pool_scale, pool_scale),
                      activation_fn=ReLU, border_mode='half')

    Pool1 = PoolLayer(image_shape=image_shape2,
                      filter_shape=filter_shape2,
                      poolsize=(pool_scale, pool_scale),
                      activation_fn=ReLU, _params=Conv1.params)

    Conv2 = ConvLayer(plt, plt_enable=False, image_shape=image_shape3,
                      filter_shape=filter_shape3,
                      poolsize=(pool_scale, pool_scale),
                      activation_fn=ReLU, border_mode='valid')

    Pool2 = PoolLayer(image_shape=image_shape4,
                      filter_shape=filter_shape4,
                      poolsize=(pool_scale, pool_scale),
                      activation_fn=ReLU, _params=Conv2.params)

    Conv3 = ConvLayer(plt, plt_enable=False, image_shape=image_shape5,
                      filter_shape=filter_shape5,
                      poolsize=(pool_scale, pool_scale),
                      activation_fn=ReLU, border_mode='half')

    MLP1 = FullyConnectedLayer(plt, plt_enable=False, image_shape=image_shape6, n_out=i_f_map[5][-1],
                               activation_fn=ReLU, p_dropout=0.5)

    MLP2 = FullyConnectedLayer(plt, plt_enable=False, image_shape=image_shape7,
                               n_out=i_f_map[6][3], activation_fn=ReLU, p_dropout=0.5)
                               
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
            

    accuracy_trained, cost, _params = net.SGD(training_data=training_data, epochs=10, 
                                                mini_batch_size=mini_batch_size, 
                                                eta=0.03, validation_data=validation_data, 
                                                test_data=test_data, lmbda=0.1)
    
    # print("size of layers: ", len(net.layers))  # skipping pooling
    # print("size of params: ", len(net.params))  # skipping pooling

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
        
    # let net2 inherit (w,b) from net
    net2.reset_params(_params, scan_range=len(_params)+4)
    test_accuracy = net2.test_network(test_data,mini_batch_size)
    print("test_accuracy from net2: {:.2%}".format(test_accuracy))
    usage_ratio = net2.occupation_list
    
    return accuracy_trained, test_accuracy, cost, usage_ratio

def plot_n(indexlists, valuelists, labellist):
    if len(indexlists) == len(valuelists) == len(labellist):
        fig, axes = plt.subplots(len(indexlists), 1)
        for ind in range(len(axes)):
            axes[ind].plot(indexlists[ind], valuelists[ind], "-.",
                           label=labellist[ind])  # Plot the chart
            axes[ind].set_title(label=labellist[ind])
            axes[ind].label_outer()

            if ind < len(axes)-1:
                for i, j in zip(indexlists[ind], valuelists[ind]):
                    axes[ind].annotate('{:.2%}'.format(j), xy=(i, j))
            else:
                for i, j in zip(indexlists[ind], valuelists[ind]):
                    axes[ind].annotate('{:.3}'.format(j), xy=(i, j))

        fig.savefig("result_after_zero_4.png")
        plt.show()  # display


if __name__ == '__main__':

    for i in range(epoch_index):
        accuracy_trained, test_accuracy, cost, usage_ratio = create_and_test()
        train_accuracylist.append(accuracy_trained)
        test_accuracylist.append(test_accuracy)
        cost_list.append(cost)
        print("Now i = ", i)

    print("DeepLearning:")
    print("train_accuracylist= ", ['{:.2%}'.format(i) for i in train_accuracylist])
    print("test_accuracylist= ", ['{:.2%}'.format(i) for i in test_accuracylist])
    print("cost_list= ", cost_list)

    plot_n([epoch_indexs, epoch_indexs, epoch_indexs], [train_accuracylist,
           test_accuracylist, cost_list], ["trained accuracy", "test accuracy", "cost in training"])

    # model = Draw_IMC(total_channels=[1, 4, 8], input_sizes=[30, 14],
    #                  MLP_ports=[i_f_map[-2], i_f_map[-1]])
    # model.draw_picture()