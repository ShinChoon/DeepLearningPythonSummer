# -*- coding: UTF-8 -*-
# softmax plus log-likelihood cost is more common in modern image classification networks.
from network3 import Network, ConvLayer, PoolLayer, FullyConnectedLayer,  SoftmaxLayer, save_data_shared
from network3 import relu, sigmoid, tanh, ReLU
from inferenceNetwork import Inference_Network
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import numpy as np
import network3
import pandas as pd
# from draw_IMC import Draw_IMC



# ----------------------
# - network3.py example:

# Set initial empty lists to store training and test accuracies, and cost values
train_accuracylist = []
quantized_test_accuracylist = []
full_test_accuracylist = []
cost_list = []
usage_ratio = []

# Set the total number of epochs and create an array of epoch indices
epoch_index = 10
epoch_indexs = np.arange(0, epoch_index, 1, dtype=int)

# Set the start value for the number of bits for quantization
# bits start value
bits_start = 2


# Load the training, validation, and test data
training_data, validation_data, test_data = network3.load_data_shared()

# Set the mini-batch size
mini_batch_size = 10
# Set the number of neurons for each convolutional layer
# The first layer has 1 input channel, and output channels are set to 4
# The second and third layers have 8 output channels
CHNLIn = 1
CHNL1 = 4  # change to 2 to keep columns in constraints
CHNL2 = 8
CHNL3 = 8

# Define the scale of pooling, kernels in the convolution and image size
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

class CNN_compiler(object):
    """A CNN compiler to include multiple layers and do training tasks.
    Args: 
        net: Network object containing layers
    """
    def __init__(self):
        """
        Constructor for CNN_compiler class.

        Initializes the Convolutional Neural Network (CNN) models for training network.

        Args:
            None

        Returns:
            None
        """

        # Define the Convolutional and Pooling Layers
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

        Conv3 = ConvLayer(image_shape=image_shape5,
                        filter_shape=filter_shape5,
                        poolsize=(pool_scale, pool_scale),
                        activation_fn=ReLU, border_mode='half')

        # Define the Fully Connected Layers
        MLP1 = FullyConnectedLayer(image_shape=image_shape6, n_out=i_f_map[5][-1],
                                   activation_fn=ReLU, p_dropout=0.5)

        MLP2 = FullyConnectedLayer(image_shape=image_shape7,
                                    n_out=i_f_map[6][3], activation_fn=ReLU, p_dropout=0.5)

        # Define the Softmax Layer
        SMLayer = SoftmaxLayer(image_shape=image_shape7,
                            n_out=i_f_map[6][3])

        # Define the network by connecting all layers
        self.net = Network(layers=[
            Conv1,
                        Pool1,
                        Conv2,
                        Pool2,
                        MLP1,
                        SMLayer
                        ], 
                    mini_batch_size=mini_batch_size)



    def training_network(self):
        """
        Function to train the CNN network.

        Trains the CNN using Stochastic Gradient Descent (SGD) and returns the accuracy,
        cost, and parameters of the trained network.

        Args:
            None

        Returns:
            accuracy_trained (float): the accuracy of the trained network
            cost (list): list of costs during the training process
            _params (list): list of trained parameters of the network
        """        
        accuracy_trained, cost, _params = self.net.SGD(training_data=training_data, epochs=0, 
                                                    mini_batch_size=mini_batch_size, 
                                                    eta=0.03, validation_data=validation_data, 
                                                    test_data=test_data, lmbda=0.1)



        return accuracy_trained, cost, _params


class CNN_simulator(object):
    """An CNN simulator to include multiple layers and do inference test tasks.
    Args: 
        net: Network object containing layers
    """
    def __init__(self):
        """
        Constructor for CNN_compiler class.

        Initializes the Convolutional Neural Network (CNN) models for tesing network.

        Args:
            None

        Returns:
            None
        """
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

        # Define the Fully Connected Layers
        MLP1 = FullyConnectedLayer(image_shape=image_shape6, n_out=i_f_map[5][-1],
                                    activation_fn=ReLU, p_dropout=0.5)

        MLP2 = FullyConnectedLayer(image_shape=image_shape7,
                                    n_out=i_f_map[6][3], activation_fn=ReLU, p_dropout=0.5)
        self.net = Inference_Network(
                                layers=[
                                Conv1,
                                Pool1,
                                Conv2,
                                Pool2,
                                MLP1,
                                MLP2  # FC
                                ], 
                                mini_batch_size=mini_batch_size)

    def raw_test_network(self, _params):
        """
        Perform raw testing on the network given the set of parameters.
        
        Args:
            _params: list of trained parameters of the network
    
        Returns:
            full_test_accuracy: accuracy of the raw test on the network
        """
        params_list = _params
        self.net.reset_params(params_list, scan_range=len(_params)+4)
        full_test_accuracy = self.net.normal_test_network(test_data, mini_batch_size)
        print("full_test_accuracy from self.net: {:.2%}".format(full_test_accuracy))
        return full_test_accuracy


    def quantized_test_network(self, bits, _params):
        """
        Perform quantized testing on the network given the set of parameters and bit size.

        Args:
            bits: bit size of the parameters
            _params: list of trained parameters of the network

        Returns:
            quantized_test_accuracy: accuracy of the quantized test on the network
            usage_ratio: list of memory usage ratio of each layer of the network
        """
        self.net.reset_params(_params, scan_range=len(_params)+4)
        quantized_test_accuracy = self.net.test_network(test_data,mini_batch_size,bits)
        print("quantized_test_accuracy from self.net: {:.2%}".format(quantized_test_accuracy))
        usage_ratio = self.net.occupation_list

        return quantized_test_accuracy, usage_ratio
 

def plot_n(indexlists, valuelists, labellist):
    """
    Plot multiple lines on a single figure, and show the results.

    Args:
        indexlists (list): List of test indexes.
        valuelists (list): List of values to be plotted.
        labellist (list): List of titles for each plot.
    """
    fig, axes = plt.subplots(2, 1)
    # for accuracy plot
    for ind in range(len(valuelists)-1):
        axes[0].plot(indexlists[0], valuelists[ind],
                     'o-', label=labellist[ind])
        axes[0].legend()
        for i, j in zip(indexlists[0], valuelists[ind]):
            axes[0].annotate('{:.2%}'.format(j), xy=(i, j))
    axes[0].set_title(
        "training, full and quantized test by 10 times")
    axes[0].set_xticks(np.arange(min(indexlists[0]), max(indexlists[0])+1, 1))

    axes[1].plot(indexlists[0], valuelists[-1], "-.",
                 label=labellist[-1])  # Plot the chart
    axes[1].legend()
    for i, j in zip(indexlists[0], valuelists[-1]):
        axes[1].annotate('{:.3}'.format(j), xy=(i, j))
    # axes[1].set_title("quantized test with 10 samples per epoch")
    axes[1].set_xticks(
        np.arange(min(indexlists[0]), max(indexlists[0])+1, 1))
    
    fig.savefig("theano_traing_fulltest.png")
    plt.show()  # display


def float_bin(number, places=2):
    """
    Convert a floating point number to a binary string.

    Args:
        number (float): The number to be converted.
        places (int): The number of bits.

    Returns:
        int: The integer number.
    """
    if np.isnan(number):
        number = 0
    # # define max and min range
    # if number > (2**2-1)/2**2:
    #     number = (2**2-1)/2**2
    # if number < -1 * (2**2-1)/2**2:
    #     number = -1 * (2**2-1)/2**2
    # source = float("{:.5f}".format(number))
    # result = float_to_int8(source, places=places)
    if number > (2**6-1)/2**6:
        number = (2**6-1)/2**6
    if number < -1 * (2**6-1)/2**6:
        number = -1 * (2**6-1)/2**6
    resolution = 2**(-6)  # corresponding to python
    result = round(number/resolution)
    return result


def convert_float_int8(number, isweights):
    """
    Convert a floating point number to a signed 8-bit integer.

    Args:
        number (float): The number to be converted.
        isweights (bool): Indicates whether the number represents weights.

    Returns:
        int: The converted number.
    """
    answer_in = 0
    if isweights:
        if number > (2**(8-2)-1)/2**6:
            number = (2**(8-2)-1)/2**6
        if number < -1 * (2**(8-2)-1)/2**6:
            number = -1 * (2**(8-2)-1)/2**6

        resolution = 2**(-8+2)  # sign bits by 2 + value range by 4
        answer_in = abs(round(number/resolution))
        if(number > 0):
            answer_in = answer_in + 64
        if(number < 0):
            answer_in = answer_in + 128
    else:
        if number > (2**6-1)/2**6:
            number = (2**6-1)/2**6
        if number < -1 * (2**6-1)/2**6:
            number = -1 * (2**6-1)/2**6
        resolution = 2**(-6)  # corresponding to python
        answer_in = abs(round(number/resolution))
    return answer_in


def converstring_int_list(nu_array):
    """
    Convert a list of strings to a numpy array of floats.

    params nu_array: list of strings to convert
    return: numpy array of floats
    """
    num_array = np.array(nu_array)
    num_array = num_array.astype(np.float32)
    return num_array


def remove_enpty_space(_list):
    """
    Remove empty spaces from a list of strings.

    params _list: list of strings to remove spaces from
    return: new list with spaces removed
    """
    w22 = []
    for ele in _list:
        if ele.strip():
            w22.append(ele)
    return w22


def param_extraction():
    """Extract parameters from a CSV file and store them in a list."""
    # Read the CSV file containing the parameters
    rainfall = pd.read_csv('param_decoded.csv', sep=',', header=None)
    sized_data = rainfall
    smessage = sized_data.values

    # Initialize variables for storing the extracted parameters
    params = []
    weights_1 = []
    weights_map_1 = []

    bias_1 = []
    bias_map_1 = []

    weights_2 = []
    sub_weights_map = []
    weights_map_2 = []

    bias_2 = []
    bias_map_2 = []

    w_fc1_group = []
    weights_fc1 = []
    bias_fc_1 = []

    w_fc2_group = []
    weights_fc2 = []
    bias_fc_2 = []

    counter = 0

    # Extract weights_map_1
    for i in range(0, 12):
        w = smessage[i][0].split(']')
        w = w[0].split('[')
        ww = remove_enpty_space(w)
        ww = ww[0].split(' ')
        ww = remove_enpty_space(ww)

        weights_1.append(ww)

        if((i+1) % 3 == 0) and (i > 0):
            weights_map_1.append([weights_1])
            weights_1 = []

    weights_map_1 = converstring_int_list(weights_map_1)
    print("weights_map_1: ", np.shape(weights_map_1))
    # prolog
    params.append(weights_map_1)

    # Extract bias_map_1
    b = smessage[12][0].split(']')
    b = b[0].split('[')
    bb = remove_enpty_space(b)
    bb = bb[0].split(' ')
    bias_1 = remove_enpty_space(bb)
    bias_map_1 = converstring_int_list(bias_1)
    params.append(bias_map_1)

    # Extract weights_map_2
    for i in range(13, 109):
        w = smessage[i][0].split(']')
        w = w[0].split('[')
        ww = remove_enpty_space(w)
        ww = ww[0].split(' ')
        ww = remove_enpty_space(ww)

        weights_2.append(ww)

    weights_map_2 = np.reshape(weights_2, (8, 4, 3, 3))
    weights_map_2 = converstring_int_list(weights_map_2)
    params.append(weights_map_2)

    # Extract bias_map_2
    for i in range(109, 111):
        b = smessage[i][0].split(']')
        b = b[0].split('[')
        bb = remove_enpty_space(b)
        bb = bb[0].split(' ')
        bb = remove_enpty_space(bb)
        for ele in bb:
            bias_2.append(ele)

    bias_map_2 = converstring_int_list(bias_2)
    params.append(bias_map_2)

    for index in range(288):
        weights_fc1 = []
        for i in range(111+index*5, 116+index*5):
            w = smessage[i][0].split(']')
            w = w[0].split('[')
            ww = remove_enpty_space(w)
            ww = ww[0].split(' ')
            ww = remove_enpty_space(ww)
            for ele in ww:
                weights_fc1.append(ele)

        w_fc1_group.append(weights_fc1)


    w_fc1_group = converstring_int_list(w_fc1_group)
    params.append(w_fc1_group)

    for i in range(1551, 1556):
        b = smessage[i][0].split(']')
        b = b[0].split('[')
        bb = remove_enpty_space(b)
        bb = bb[0].split(' ')
        bb = remove_enpty_space(bb)
        for ele in bb:
            bias_fc_1.append(ele)

    bias_fc_1 = converstring_int_list(bias_fc_1)
    params.append(bias_fc_1)

    for index in range(32):
        weights_fc2 = []
        for i in range(1556+index*2, 1558+index*2):
            w = smessage[i][0].split(']')
            w = w[0].split('[')
            ww = remove_enpty_space(w)
            ww = ww[0].split(' ')
            ww = remove_enpty_space(ww)
            for ele in ww:
                weights_fc2.append(ele)

        w_fc2_group.append(weights_fc2)
    w_fc2_group = converstring_int_list(w_fc2_group)
    params.append(w_fc2_group)

    for i in range(1619, 1621):
        # print("smessage[i][0]: ", smessage[i][0])
        b = smessage[i][0].split(']')
        b = b[0].split('[')
        bb = remove_enpty_space(b)
        bb = bb[0].split(' ')
        bb = remove_enpty_space(bb)
        for ele in bb:
            bias_fc_2.append(ele)

    bias_fc_2 = converstring_int_list(bias_fc_2)
    params.append(bias_fc_2)

    return params


if __name__ == '__main__':
    # Training and testing the CNN compiler and extracting the optimized parameters
    for i in range(epoch_index):
        # compiler = CNN_compiler()
        # accuracy_trained, cost, _params = compiler.training_network()
        _params = param_extraction()

        # Testing the network with optimized parameters
        simulator = CNN_simulator()
        full_test_accuracy = simulator.raw_test_network(_params)
        quantized_test_accuracy, usage_ratio = simulator.quantized_test_network(
            i+bits_start, _params)

        # Storing the results for plotting
        # train_accuracylist.append(accuracy_trained)
        full_test_accuracylist.append(full_test_accuracy)
        quantized_test_accuracylist.append(quantized_test_accuracy)
        # cost_list.append(cost)
        print("Now i = ", i)

    # Printing the results
    print("DeepLearning:")
    print("train_accuracylist= ", ['{:.2%}'.format(i) 
        for i in train_accuracylist])
    print("full_test_accuracylist= ", ['{:.2%}'.format(i) 
        for i in full_test_accuracylist])
    print("quantized_test_accuracylist= ", ['{:.2%}'.format(i) 
        for i in quantized_test_accuracylist])

    # Plotting the results
    # plot_n([epoch_indexs+bits_start, epoch_indexs], [train_accuracylist, full_test_accuracylist, quantized_test_accuracylist],
        #    ["trained accuracy","full test accuracy", "quantized test accuracy"])

    # model = Draw_IMC(total_channels=[1, 4, 8], input_sizes=[30, 14],
    #                  MLP_ports=[i_f_map[-2], i_f_map[-1]])
    # model.draw_picture()
