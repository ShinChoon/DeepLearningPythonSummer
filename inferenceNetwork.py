#### Libraries
# Standard library
# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from network3 import *

class Inference_Network(object):
    """A network model to include multiple layers: Convolution layer, Pool layer and FullyConnected layer
    Args:
        layers: A list of `Layer` instances that describe the network architecture.
        mini_batch_size: An integer value for the mini-batch size used during training.
        cost_list: the list for storing the cost in each epoch of training and test
        accuracy_list:  the list for storing the accuracy in each epoch of training and test
        params: the list to store params from each layer
        decoded_params: the list to store the params after quantization.
        occupation_list: the list to store the utilization ratio of area on IMC
        x: function input variable
        y: function output variable
        output: output variable from the output layer
        output_dropout: dropout of output from the output layer, for the cost calculation

    """

    def __init__(self, layers, mini_batch_size):
        """
        Initializes an Inference_Network object with the given layers and mini-batch size.

        Args:
            layers: A list of Layer instances that describe the network architecture.
            mini_batch_size: An integer value for the mini-batch size used during training.

        Returns:
            None
        """

        # initialize variables
        self.layers = layers
        self.mini_batch_size = mini_batch_size

        self.cost_list = np.array([0])
        self.accuracy_list = np.array([0])

        # expand the assigning process in for loop to skip the pool layer
        self.params = []
        self.decoded_params = []
        for layer in self.layers:
            if not layer.skip_paramterize():
                for param in layer.params:
                    self.params.append(param)

        self.occupation_list = []

        # populate the params and occupation_list based on the layers
        for layer in self.layers:  # skip softmax and MLP
            if not layer.skip_paramterize():
                self.occupation_list.append(layer.occupation)

        # initialize input and output variables
        self.x = T.matrix("x")
        self.y = T.ivector("y")
        # xrange() was renamed to range() in Python 3.
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        # xrange() was renamed to range() in Python 3.

        # set input for each layer based on output of previous layer
        for j in range(1, len(self.layers)):
            prev_layer, layer = self.layers[j-1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
            
        # set final output and output_dropout
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def ADC_DAC(self, output, output_dropout, bits,bypass=False):
        """
        Applies analog-to-digital and digital-to-analog conversion to the given output and output_dropout.

        Args:
            output: The output to be quantized.
            output_dropout: Dropout of output from the output layer, for the cost calculation.
            bits: The number of bits to use for quantization.
            normalize: Whether to normalize the output.
            bypass: Whether to bypass quantization.

        Returns:
            Decoded output and dropout if bypass is False, else returns output and output_dropout.
        """
        if not bypass:
            decoded_output = quantize_linear_output(output, bits)
            decoded_dropout = quantize_linear_output(
                output_dropout, bits)

            return decoded_output, decoded_dropout
        else:
            return output, output_dropout

    def test_network(self, test_data, mini_batch_size, bits):
        """
        Tests the network's accuracy on the given test data.

        Args:
            test_data: The test data.
            mini_batch_size: An integer value for the mini-batch size used during testing.
            bits: The number of bits to use for quantization.

        Returns:
            The network's accuracy on the test data.
        """
        num_test_batches = int(size(test_data)/mini_batch_size/10)
        test_mb_accuracy = np.mean([self.quantization_test_batch(
                                    test_data, j, bits) for j in range(1,2,1)])
        
        print('corresponding test accuracy is {0:.2%} at output solution {1} bits'.format(
            test_mb_accuracy, bits))
        return test_mb_accuracy


### suggest to create an individual network2 class definition
    def quantization_test_batch(self, test_data, batch_index, quantized_bits):
        """Performs quantization on the given test data with the given batch index and number of bits.

        Args:
            test_data (tuple): A tuple containing the test input and test output.
            batch_index (int): An integer value for the batch index.
            quantized_bits (int): The number of bits used in the operation.

        Returns:
            float: The accuracy of the test.
        """
        i = T.lscalar()  # mini-batch index
        test_x, test_y = test_data
        _index = 0  # index to get output at layer 1

        #Conv1
        init_layer = self.layers[0]

        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        vis_layer = theano.function(
            [i], [self.layers[_index].output,
                  self.layers[_index].output_dropout, 
                  self.x],
            givens={
                self.x:
                    test_x[i *
                            self.mini_batch_size: (i+1)*self.mini_batch_size],
            },
        )
        print("batch_index: ", batch_index)
        print("self.mini_batch_size: ", self.mini_batch_size)


        output_L0 = vis_layer(batch_index)
        resize_image = np.reshape(output_L0[2][7], (28, 28))
        np.savetxt('resize_image.csv',
                   resize_image, fmt='%s', delimiter=', ')


        output_0 = output_L0[0]
        output_dropout_0 = output_L0[1]

        np.savetxt('original_conv1_0.csv',
                   output_L0[0][1][0], fmt='%s', delimiter=', ')
        

        _l0_output, _l0_output_dropout = self.ADC_DAC(output_0,
                                                      output_dropout_0, quantized_bits, bypass=False)
        
        output0 = np.int_(_l0_output * 64)
        np.savetxt('dequantized_conv1_0.csv',
                   output0[1][0], fmt='%s', delimiter=' ')
        np.savetxt('dequantized_conv1_1.csv',
                   output0[1][1], fmt='%s', delimiter=' ')
        np.savetxt('dequantized_conv1_2.csv',
                   output0[1][2], fmt='%s', delimiter=' ')
        np.savetxt('dequantized_conv1_3.csv',
                   output0[1][3], fmt='%s', delimiter=' ')
        
        #Pool 1
        self.layers[1].set_inpt(_l0_output,
                                _l0_output_dropout, self.mini_batch_size)
        _data_l1_output = self.layers[1].output.eval()
        _data_l1_dropout = self.layers[1].output_dropout.eval()

        # np.savetxt('pool1_0.csv',
        #    _data_l1_output[7][0], fmt='%s', delimiter=', ')
        
        # np.savetxt('pool1_1.csv',
        #    _data_l1_output[7][1], fmt='%s', delimiter=', ')
        

        #Conv2
        self.layers[2].set_inpt(_data_l1_output,
                                _data_l1_dropout, self.mini_batch_size)
        _data_l2_output = self.layers[2].output.eval()
        _data_l2_output_dropout = self.layers[2].output_dropout.eval()
        _l2_output, _l2_output_dropout = self.ADC_DAC(_data_l2_output,
                                                      _data_l2_output_dropout, quantized_bits, bypass=False)

        output2 = np.int_(_l2_output * 64)
        np.savetxt('dequantized_conv2_0.csv',
                   output2[1][0], fmt='%s', delimiter=' ')
        
        np.savetxt('dequantized_conv2_1.csv',
                   output2[1][1], fmt='%s', delimiter=' ')
        
        np.savetxt('dequantized_conv2_2.csv',
                   output2[1][2], fmt='%s', delimiter=' ')
        
        np.savetxt('dequantized_conv2_3.csv',
                   output2[1][3], fmt='%s', delimiter=' ')
        
        np.savetxt('dequantized_conv2_4.csv',
                   output2[1][4], fmt='%s', delimiter=' ')
        
        np.savetxt('dequantized_conv2_5.csv',
                   output2[1][5], fmt='%s', delimiter=' ')
        
        np.savetxt('dequantized_conv2_6.csv',
                   output2[1][6], fmt='%s', delimiter=' ')
        
        np.savetxt('dequantized_conv2_7.csv',
                   output2[1][7], fmt='%s', delimiter=' ')
        
        #Pool 2
        self.layers[3].set_inpt(_l2_output,
                                _l2_output_dropout, self.mini_batch_size)
        _data_l3_output = self.layers[3].output.eval()
        _data_l3_dropout = self.layers[3].output_dropout.eval()

        # np.savetxt('pool2_0.csv',
        #            _data_l3_output[7][0], fmt='%s', delimiter=', ')
        
        # np.savetxt('pool2_1.csv',
        #            _data_l3_output[7][1], fmt='%s', delimiter=', ')
        
        # np.savetxt('pool2_2.csv',
        #            _data_l3_output[7][2], fmt='%s', delimiter=', ')
        
        # np.savetxt('pool2_3.csv',
        #            _data_l3_output[7][3], fmt='%s', delimiter=', ')

        #FC1
        self.layers[4].set_inpt(_data_l3_output,
                                _data_l3_dropout, self.mini_batch_size)
        _data_l4_output = self.layers[4].output.eval()
        _data_l4_dropout = self.layers[4].output_dropout.eval()

        # np.savetxt('original_fc1.csv',
                #    _data_l4_output, fmt='%s', delimiter=', ')

        _l4_output, _l4_output_dropout = self.ADC_DAC(
            _data_l4_output, _data_l4_dropout, quantized_bits,  bypass=False)

        output4 = np.int_(_l4_output * 64)
        np.savetxt('dequantized_fc1.csv',
                   output4, fmt='%s', delimiter=' ')

        # probably no need to noralize the FC?
        #FC2
        self.layers[5].set_inpt(_l4_output,
                                _l4_output_dropout, self.mini_batch_size)
        _data_l5_output = self.layers[5].output.eval()
        _data_l5_dropout = self.layers[5].output_dropout.eval()

        # np.savetxt('original_fc2.csv',
                #    _data_l5_output, fmt='%s', delimiter=', ')

        _l5_output, _l5_output_dropout = self.ADC_DAC(
            _data_l5_output, _data_l5_dropout, quantized_bits, bypass=False)
        # probably no need to noralize the FC?

        output5 = np.int_(_l5_output * 64)
        np.savetxt('dequantized_fc2.csv',
                   output5, fmt='%s', delimiter=' ')
        accuray_fn = theano.function(
            [i], T.mean(T.eq(self.y, T.argmax(_l5_output, axis=1))),
            givens={
                self.y:
                    test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            },
        )
        test_accuracy = accuray_fn(batch_index)
        print("test_accuracy:  ", test_accuracy)
        return test_accuracy

### suggest to create an individual network2 class definition
    def normal_test_network(self, test_data, mini_batch_size):
        """Tests the inference network with the given test data and mini-batch size.

        Args:
            test_data (tuple): A tuple containing the test input and test output.
            mini_batch_size (int): An integer value for the mini-batch size used during testing.

        Returns:
            float: The accuracy of the test.
        """
        i = T.lscalar()  # mini-batch index
        test_x, test_y = test_data
        num_test_batches = int(size(test_data)/mini_batch_size)
        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                    test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                    test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })

        test_mb_predictions = theano.function(
            [i], self.layers[-1].y_out,
            givens={
                self.x:
                    test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })

        vis_layer0 = theano.function(
            [i], [self.layers[0].output],
            givens={
                self.x:
                    test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })

        if test_data:

            test_accuracy = np.mean(
                [test_mb_accuracy(j) for j in range(num_test_batches)])
            print('corresponding test accuracy is {0:.2%}'.format(
                test_accuracy))

        return test_accuracy

    def reset_params(self, _params=None, scan_range=0):
        """Resets the parameters of the inference network with the given parameters and scan range.

        Args:
            _params (list, optional): The parameters to reset. Defaults to None.
            scan_range (int, optional): The range to scan. Defaults to 0.
        """
        # reset params w,b based on local/external params
        if _params is None:
            _params = self.decoded_params
        decode_index = -1
        for index in range(scan_range):
            # print("index: ", index)
            if index % 2 == 0:
                if not self.layers[int(index/2)].skip_paramterize():
                    decode_index = decode_index + 1
                    self.layers[int(
                        index/2)].w.set_value(_params[decode_index])

            else:
                if not self.layers[int((index-1)/2)].skip_paramterize():
                    decode_index = decode_index + 1
                    self.layers[int((index-1)/2)
                                ].b.set_value(_params[decode_index])
