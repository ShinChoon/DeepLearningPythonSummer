#### Libraries
# Standard library
# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from network3 import *

class Inference_Network(object):

    def __init__(self, layers, mini_batch_size):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.

        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size

        self.epoch_index = np.array([0])
        self.cost_list = np.array([0])
        self.accuracy_list = np.array([0])

        self.y_cost_train = self.epoch_index*2  # Y-axis points
        self.y_accuracy_train = self.epoch_index*2  # Y-axis points
        self.y_cost_evaluate = self.epoch_index*2
        self.y_accuracy_evaluate = self.epoch_index*2  # Y-axis points
        # expand the assigning process in for loop to skip the pool layer
        self.params = []
        self.decoded_params = []
        self.columns = []
        self.rows = []
        for layer in self.layers:
            if not layer.skip_paramterize():
                for param in layer.params:
                    self.params.append(param)

        self.occupation_list = []
        for layer in self.layers:  # skip softmax and MLP
            if not layer.skip_paramterize():
                self.occupation_list.append(layer.occupation)
        self.x = T.matrix("x")
        self.y = T.ivector("y")
        # xrange() was renamed to range() in Python 3.
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        # xrange() was renamed to range() in Python 3.
        for j in range(1, len(self.layers)):
            prev_layer, layer = self.layers[j-1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def ADC_DAC(self, output, output_dropout, bits, normalize=False, bypass=False):
        if normalize:
            if len(output.shape) > 2:
                ## Conv1/2
                for i in range(len(output)):
                    for h in range(len(output[i])):
                        output[i][h] = rescale_linear_output(
                        output[i][h], -1, 1)

                for i in range(len(output_dropout)):
                    for h in range(len(output[i])):
                        output_dropout[i][h] = rescale_linear_output(
                        output_dropout[i][h], -1, 1)
            else:
                #FC1 FC2
                output = rescale_linear_output(output, -1, 1)
                output_dropout = rescale_linear_output(output_dropout, -1, 1)


            # print("output shape: ", output.shape)
            # np.savetxt('normalized_output_fc2.csv',
                    #    output, fmt='%s', delimiter=', ')

        if not bypass:
            decoded_output = quantize_linear_output(output, bits)
            decoded_dropout = quantize_linear_output(
                output_dropout, bits)

            # _quantized_output = local_quantitatize_layer(output, bits)
            # _quantized_output_dropout = local_quantitatize_layer(
            #     output_dropout, bits)

            # # print("_quantized_output shape: ", _quantized_output.shape)
            # # np.savetxt('quantized_output_fc2.csv',
            #         #    _quantized_output, fmt='%s', delimiter=', ')

            # decoded_output = local_dequantitatize_layer(
            #     _quantized_output, bits)
            # decoded_dropout = local_dequantitatize_layer(
            #     _quantized_output_dropout, bits)

            return decoded_output, decoded_dropout
        else:
            return output, output_dropout

    def test_network(self, test_data, mini_batch_size, bits):
        num_test_batches = int(size(test_data)/mini_batch_size/10)
        test_mb_accuracy =self.quantization_test_batch(
                                    test_data, 1, bits)
        
        print('corresponding test accuracy is {0:.2%} at output solution {1} bits'.format(
            test_mb_accuracy, bits))
        return test_mb_accuracy


### suggest to create an individual network2 class definition
    def quantization_test_batch(self, test_data, batch_index, quantized_bits):
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
                   output_L0[0][7][0], fmt='%s', delimiter=', ')
        

        _l0_output, _l0_output_dropout = self.ADC_DAC(output_0,
                                                      output_dropout_0, quantized_bits, normalize=False, bypass=False)
        
        np.savetxt('dequantized_conv1_0.csv',
                   _l0_output[7][0], fmt='%s', delimiter=', ')
        np.savetxt('dequantized_conv1_1.csv',
                    _l0_output[7][1], fmt='%s', delimiter=', ')
        np.savetxt('dequantized_conv1_2.csv',
                    _l0_output[7][2], fmt='%s', delimiter=', ')
        np.savetxt('dequantized_conv1_3.csv',
                    _l0_output[7][3], fmt='%s', delimiter=', ')
        
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
                                                      _data_l2_output_dropout, quantized_bits, normalize=False, bypass=False)

        np.savetxt('dequantized_conv2_0.csv',
                   _l2_output[7][0], fmt='%s', delimiter=', ')
        
        np.savetxt('dequantized_conv2_1.csv',
                   _l2_output[7][1], fmt='%s', delimiter=', ')
        
        np.savetxt('dequantized_conv2_2.csv',
                   _l2_output[7][2], fmt='%s', delimiter=', ')
        
        np.savetxt('dequantized_conv2_3.csv',
                   _l2_output[7][3], fmt='%s', delimiter=', ')
        
        np.savetxt('dequantized_conv2_4.csv',
                   _l2_output[7][4], fmt='%s', delimiter=', ')
        
        np.savetxt('dequantized_conv2_5.csv',
                   _l2_output[7][5], fmt='%s', delimiter=', ')
        
        np.savetxt('dequantized_conv2_6.csv',
                   _l2_output[7][6], fmt='%s', delimiter=', ')
        
        np.savetxt('dequantized_conv2_7.csv',
                   _l2_output[7][7], fmt='%s', delimiter=', ')
        
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
            _data_l4_output, _data_l4_dropout, quantized_bits,  normalize=False, bypass=False)


        # _l4_output = np.array(_l4_output)
        # np.savetxt('dequantized_fc1.csv',
                #    _l4_output, fmt='%s', delimiter=', ')

        # probably no need to noralize the FC?
        #FC2
        self.layers[5].set_inpt(_l4_output,
                                _l4_output_dropout, self.mini_batch_size)
        _data_l5_output = self.layers[5].output.eval()
        _data_l5_dropout = self.layers[5].output_dropout.eval()

        np.savetxt('original_fc2.csv',
                   _data_l5_output, fmt='%s', delimiter=', ')

        _l5_output, _l5_output_dropout = self.ADC_DAC(
            _data_l5_output, _data_l5_dropout, quantized_bits,  normalize=False, bypass=False)
        # probably no need to noralize the FC?

        np.savetxt('dequantized_fc2.csv',
                   _l5_output, fmt='%s', delimiter=', ')
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
        # reset params w,b based on local/external params
        if _params is None:
            _params = self.decoded_params
        decode_index = -1
        for index in range(scan_range):
            # print("index: ", index)
            if index % 2 == 0:
                if not self.layers[int(index/2)].skip_paramterize():
                    decode_index = decode_index + 1
                    array_w = np.array(self.layers[int(index/2)].w.get_value())
                    self.layers[int(
                        index/2)].w.set_value(_params[decode_index])

            else:
                if not self.layers[int((index-1)/2)].skip_paramterize():
                    decode_index = decode_index + 1
                    array_b = np.array(
                        self.layers[int((index-1)/2)].b.get_value())
                    self.layers[int((index-1)/2)
                                ].b.set_value(_params[decode_index])
