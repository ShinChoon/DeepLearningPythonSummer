#### Libraries
# Standard library
from theano.tensor.nnet import sigmoid
import pickle
import gzip
# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal.pool import pool_2d
from network3 import *


def local_bin_float(reciv_str, _bits=5):
    #remove decimal
    digit_location = reciv_str.find('.')
    if digit_location != -1:
        clip_str = reciv_str[(digit_location+1):]
        str_num = clip_str
    else:
        clip_str = reciv_str
        str_num = clip_str[1:]
    P_flag = False if reciv_str[0] == '1' else True
    answer = 0
    factor = 0
    for i in str_num:
        factor += 1
        answer += float(int(i) * (1/(2**factor)))

    factor = 0
    if digit_location != -1:
        reciv_str = reciv_str[1:digit_location]
        reverse_num = reciv_str[::-1]
        for i in reverse_num:
            answer = answer + int(i) * 1 * 2**factor
            factor = factor + 1

    if not P_flag and answer != 0:
        answer = -1 * answer

    return answer


def local_float_bin(number, places=5):
    number = float(number)
    if np.isnan(number):
        number = 0
    source = float("{:.4f}".format(number))
    N_flag = True if source <= 0 else False
    _number = source if source >= 0 else -1*source
    whole, dec = str(source).split(".")
    dec = int(dec)
    whole = abs(int(whole))
    dec = _number - int(whole)
    res = bin(0).lstrip("0b")
    if whole > 0:
        #detect if any value more than 1
        res = bin(whole).lstrip("0b") + "."
    else:
        res = bin(0).lstrip("0b")
    for x in range(places-1):
        answer = (decimal_converter(float(dec))) * 2
        # Convert the decimal part
        # to float 4-digit again
        whole, _dec = str(answer).split(".")
        if answer > 0:
            dec = answer - int(whole)
        else:
            whole, _dec = str(0.0).split(".")
        # Keep adding the integer parts
        # receive to the result variable
        res += whole
    result = str(res)
    if N_flag:
        result = '1' + result
    else:
        result = '0' + result

    return result


def local_dequantitatize_layer(params, bits):
    # params: one layer
    #debug
    params = np.array(params)
    _params_result = []
    if len(params.shape) > 2:  # normally for Conv
        for neuron in params:  # in one kernel
            _neuron_result = []
            for ele in neuron:
                _ele_result = []
                for C in ele:
                    _C_result = []
                    for index in range(len(C)):
                        float_result = local_bin_float(C[index], bits)
                        _C_result.append(float_result)
                    _ele_result.append(_C_result)
                _neuron_result.append(_ele_result)
            _params_result.append(_neuron_result)

    elif len(params.shape) == 2:  # normally for MLP
        _params_result = []
        for C in params:
            _C_result = []
            for index in range(len(C)):
                float_result = local_bin_float(C[index], bits)
                _C_result.append(float_result)
            _params_result.append(_C_result)

    return np.array(_params_result)

def local_quantitatize_layer(params, bits):
    # params: one layer
    #debug
    # generate the normalization scale
    normalize_scale = 0
    for i in range(bits):
        normalize_scale += 1/(2**(i+1))

    params = np.array(params)
    _params_result = []
    if len(params.shape) > 2:  # normally for Conv
        for neuron in params:  # in one kernel
            _neuron_result = []
            for ele in neuron:
                _ele_result = []

                for C in ele:
                    _C_result = []
                    for index in range(len(C)):
                        _C_q = C[index]
                        float_result = local_float_bin(_C_q, places=bits)
                        _C_result.append(float_result)
                    _ele_result.append(_C_result)
                _neuron_result.append(_ele_result)
            _params_result.append(_neuron_result)

    elif len(params.shape) == 2:  # normally for MLP
        _params_result = []
        for C in params:
            _C_result = []

            for index in range(len(C)):
                _C_q = C[index]
                float_result = local_float_bin(_C_q, places=bits)
                _C_result.append(float_result)
            _params_result.append(_C_result)

    return np.array(_params_result)


class Inference_Network(object):

    def __init__(self, plt, plt_enable, layers, mini_batch_size):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.

        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.plt = plt
        self.plt_enable = plt_enable

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
                layer.plt_enable = plt_enable
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

    def plot_image(self, index, data=[], name=''):
        if self.plt_enable:
            self.plt.figure(index)
            self.plt.title('{}'.format(name))
            # print("shape of data: {}".format(data.shape))
            self.plt.imshow(np.reshape(
                data[0],
                (self.layers[index+1].image_shape[2],
                 self.layers[index+1].image_shape[3])),
                cmap='gray')
            self.plt.show()

    def ADC_DAC(self, output, output_dropout, bits, normalize=False, bypass=False):
        if normalize:
            if len(output.shape) > 2:
                ## Conv1/2
                for i in range(len(output)):
                    for h in range(len(output[i])):
                        output[i][h] = rescale_linear_params(
                        output[i][h], -1, 1)

                for i in range(len(output_dropout)):
                    for h in range(len(output[i])):
                        output_dropout[i][h] = rescale_linear_params(
                        output_dropout[i][h], -1, 1)
            else:
                #FC1 FC2
                output = rescale_linear_params(output, -1, 1)
                output_dropout = rescale_linear_params(output_dropout, -1, 1)


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
        test_mb_accuracy = np.mean([self.quantization_test_batch(
                                    test_data, _i, bits)
                                    for _i in range(num_test_batches)])
        
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
            [i], [self.layers[_index].output, self.layers[_index].output_dropout],
            givens={
                self.x:
                    test_x[i *
                            self.mini_batch_size: (i+1)*self.mini_batch_size],
            },
        )

        output_L0 = vis_layer(batch_index)


        output_0 = output_L0[0]
        output_dropout_0 = output_L0[1]

        # np.savetxt('original_conv1.csv',
                #    output_L0[0][3][2], fmt='%s', delimiter=', ')

        _l0_output, _l0_output_dropout = self.ADC_DAC(output_0,
                                                      output_dropout_0, quantized_bits, normalize=False, bypass=False)
        
        # np.savetxt('dequantized_conv1.csv',
        #            _l0_output[3][2], fmt='%s', delimiter=', ')
        #Pool 1
        self.layers[1].set_inpt(_l0_output,
                                _l0_output_dropout, self.mini_batch_size)
        _data_l1_output = self.layers[1].output.eval()
        _data_l1_dropout = self.layers[1].output_dropout.eval()
        #Conv2
        self.layers[2].set_inpt(_data_l1_output,
                                _data_l1_dropout, self.mini_batch_size)
        _data_l2_output = self.layers[2].output.eval()
        _data_l2_output_dropout = self.layers[2].output_dropout.eval()
        _l2_output, _l2_output_dropout = self.ADC_DAC(_data_l2_output,
                                                      _data_l2_output_dropout, quantized_bits, normalize=False, bypass=False)

        # np.savetxt('dequantized_conv2.csv',
                #    output_L0[0][8][2], fmt='%s', delimiter=', ')
        #Pool 2
        self.layers[3].set_inpt(_l2_output,
                                _l2_output_dropout, self.mini_batch_size)
        _data_l3_output = self.layers[3].output.eval()
        _data_l3_dropout = self.layers[3].output_dropout.eval()

        #FC1
        self.layers[4].set_inpt(_data_l3_output,
                                _data_l3_dropout, self.mini_batch_size)
        _data_l4_output = self.layers[4].output.eval()
        _data_l4_dropout = self.layers[4].output_dropout.eval()

        np.savetxt('original_fc1.csv',
                   _data_l4_output, fmt='%s', delimiter=', ')

        _l4_output, _l4_output_dropout = self.ADC_DAC(
            _data_l4_output, _data_l4_dropout, quantized_bits,  normalize=False, bypass=False)


        # _l4_output = np.array(_l4_output)
        np.savetxt('dequantized_fc1.csv',
                   _l4_output, fmt='%s', delimiter=', ')

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
        num_test_batches = int(size(test_data)/mini_batch_size/10)
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
