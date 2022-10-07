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
    _bits -= 1
    digit_location = reciv_str.find('.')
    if digit_location != -1:
        clip_str = reciv_str[(digit_location+1):]
    else:
        clip_str = reciv_str
    P_flag = False if clip_str[0] == '1' else True
    str_num = clip_str[1:]
    answer = 0
    factor = 1
    for i in str_num:
        answer += float(int(i) * (1/(2**factor)))
        factor += 1

    factor = 1
    if digit_location != -1:
        for i in reciv_str[0:digit_location]:
            answer = answer + int(i) * 1 * 2**factor
            factor = factor + 1

    if not P_flag:
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
    whole = int(whole)
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
            # print("neuron: ", neuron)
            _neuron_result = []
            for ele in neuron:
                _ele_result = []
                for C in ele:
                    _C_result = []
                    for index in range(len(C)):
                        # C[index] = float("{:.5f}".format(C[index]))
                        # float_result = local_float_bin(C[index])
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
                # C[index] = float("{:.5f}".format(C[index]))
                # float_result = local_float_bin(C[index])
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
            # print("neuron: ", neuron)
            _neuron_result = []
            for ele in neuron:

                abs_normalize = np.amax(ele)
                normalize_factor = 1
                if abs_normalize <= normalize_scale:
                    normalize_factor = 1
                else:
                    normalize_factor = 1/abs_normalize*normalize_scale

                _ele_result = []
                for C in ele:
                    _C_result = []
                    for index in range(len(C)):
                        # abs_normalize = abs(C).max()
                        _C_q = C[index]*normalize_factor
                        float_result = local_float_bin(_C_q, places=bits)
                        _C_result.append(float_result)
                    _ele_result.append(_C_result)
                _neuron_result.append(_ele_result)
            _params_result.append(_neuron_result)

    elif len(params.shape) == 2:  # normally for MLP
        _params_result = []

        for C in params:
            _C_result = []
            abs_normalize = np.amax(C)
            normalize_factor = 1
            if abs_normalize <= normalize_scale:
                normalize_factor = 1
            else:
                normalize_factor = 1/abs_normalize*normalize_scale
            for index in range(len(C)):
                # abs_normalize = abs(C).max()
                _C_q = C[index]*normalize_factor
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

    def DAC_ADC(self, output, output_dropout, bits, bypass=False):
        if not bypass:
            _quantized_output = local_quantitatize_layer(output,bits)
            _quantized_output_dropout = local_quantitatize_layer(
                output_dropout,bits)
            decoded_output = local_dequantitatize_layer(_quantized_output,bits)
            decoded_dropout = local_dequantitatize_layer(
                _quantized_output_dropout,bits)

            return decoded_output, decoded_dropout
        else:
            return output, output_dropout

    def test_network(self, test_data, mini_batch_size):
        test_mb_accuracy = np.mean([self.test_batch(
                                    test_data, mini_batch_size, _i, 12) 
                                    for _i in range(mini_batch_size)])
        return test_mb_accuracy

### suggest to create an individual network2 class definition
    def test_batch(self, test_data, mini_batch_size, index, quantized_bits):
        i = T.lscalar()  # mini-batch index
        test_x, test_y = test_data
        num_test_batches = int(size(test_data)/mini_batch_size)
        _index = 0

        #Conv1
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)

        vis_layer = theano.function(
            [i], [self.layers[_index].output, self.layers[_index].output_dropout],
            givens={
                self.x:
                    test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
            },
        )
        
        output_L0 = vis_layer(index)

        original_output_L0 = np.reshape(output_L0[0],  (self.layers[1].image_shape[0],
                                                        self.layers[0].filter_shape[0],
                                                        self.layers[1].image_shape[2],
                                                        self.layers[1].image_shape[3]))
        # self.plot_image(0, original_output_L0[0],
            # ' original layer_{}'.format(0))
        _l0_output, _l0_output_dropout = self.DAC_ADC(output_L0[0], 
                                                      output_L0[1], quantized_bits, bypass=False)
        # self.plot_image(0, _l0_output[0],
                        # ' quantized layer_{}'.format(0))
        # np.savetxt('dequantized_conv1.csv',  output_L0[0][8][2], fmt='%s', delimiter='  ')
        # np.savetxt('quantized_conv1.csv', _l0_output[8][2], fmt='%s', delimiter='  ')

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
        _l2_output, _l2_output_dropout = self.DAC_ADC(_data_l2_output, 
                                                      _data_l2_output_dropout, quantized_bits, bypass=False)

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

        _l4_output, _l4_output_dropout = self.DAC_ADC(
            _data_l4_output, _data_l4_dropout, quantized_bits, bypass=False)

        #FC2
        self.layers[5].set_inpt(_l4_output,
                                _l4_output_dropout, self.mini_batch_size)
        _data_l5_output = self.layers[5].output.eval()
        _data_l5_dropout = self.layers[5].output_dropout.eval()

        _l5_output, _l5_output_dropout = self.DAC_ADC(
            _data_l5_output, _data_l5_dropout, quantized_bits, bypass=False)

        accuray_fn = theano.function(
            [i], T.mean(T.eq(self.y, T.argmax(_l5_output, axis=1))),
            givens={
                self.y:
                    test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            },
        )
        test_accuracy = accuray_fn(index)
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

                    # print(self.layers[int((index)/2)])
                    # print(int((index)/2))

            else:
                if not self.layers[int((index-1)/2)].skip_paramterize():
                    decode_index = decode_index + 1
                    array_b = np.array(
                        self.layers[int((index-1)/2)].b.get_value())
                    self.layers[int((index-1)/2)
                                ].b.set_value(_params[decode_index])

                    # print(self.layers[int((index-1)/2)])
                    # print(int((index-1)/2))
