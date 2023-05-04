#### Libraries
# Standard library
# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
# from network3 import *

class ConvLayer(object):
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.

    """

    def __init__(self, filter_shape, image_shape, poolsize=(2, 2),
                 activation_fn=sigmoid, border_mode='valid'):
        """`filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the
        filter width.

        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.

        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.

        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn = activation_fn
        self.border_mode = border_mode

        self.row = self.image_shape[1]*filter_shape[2]*filter_shape[3]
        self.column = self.filter_shape[0]
        self.occupation = self.column*self.row/(36*32)

        print("Conv rows: ", self.row)
        print("Conv columns: ", self.column)
        print("Conv occupy: {:.2%}".format(self.occupation))
        # initialize weights and biases
        n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
        n_in = (image_shape[1]*np.prod(image_shape[2:]))
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(
                    1.0/n_out), size=filter_shape),
                dtype=theano.config.floatX),
            borrow=True)

        self.b = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                dtype=theano.config.floatX),
            borrow=True)

        self.params = [self.w, self.b]

    def __str__(self):
        return f'ConvLayer(Object)'

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = conv2d(
            input=self.inpt, filters=self.w, filter_shape=self.filter_shape,
            input_shape=self.image_shape, border_mode=self.border_mode)
        activated_out = self.activation_fn(
            conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output = activated_out
        self.y_out = T.argmax(self.output, axis=1)
        self.output_dropout = self.output  # no dropout in the convolutional layers

    def cost(self, net):
        "Return the log-likelihood cost."
        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))

    def skip_paramterize(self):
        return False

    def dimension_show(self):
        weight_array = np.array(self.w)
        bias_array = np.array(self.b)
        return weight_array.shape, bias_array.shape


class PoolLayer(object):
    """Used to create a convolutional and a max-pooling
    layer.  

    """

    def __init__(self, filter_shape, image_shape, activation_fn=sigmoid,
                 poolsize=(2, 2), border_mode='valid', _params=[]):
        """`filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the
        filter width.

        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.

        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.

        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn = activation_fn
        self.border_mode = border_mode

        self.params = _params
        self.w = self.params[0]
        self.b = self.params[1]
        print("recycle times in Conv: {0} x {0} = {1}".format(
            self.image_shape[2],self.image_shape[2] *self.image_shape[3]))

    def __str__(self):
        return f'Pool(Object)'

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        #pass from conv to pooling
        self.inpt = inpt.reshape(self.image_shape)
        pooled_out = pool_2d(
            input=self.inpt, ws=self.poolsize, ignore_border=True, mode='max')
        self.output = pooled_out
        self.output_dropout = self.output  # no dropout in the convolutional layers

    def skip_paramterize(self):
        return True

    def dimension_show(self):
        weight_array = np.array(self.w)
        bias_array = np.array(self.b)
        return weight_array.shape, bias_array.shape
    
class FullyConnectedLayer(object):

    def __init__(self, image_shape, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.image_shape = image_shape
        self.n_in = image_shape[1]*image_shape[2]*image_shape[3]
        self.n_out = n_out
        self.p_dropout = p_dropout
        self.activation_fn = activation_fn


        print("n_in: {}, n_out:{}".format(self.n_in, self.n_out))
        self.occupation = self.n_in * self.n_out / (36*32)
        print("Full occupy: {:.2%}".format(self.occupation))
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/n_out), size=(self.n_in, n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def __str__(self):
        return f'FullyConnectedLayer(Object)'

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)

    def cost(self, net):
        # # MSE
        return T.sqr(self.output_dropout-net.y).mean()

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))

    def skip_paramterize(self):
        return False

    def dimension_show(self):
        weight_array = np.array(self.w)
        bias_array = np.array(self.b)
        return weight_array.shape, bias_array.shape

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

        if not bypass:
            decoded_output = quantize_linear_output(output, bits)
            decoded_dropout = quantize_linear_output(
                output_dropout, bits)


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
                    self.layers[int(
                        index/2)].w.set_value(_params[decode_index])

            else:
                if not self.layers[int((index-1)/2)].skip_paramterize():
                    decode_index = decode_index + 1
                    self.layers[int((index-1)/2)
                                ].b.set_value(_params[decode_index])


#### Miscellanea

def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]


def quantize_linear_output(array, bits):
    """
    no need to do binary transformation
    """
    max_v = np.amax(array)
    min_v = np.amin(array)

    resolution = (max_v-min_v)/(2**(bits))
    result = resolution*np.round(array/resolution)
    return result

def rescale_linear_output(array, down, upper):
    """Rescale an arrary linearly."""
    result = (upper - down)*(array - np.min(array))/(np.max(array)-np.min(array)) + down
    return result


def dropout_layer(layer, p_dropout):
    srng = T.shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)
