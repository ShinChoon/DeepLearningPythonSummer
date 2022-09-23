"""network3.py
~~~~~~~~~~~~~~
"""

#### Libraries
# Standard library
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh
import pickle
import gzip

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from theano import printing
from theano.tensor.nnet import conv
from theano.tensor.nnet import conv2d
from theano.tensor.nnet import softmax
import tensorflow as tf
from theano.tensor import shared_randomstreams
from theano.tensor.signal.pool import pool_2d


# Activation functions for neurons
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
def _sigmoid(z): 1 / (1 + T.exp(-z))

def nan_num(z): 
    z = T.switch(T.isnan(z), 0., z)
    return z


# class QuadraticCost(object):

#     @staticmethod
#     def fn(a, y):
#         """Return the cost associated with an output ``a`` and desired output
#         ``y``.

#         """
#         return 0.5*np.linalg.norm(a-y)**2

#     @staticmethod
#     def delta(z, a, y):
#         """Return the error delta from the output layer."""
#         return (a-y) * sigmoid_prime(z)


# class CrossEntropyCost(object):

#     @staticmethod
#     def fn(a, y):
#         """Return the cost associated with an output ``a`` and desired output
#         ``y``.  Note that np.nan_to_num is used to ensure numerical
#         stability.  In particular, if both ``a`` and ``y`` have a 1.0
#         in the same slot, then the expression (1-y)*np.log(1-a)
#         returns nan.  The np.nan_to_num ensures that that is converted
#         to the correct value (0.0).

#         """
#         return -np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

#     @staticmethod
#     def delta(z, a, y):
#         """Return the error delta from the output layer.  Note that the
#         parameter ``z`` is not used by the method.  It is included in
#         the method's parameters in order to make the interface
#         consistent with the delta method for other cost classes.

#         """
#         return (a-y)


#### Constants
GPU = True
if GPU:
    print("Trying to run under a GPU.  If this is not desired, then modify " +
          "network3.py\nto set the GPU flag to False.")
    try:
        theano.config.device = 'gpu'
    except:
        pass  # it's already set
    theano.config.floatX = 'float32'
else:
    print("Running with a CPU.  If this is not desired, then the modify " +
          "network3.py to set\nthe GPU flag to True.")


def bin_float(reciv_str):
    #remove decimal
    digit_location = reciv_str.find('.')
    if digit_location != -1:
        clip_str = reciv_str[(digit_location+1):]
    else:
        clip_str = reciv_str
    P_flag = False if clip_str[0] == '1' else True
    str_num = clip_str[1:]
    reverse_num = str_num[::-1]
    answer = 0.0
    factor = 0
    for i in reverse_num:

        answer = answer + int(i) * 0.0625 * 2**factor
        factor = factor + 1

    factor = 0
    if digit_location != -1:
        for i in reciv_str[0:digit_location]:
            answer = answer + int(i) * 1 * 2**factor
            factor = factor + 1

    if not P_flag:
        answer = -1 * answer

    return answer


def float_bin(number, places=4):
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
    for x in range(places):
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


def normal_initalization(_name, _shape):
    array_factors = np.random.normal(loc=0, scale=0.013298, size=_shape)
    array_limited = np.asarray(array_factors, dtype='int8') * 0.0625
    array_limited_32 = np.asarray(array_limited, dtype=theano.config.floatX)
    params = theano.shared(array_limited_32, name=_name, borrow=True)
    return params


def decimal_converter(num):
    while num > 1:
        # print("num: ", num)
        num /= 10
    return num


def save_data_shared(filename, params, columns):
    """
    Below 3 lines are the step to save it as readable csv
    """
    param_list = [param.get_value() for param in params]
    _param_list_result = []
    _decoded_params = []
    for param in param_list:
        _param_result = quantitatize_layer(param)
        _param_list_result.append(_param_result)

    for param in _param_list_result:
        _param_result = dequantitatize_layer(param)
        _decoded_params.append(_param_result)

    decoded_params_output = []
    for decod_i in _decoded_params:
        array = np.array(decod_i).astype(np.float32)
        decoded_params_output.append(array)

    # quantitation of weights
    # weights = [quantitatize_layer(param_list[0])]
    downgraded_weights = []
    max = []
    min = []
    weights = []

    if len(_param_list_result) > 0:
        weights.append([downgrade_dimension(elem)
                        for elem in _param_list_result[0]])

    bias = []
    repeat_weight_printout = []
    repeat_bias_printout = []
    repeat = 0
    # no need for MP1 and SM, and exclude weights and bias both
    for i in range(1, len(_param_list_result)):

        if i % 2 == 0:
            # quantitation of weights
            downgraded_weights = []
            _param_list_local = np.array(_param_list_result[i])
            if len(_param_list_local.shape) >= 4:
                weights.append([downgrade_dimension(elem)
                                for elem in _param_list_local])
            else:
                weights.append(_param_list_local)

        else:
            _param_list_local = np.array(_param_list_result[i])
            bias.append(_param_list_result[i])
    # repeat_weight_printout = repeat_by_column_weights(weights, columns)
    # repeat_bias_printout = repeat_by_column_bias(bias, columns)

    # np.savetxt('param_b.csv', weights, fmt='%s', delimiter='')
    np.savetxt('params.csv', _param_list_result, fmt='%s', delimiter='')
    np.savetxt('param_decoded.csv', _decoded_params,
               fmt='%s', delimiter='')

    return decoded_params_output


def quantitatize_layer(params):
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
                        abs_normalize = abs(C).max()
                        _C_q = C[index]/abs_normalize*0.9375
                        float_result = float_bin(_C_q)
                        _C_result.append(float_result)
                    _ele_result.append(_C_result)
                _neuron_result.append(_ele_result)
            _params_result.append(_neuron_result)

    elif len(params.shape) == 2:  # normally for MLP
        _params_result = []
        for C in params:
            _C_result = []
            for index in range(len(C)):
                abs_normalize = abs(C).max()
                _C_q = C[index]/abs_normalize*0.9375
                float_result = float_bin(_C_q)
                _C_result.append(float_result)
            _params_result.append(_C_result)
    else:  # normally for bias
        _params_result = []
        for index in range(len(params)):
            abs_normalize = abs(params).max()
            _param_q = params[index]/abs_normalize*0.9375
            float_result = float_bin(_param_q)
            _params_result.append(float_result)

    return _params_result


def dequantitatize_layer(params):
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
                        # float_result = float_bin(C[index])
                        float_result = str(bin_float(C[index]))
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
                # float_result = float_bin(C[index])
                float_result = str(bin_float(C[index]))
                _C_result.append(float_result)
            _params_result.append(_C_result)
    else:  # normally for bias
        _params_result = []
        for index in range(len(params)):
            # params[index] = float("{:.5f}".format(params[index]))
            # float_result = float_bin(params[index])
            float_result = str(bin_float(params[index]))
            _params_result.append(float_result)

    return _params_result


def downgrade_dimension(params):
    result = [e for f in params[0] for e in f]
    return result


def repeat_by_column_weights(source, columns):
    print("source: ", source)
    repeat_source_printout = []
    for h in source:  # in one layer
        repeat_source = []
        for c in h:  # in one kernel
            repeat_source_r = []
            repeat_source_col = []
            for i in range(8):  # repeat as a column
                repeat_source_r.append(c)  # build one 2Cu

            cols = int(columns[source.index(h)])
            for i in range(cols):
                # duplicate as columns in one kernel
                repeat_source_col.append(repeat_source_r)

            # collect 2Cus for kernel sets
            repeat_source.append(repeat_source_col)
        repeat_source_printout.append(repeat_source)  # finish 2Cus for layers

    return repeat_source_printout


def repeat_by_column_bias(layers, columns):
    repeat_resut = []
    for layer in layers:
        wshape, bshape = layer.di
        row = []
        repeat_row = []
        cols = int(columns[layers.index(layer)])
        for ele in layer:
            row.append([[ele for i in range(9)] for h in range(8)])
        repeat_row.append([row for i in range(cols)])
        repeat_resut.append(repeat_row)

    return repeat_resut

#### Load the MNIST


def resize_images(data):
    _result = [[], []]
    for h in data[0]:
        _reshaped = np.reshape(h, (28, 28))
        _padded = np.pad(_reshaped, (0, 0)) # to 30, 30 
        _result[0].append(_padded.flatten())
    _result[1] = data[1]
    return _result

def load_data_shared(filename="mnist.pkl.gz"):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(
        f, encoding="latin1")
    f.close()

## Update the dataset
    training_data = resize_images(training_data)
    validation_data = resize_images(validation_data)
    test_data = resize_images(test_data)

    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.

        """
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    return [shared(training_data), shared(validation_data), shared(test_data)]

#### Main class used to construct and train networks


class Network(object):

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
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        # xrange() was renamed to range() in Python 3.
        for j in range(1, len(self.layers)):
            prev_layer, layer = self.layers[j-1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, lmbda=0.0):
        """Train the network using mini-batch stochastic gradient descent."""
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        # compute number of minibatches for training, validation and testing
        num_training_batches = int(size(training_data)/mini_batch_size)
        num_validation_batches = int(size(validation_data)/mini_batch_size)
        num_test_batches = int(size(test_data)/mini_batch_size)

        # define the (regularized) cost function, symbolic gradients, and updates
        # expand the assigning process in for loop to skip the pool layer
        w_layers = []
        for layer in self.layers:
            if not layer.skip_paramterize():
                w_layers.append((layer.w**2).sum())
        l2_norm_squared = sum(w_layers)
        cost_train = self.layers[-1].cost(self) +\
            0.5*lmbda*l2_norm_squared/num_training_batches
        # need to make grads integer
        grads = T.grad(cost_train, self.params)
        # need to make eta as resolution of quantitation
        updates = [(param, param-eta*grad)
                   for param, grad in zip(self.params, grads)]


        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.
        i = T.lscalar()  # mini-batch index
        train_mb = theano.function(
            [i], cost_train, updates=updates,
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        validate_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                validation_x[i *
                             self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                validation_y[i *
                             self.mini_batch_size: (i+1)*self.mini_batch_size]
            })

        # Do the actual training
        best_validation_accuracy = 0.0
        best_iteration = 0
        for epoch in range(epochs):
            cost_list_local = np.array([0])
            accuracy_list_local = np.array([0])
            for minibatch_index in range(num_training_batches):
                iteration = num_training_batches*epoch+minibatch_index
                if iteration % 1000 == 0:
                    print("Training mini-batch number {0}".format(iteration))
                cost_ij = train_mb(minibatch_index)
                cost_list_local = np.append(cost_list_local, cost_ij)
                if (iteration+1) % num_training_batches == 0:
                    

                    validation_accuracy = np.mean(
                        [validate_mb_accuracy(j) for j in range(num_validation_batches)])
                    print("Epoch {0}: validation accuracy {1:.2%}".format(
                        epoch, validation_accuracy))
                    if validation_accuracy >= best_validation_accuracy:
                        print("This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                    accuracy_list_local = np.append(
                        accuracy_list_local, validation_accuracy)

            cost_mean = np.mean(cost_list_local[1:])
            accuracy_mean = np.mean(accuracy_list_local[1:])
            print("Epoch {0}: training_cost {1}, accuracy {2:.2%}".format(
                epoch, cost_mean, accuracy_mean))

            self.epoch_index = np.append(self.epoch_index, epoch)
            self.cost_list = np.append(self.cost_list, cost_mean)
            self.accuracy_list = np.append(self.accuracy_list, accuracy_mean)
            
            # encode and decode the params
            self.decoded_params = save_data_shared(
                "params.csv", self.params, self.columns)
            self.reset_params(scan_range=len(self.params)+4)


        print("Finished training network.")
        print("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(
            best_validation_accuracy, best_iteration))
        # print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))

        # test_accuracy = self.test_network(test_data, num_test_batches)

        print("self.accuracy_list: ", self.accuracy_list)

        return self.accuracy_list[-1], self.cost_list[-1], self.decoded_params

    def test_network(self, test_data, mini_batch_size):
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

        if test_data:
            test_accuracy = np.mean(
                [test_mb_accuracy(j) for j in range(num_test_batches)])
            print('corresponding test accuracy is {0:.2%}'.format(
                test_accuracy))

        test_predictions = [test_mb_predictions(
            j) for j in range(num_test_batches)]

        # for prediction in test_predictions:
            # print('The corresponding test prediction is ', prediction)
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

                    print(self.layers[int((index)/2)])
                    # print(int((index)/2))

            else:
                if not self.layers[int((index-1)/2)].skip_paramterize():
                    decode_index = decode_index + 1
                    array_b = np.array(
                        self.layers[int((index-1)/2)].b.get_value())
                    self.layers[int((index-1)/2)
                                ].b.set_value(_params[decode_index])
                    
                    print(self.layers[int((index-1)/2)])
                    # print(int((index-1)/2))



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

        self.column = self.filter_shape[0]
        self.row = self.image_shape[1]
        self.occupation = self.column * self.row * \
            filter_shape[2] * filter_shape[3]/(36*32)

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

        # self.w = normal_initalization('w', filter_shape)

        self.b = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                dtype=theano.config.floatX),
            borrow=True)

        # self.b = normal_initalization('b', (filter_shape[0],))

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

    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        self.activation_fn = activation_fn
        self.occupation = self.n_in * self.n_out /(36*32)
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


class SoftmaxLayer(object):

    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        # ???? all zero
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]
        self.occupation = self.n_in * self.n_out /(36*32)


    def __str__(self):
        return f'SofmaxLayer(Object)'

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax((1-self.p_dropout) *
                              T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(
            T.dot(self.inpt_dropout, self.w) + self.b)

    def cost(self, net):
        "Return the log-likelihood cost."
        # return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])
        return T.mean(T.nnet.categorical_crossentropy(self.output_dropout, net.y))

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))

    def skip_paramterize(self):
        return False

    def dimension_show(self):
        weight_array = np.array(self.w)
        bias_array = np.array(self.b)
        return weight_array.shape, bias_array.shape
#### Miscellanea


class Output_FC(object):

    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        self.activation_fn = activation_fn

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

    def __str__(self) -> str:
        return f'Output_FC(Object)'

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn((1-self.p_dropout) *
                              T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)

    def cost(self, net):
        # return T.mean(T.nnet.categorical_crossentropy(self.output_dropout, net.y))
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
#### Miscellanea

def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]


def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)
