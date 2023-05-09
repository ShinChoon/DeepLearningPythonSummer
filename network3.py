"""network3.py
~~~~~~~~~~~~~~
"""

#### Libraries
# Standard library
from theano.tensor.nnet import sigmoid, relu
from theano.tensor import tanh
import pickle
import gzip
import sys
# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal.pool import pool_2d

# Activation functions for neurons
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)

def nan_num(z): 
    z = T.switch(T.isnan(z), 0., z)
    return z


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


def bin_float(reciv_str, _bits=5):
    """Converts a binary string to a floating-point number.

    Args:
        reciv_str (str): A string containing a binary number.
        _bits (int): The number of bits in the binary representation that are used for the fractional part of the number.

    Returns:
        float: The floating-point representation of the binary number.

    """
    # Remove decimal point
    _bits -= 1
    digit_location = reciv_str.find('.')
    if digit_location != -1:
        clip_str = reciv_str[(digit_location+1):]
        str_num = clip_str
    else:
        clip_str = reciv_str
        str_num = clip_str[1:]

    # Determine the sign of the number
    P_flag = False if reciv_str[0] == '1' else True
    # Compute the fractional part of the number
    answer = 0
    factor = 0
    for i in str_num:
        factor += 1
        answer += float(int(i) * (1/(2**factor)))

    # Compute the integer part of the number, if applicable
    factor = 0
    if digit_location != -1:
        reciv_str = reciv_str[0:digit_location]
        reverse_num = reciv_str[::-1]
        for i in reverse_num:
            answer = answer + int(i) * 1 * 2**factor
            factor = factor + 1

    # Negate the number if it is negative and not zero
    if not P_flag and answer != 0:
        answer = -1 * answer

    return answer


def float_bin(number, places=5):
    """Converts a floating-point number to a binary string.

    Args:
        number (float): The number to convert.
        places (int): The number of bits to use for the fractional part of the binary representation.

    Returns:
        str: The binary string representation of the number.

    """

    # Handle NaN values
    if np.isnan(number):
        number = 0
    
    # Determine the sign of the number
    N_flag = True if source <= 0 else False

    # Convert the number to a positive value for processing
    source = float("{:.4f}".format(number))
    _number = source if source >= 0 else -1*source

    # Split the number into its integer and fractional parts
    whole, dec = str(source).split(".")
    dec = int(dec)
    whole = int(whole)

    # Compute the binary representation of the fraction part of the number
    dec = _number - int(whole)
    res = bin(0).lstrip("0b")
    if whole > 0:
        res = bin(whole).lstrip("0b") + "."
    else:
        res = bin(0).lstrip("0b")

    # Compute the binary representation of the fractional part of the number
    for x in range(places-1):
        answer = (decimal_converter(float(dec))) * 2
        whole, _dec = str(answer).split(".")
        if answer > 0:
            dec = answer - int(whole)
        else:
            whole, _dec = str(0.0).split(".")
        res += whole


    # Combine the integer and fractional parts of the binary representation
    result = str(res)
    if N_flag:
        result = '1' + result
    else:
        result = '0' + result

    return result


def decimal_converter(num):
    """Converts a decimal number to a value between 0 and 1.

    Args:
        num (float): The decimal number to convert.

    Returns:
        float: The converted value between 0 and 1.

    """
    while num > 1:
        num /= 10
    return num

def quantize_linear_params(array, bits):
    """Quantizes an array of linear parameters.

    Args:
        array (np.ndarray): The array to quantize.
        bits (int): The number of bits to use for quantization.

    Returns:
        np.ndarray: The quantized array.

    """
    # Calculate the base value for quantization
    base = (2**(bits)-1)/(2**bits)
    # Clip the array to the quantization range
    np.clip(array, -1*base, base, out=array)
    # Calculate the resolution of the quantization
    resolution = (1)/(2**(bits))  # resolution = 1/2**6 = 0.015625
    # Quantize the array
    result = resolution*np.round(array/resolution)
    return result

def quantize_linear_output(array, bits):
    """Quantizes an array of linear output values.

    Args:
        array (np.ndarray): The array to quantize.
        bits (int): The number of bits to use for quantization.

    Returns:
        np.ndarray: The quantized array.

    """
    # Calculate the base value for quantization
    base = (2**(bits)-1)/(2**(bits-2))

    # Clip the array to the quantization range
    np.clip(array, 0, base, out=array)

    # Calculate the resolution of the quantization
    resolution = base/(2**(bits-2))
    
    # Quantize the array
    result = resolution*np.round(array/resolution)
    return result

def rescale_linear_params(array, down, upper):
    """Rescales an array linearly.

    Args:
        array (np.ndarray): The array to rescale.
        down (float): The new minimum value of the array.
        upper (float): The new maximum value of the array.

    Returns:
        np.ndarray: The rescaled array.

    """
    # Calculate the rescaled array
    result = (upper - down)*(array - np.min(array))/(np.max(array)-np.min(array)) + down
    return result

def rescale_linear_training(array):
    """Rescales an array linearly by dividing by its standard deviation.

    Args:
        array (theano.tensor): The array to rescale.

    Returns:
        theano.tensor: The rescaled array.

    """
    W_axes_to_sum = 0
    result = array / T.sqrt(T.sum(T.sqr(array), axis=W_axes_to_sum, keepdims=True))
    return result

def save_data_shared(params, bits=6):
    """Saves the given parameters as CSV files, after quantizing and rescaling them.

    Args:
        filename (str): The name of the file to save.
        params (list): The parameters to save.
        bits (int): The number of bits to use for quantization.

    Returns:
        list: The saved parameters.

    """
    # Calculate the normalization scale
    normalize_scale = 0
    for i in range(bits-1):
        normalize_scale += 1/(2**(i+1))
    
    # Convert shared variables to numpy arrays
    param_list = [param.get_value() for param in params]

    # Initialize variables for quantization
    _param_list_result = []
    decoded_params_output = []
    min_v_list = np.array([])
    max_v_list = np.array([])


    # Quantize each parameter using linear quantization
    for param in param_list:
        max_v_list = np.append(max_v_list, np.amax(param))
        min_v_list = np.append(min_v_list, np.amin(param))


    for param in param_list:
        _param_result = quantize_linear_params(param, bits)
        decoded_params_output.append(_param_result)

    weights = []

    if len(_param_list_result) > 0:
        weights.append([downgrade_dimension(elem)
                        for elem in _param_list_result[0]])

    bias = []
    # no need for MP1 and SM, and exclude weights and bias both
    for i in range(1, len(_param_list_result)):

        if i % 2 == 0:
            # quantitation of weights
            _param_list_local = np.array(_param_list_result[i])
            if len(_param_list_local.shape) >= 4:
                weights.append([downgrade_dimension(elem)
                                for elem in _param_list_local])
            else:
                weights.append(_param_list_local)

        else:
            _param_list_local = np.array(_param_list_result[i])
            bias.append(_param_list_result[i])

    # Save the parameters to CSV files
    np.set_printoptions(threshold=sys.maxsize)
    np.savetxt('params.csv', param_list, fmt='%s', delimiter='')
    np.savetxt('param_decoded.csv', decoded_params_output,
               fmt='%s', delimiter='')
    return param_list


def downgrade_dimension(params):
    """
    Downgrade the dimension of an array by flattening the last two dimensions into a single dimension.

    Args:
        params: a numpy array of shape (..., m, n)

    Returns:
        A 2D numpy array of shape (..., m * n)

    Example:
        Given a numpy array of shape (2, 2, 3), this function returns a flattened 2D array of shape (2, 6).
    """
    result = [e for f in params[0] for e in f]
    return result



def resize_images(data):
    """Resizes images in the input data to 30x30.

    Args:
        data: A tuple of two lists. The first list contains the image data as flattened arrays of length 784,
            and the second list contains the corresponding labels.

    Returns:
        A tuple of two lists with the same format as the input data, but with the images resized to 30x30.
    """
    _result = [[], []]
    for h in data[0]:
        _reshaped = np.reshape(h, (28, 28))
        _padded = np.pad(_reshaped, (0, 0)) # to 30, 30 
        _result[0].append(_padded.flatten())
    _result[1] = data[1]
    return _result

def load_data_shared(filename="mnist.pkl.gz"):
    """Load the MNIST dataset and preprocess it for use in Theano.

    Args:
        filename (str): The name of the file containing the MNIST dataset.

    Returns:
        A list of three tuples, each containing a pair of shared variables 
        (one for the input data and one for the labels) for the training, 
        validation, and test sets, respectively.

    """
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(
        f, encoding="latin1")
    f.close()

    ## Update the dataset
    training_data = resize_images(training_data)
    validation_data = resize_images(validation_data)
    test_data = resize_images(test_data)

    def shared(data):
        """Place the data into shared variables. This allows Theano to copy
        the data to the GPU, if one is available.

        Args:
            data (tuple): A tuple containing the input data and the labels.

        Returns:
            A tuple of shared variables (one for the input data and one for 
            the labels).

        """
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    return [shared(training_data), shared(validation_data), shared(test_data)]


class Network(object):
    """A network model to include multiple layers: Convolution layer, Pool layer and FullyConnected layer
    Args:
        layers: A list of `Layer` instances that describe the network architecture.
        mini_batch_size: An integer value for the mini-batch size used during training.
        epoch_index: index of epoch tests
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
        """Initializes a new instance of the Network class.
        
        Args:
            layers: A list of `Layer` instances that describe the network architecture.
            mini_batch_size: An integer value for the mini-batch size used during training.
        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size

        self.epoch_index = np.array([0])
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
        """Trains the network using mini-batch stochastic gradient descent.
        
        Args:
            training_data: A tuple of numpy arrays representing the training data.
            epochs: An integer value indicating the number of epochs to train for.
            mini_batch_size: An integer value for the mini-batch size used during training.
            eta: A float value representing the learning rate.
            validation_data: A tuple of numpy arrays representing the validation data.
            test_data: A tuple of numpy arrays representing the test data.
            lmbda: A float value for the L2 regularization parameter.
        Returns: 
            accuracy_list[-1]: the last accruacy in the training batch
            cost_list[-1]: the last cost in the training batch
            params: weights and bias

        """
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
        updates = [(param, rescale_linear_training(param-eta*grad))
                   for param, grad in zip(self.params, grads)]


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
            self.decoded_params = save_data_shared(self.params)
            self.reset_params(scan_range=len(self.params)+4)


        print("Finished training network.")
        print("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(
            validation_accuracy, iteration))

        print("self.accuracy_list: ", self.accuracy_list)

        return self.accuracy_list[-1], self.cost_list[-1], self.params


### suggest to create an individual network2 class definition
    def test_network(self, test_data, mini_batch_size, quantized=False):
        """Tests the network on the given test data.
        
        Args:
            test_data: A tuple of numpy arrays representing the test data.
            mini_batch_size: An integer value for the mini-batch size used during testing.
            quantized: A boolean indicating whether the parameters should be quantized.
        Returns: 
            test_accuracy: the average accuracy in a test batch.
        
        """

        i = T.lscalar()  # mini-batch index
        if quantized:
            updates = [(param, 0.03125*T.round(param/0.03125))for param in self.params]
        else:
            updates = [(param, param) for param in self.params]
        test_x, test_y = test_data
        num_test_batches = int(size(test_data)/mini_batch_size/10)
        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y), updates=updates,
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
        """Resets the network parameters to the given values.
        
        Args:
            _params: A list of numpy arrays representing the network parameters.
            scan_range: An integer value indicating the range of indices to scan.
        """

        if _params is None:
            _params = self.decoded_params
        decode_index = -1
        for index in range(scan_range):
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




class ConvLayer(object):
    """Used to create a combination of a convolutional and a max-pooling
    layer. A more sophisticated implementation would separate the two, but for
    our purposes, we'll always use them together, and it simplifies the code,
    so it makes sense to combine them.

    Args:
        filter_shape (tuple): A tuple of length 4, whose entries are the number
            of filters, the number of input feature maps, the filter height, and
            the filter width.
        image_shape (tuple): A tuple of length 4, whose entries are the
            mini-batch size, the number of input feature maps, the image height,
            and the image width.
        poolsize (tuple): A tuple of length 2, whose entries are the y and x
            pooling sizes. Default is (2,2).
        activation_fn (callable): An activation function. Default is sigmoid.
        border_mode (str): A string representing the border mode for the
            convolutional layer. Default is 'valid'.
        row: kernel height * kernel width * input channels
        column: output channels
        occupation: the ratio of occupancy of weight replicas on IMC
        w: theano variable of weights
        b: theano variable of bias
        params: list including weights and bias

    """

    def __init__(self, filter_shape, image_shape, poolsize=(2, 2),
                 activation_fn=sigmoid, border_mode='valid'):
        """
        Initializes a Convolutional Layer object.

        Args:
            filter_shape: a tuple of length 4, whose entries are the number
            of filters, the number of input feature maps, the filter height, and the
            filter width.
            image_shape: a tuple of length 4, whose entries are the
            mini-batch size, the number of input feature maps, the image
            height, and the image width.
            poolsize: a tuple of length 2, whose entries are the y and
            x pooling sizes.
            activation_fn: an activation function for the layer (default: sigmoid)
            border_mode: 'valid' (default) or 'full', specifying whether the filter should be applied only where the whole
            filter fits (valid) or where it partially fits (full).

        Returns:
            A Convolutional Layer object.
        """

        # initialize object attributes
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn = activation_fn
        self.border_mode = border_mode

        # compute some internal attributes
        self.row = self.image_shape[1]*filter_shape[2]*filter_shape[3]
        self.column = self.filter_shape[0]
        self.occupation = self.column*self.row/(36*32)

        # print information about the layer's dimensions
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

        # set object parameters
        self.params = [self.w, self.b]

    def __str__(self):
        """Returns a string representation of the ConvLayer object."""        
        return f'ConvLayer(Object)'

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        """Sets the input for the ConvLayer and computes the output.

        Args:
            inpt: Input to the ConvLayer.
            inpt_dropout: Dropout version of input.
            mini_batch_size: Mini-batch size.

        """
        
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
        """Return False for parameterization."""
        return False

    def dimension_show(self):
        """Return the dimensions of weights and biases."""
        weight_array = np.array(self.w)
        bias_array = np.array(self.b)
        return weight_array.shape, bias_array.shape


class PoolLayer(object):
    """Used to create a convolutional and a max-pooling
    layer.  
    Args:
        filter_shape: A tuple of length 4, whose entries are the number of
            filters, the number of input feature maps, the filter height, and
            the filter width.
        image_shape: A tuple of length 4, whose entries are the mini-batch
            size, the number of input feature maps, the image height, and the
            image width.
        activation_fn: The activation function used in this layer.
        poolsize: A tuple of length 2, whose entries are the y and x pooling sizes.
        border_mode: A string indicating the mode used for the border. Can be 'valid' or 'full'.
        params: A list containing the shared variables for the weights and biases.
        w: theano variable of weights
        b: theano variable of bias
    """

    def __init__(self, filter_shape, image_shape, activation_fn=sigmoid,
                 poolsize=(2, 2), border_mode='valid', _params=[]):
        """Initializes a PoolLayer object.

        Args:
            filter_shape: A tuple of length 4, whose entries are the number of
                filters, the number of input feature maps, the filter height, and
                the filter width.
            image_shape: A tuple of length 4, whose entries are the mini-batch
                size, the number of input feature maps, the image height, and the
                image width.
            activation_fn: The activation function used in this layer.
            poolsize: A tuple of length 2, whose entries are the y and x pooling sizes.
            border_mode: A string indicating the mode used for the border. Can be 'valid' or 'full'.
            _params: A list containing the shared variables for the weights and biases.

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
        """Returns a string representation of the PoolLayer object."""
        return f'Pool(Object)'

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        """Sets the input of the PoolLayer object and performs max-pooling on it.

        Args:
            inpt: The input data to the PoolLayer object.
            inpt_dropout: The input data to the PoolLayer object with dropout applied.
            mini_batch_size: The size of the mini-batch.

        """
        # pass from conv to pooling
        self.inpt = inpt.reshape(self.image_shape)
        pooled_out = pool_2d(
            input=self.inpt, ws=self.poolsize, ignore_border=True, mode='max')
        self.output = pooled_out
        self.output_dropout = self.output  # no dropout in the convolutional layers

    def skip_paramterize(self):
        """Indicates that no parameterization is needed for the PoolLayer object."""        
        return True

    def dimension_show(self):
        """Returns the shape of the weight and bias arrays."""        
        weight_array = np.array(self.w)
        bias_array = np.array(self.b)
        return weight_array.shape, bias_array.shape
    
class FullyConnectedLayer(object):

    """A fully connected layer with sigmoid activation function and dropout.
    Args:
        image_shape: A tuple of length 4, whose entries are the mini-batch size,
            the number of input feature maps, the image height, and the image width.
        n_in: the length of 1d params
        n_out: An integer specifying the number of output units in the layer.
        p_dropout: A float between 0 and 1 specifying the dropout probability (default: 0.0).
        activation_fn: A function specifying the activation function to use (default: sigmoid).
        occupation: The ratio of area of weights replicas on IMC

    """

    def __init__(self, image_shape, n_out, activation_fn=sigmoid, p_dropout=0.0):
        """Initialize the fully connected layer with the given image shape, number
        of output units, activation function, and dropout probability.

        Args:
            image_shape: A tuple of length 4, whose entries are the mini-batch size,
                the number of input feature maps, the image height, and the image width.
            n_in: the length of 1d params
            n_out: An integer specifying the number of output units in the layer.
            p_dropout: A float between 0 and 1 specifying the dropout probability (default: 0.0).
            activation_fn: A function specifying the activation function to use (default: sigmoid).
            params: A list containing the shared variables for the weights and biases.
            w: theano variable of weights
            b: theano variable of bias

        """
        self.image_shape = image_shape
        self.n_in = image_shape[1]*image_shape[2]*image_shape[3]
        self.n_out = n_out
        self.p_dropout = p_dropout
        self.activation_fn = activation_fn


        # Calculate the percentage of the space that the weights and biases occupy
        print("n_in: {}, n_out:{}".format(self.n_in, self.n_out))
        self.occupation = self.n_in * self.n_out / (36*32)
        print("Full occupy: {:.2%}".format(self.occupation))

        # Initialize the weights and biases
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
        """Return a string representation of the fully connected layer object."""        
        return f'FullyConnectedLayer(Object)'

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        """Set the input to the layer, perform dropout, and compute the output.

        Args:
            inpt: An array representing the input to the layer.
            inpt_dropout: An array representing the input to the layer with dropout applied.
            mini_batch_size: An integer representing the mini-batch size.

        """
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)

    def cost(self, net):
        """Return the mean squared error cost for the layer.

        Args:
            net: A neural network object.

        """
        # # MSE
        return T.sqr(self.output_dropout-net.y).mean()

    def accuracy(self, y):
        """Return the accuracy for the mini-batch.

        Args:
            y: An array of the target outputs.
        
        Returns:
            Thenoa experssion of mean scores when the answer mathces the label

        """
        return T.mean(T.eq(y, self.y_out))

    def skip_paramterize(self):
        """Return False to indicate that the layer should not be parameterized."""
        return False

    def dimension_show(self):
        """Return the shapes of weights and bias"""
        weight_array = np.array(self.w)
        bias_array = np.array(self.b)
        return weight_array.shape, bias_array.shape


class SoftmaxLayer(object):
    """Used to create a softmax layer.
    Args: 
        image_shape: A tuple whose entries are the output from the previous layer,
            the number of input feature maps, the image height, and the image width.
        n_in: the length of 1d params
        n_out: An integer specifying the number of output units in the layer.
        p_dropout: A float between 0 and 1 specifying the dropout probability (default: 0.0).
        activation_fn: A function specifying the activation function to use (default: sigmoid).
        params: A list containing the shared variables for the weights and biases.
        w: theano variable of weights
        b: theano variable of bias
        occupation: the ratio of occupancy of weight replicas on IMC
        inpt: softmax input, Theano variable
        output: softmax output, Theano variable


    """

    def __init__(self, image_shape, n_out, p_dropout=0.0):
        """Initialize a SoftmaxLayer object.
        Args:
            image_shape: It is a tuple of length 4, whose entries are the mini-batch
            size, the number of input feature maps, the image height, and the image
            width.
            n_out: The number of output units.
            p_dropout: The probability of dropout.

        """
        self.image_shape = image_shape
        self.n_in = image_shape[1]*image_shape[2]*image_shape[3]        
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        # ???? all zero
        self.w = theano.shared(
            np.zeros((self.n_in, n_out), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]
        self.occupation = self.n_in * self.n_out /(36*32)


    def __str__(self):
        """Returns a string representation of the SoftmaxLayer object."""                
        return f'SofmaxLayer(Object)'

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        """Set the input for the SoftmaxLayer object.
        Args: 
            inpt: is the input for the layer.
            inpt_dropout: is the input for dropout layer.
            mini_batch_size: is the size of the mini-batch.

        """
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
        """Calculate the accuracy for the mini-batch.
        Args:
            y: the true label.

        Returns:
            Thenoa experssion of mean scores when the answer mathces the label

        """
        return T.mean(T.eq(y, self.y_out))

    def skip_paramterize(self):
        """Return False for parameterization."""
        return False

    def dimension_show(self):
        """Return the dimensions of weights and biases."""
        weight_array = np.array(self.w)
        bias_array = np.array(self.b)
        return weight_array.shape, bias_array.shape
        
#### Miscellanea

def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]


def dropout_layer(layer, p_dropout):
    """Apply dropout to the given layer with a given probability.

    `layer` is the layer to apply dropout to.
    `p_dropout` is the probability of dropping out a unit.

    """
    # Create a random stream with a shared seed for reproducibility.
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    # Create a binary mask with the same shape as the layer.
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    # Multiply the layer by the mask and return it.
    return layer*T.cast(mask, theano.config.floatX)
