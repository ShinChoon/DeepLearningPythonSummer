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
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        # xrange() was renamed to range() in Python 3.
        # for j in range(1, len(self.layers)):
            # prev_layer, layer = self.layers[j-1], self.layers[j]
            # layer.set_inpt(
                # prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)

    def plot_image(self, index, data=[], name=''):
        if self.plt_enable:
            self.plt.figure(index)
            self.plt.title('{}'.format(name))
            print("shape of data: {}".format(data.shape))
            self.plt.imshow(np.reshape(
                data[0], (self.layers[index+1].image_shape[2], self.layers[index+1].image_shape[3])), cmap='gray')
            self.plt.show()

### suggest to create an individual network2 class definition
    def test_network(self, test_data, mini_batch_size):
        i = T.lscalar()  # mini-batch index
        test_x, test_y = test_data
        num_test_batches = int(size(test_data)/mini_batch_size)
        _index = 0
        vis_layer0 = theano.function(
            [i], [self.layers[_index].output, self.layers[_index].output_dropout],
            givens={
                self.x:
                    test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        output_L0 = vis_layer0(0)
        self.plot_image(0, output_L0[0][0], 'layer_{}'.format(0))

        #Pool
        self.layers[1].set_inpt(output_L0[0],
                                output_L0[1], self.mini_batch_size)

        _data_l1_output = self.layers[1].output.eval()
        _data_l1_dropout = self.layers[1].output_dropout.eval()
        _data_l1_output[0] *= 0
        _data_l1_dropout[0] *= 0
        self.plot_image(1, _data_l1_output[0], 'layer_{}'.format(1))

        self.layers[2].set_inpt(_data_l1_output,
                                _data_l1_dropout, self.mini_batch_size)
      
        _data_l2 = self.layers[2].output.eval()
        self.plot_image(2, _data_l2[0], 'layer_{}'.format(2))

        self.layers[3].set_inpt(self.layers[2].output,
                                self.layers[2].output_dropout, self.mini_batch_size)

        self.layers[4].set_inpt(self.layers[3].output,
                                self.layers[3].output_dropout, self.mini_batch_size)

        self.layers[5].set_inpt(self.layers[4].output,
                                self.layers[4].output_dropout, self.mini_batch_size)
                                

        # vis_layer1 = theano.function(
        #     [i], [self.layers[_index+1].output, self.layers[_index+1].output_dropout],
        #     givens={
        #         self.x:
        #             test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
        #     }
        #     )

        # vis_layer2 = theano.function(
        #     [i], [self.layers[_index+2].output,
        #           self.layers[_index+2].output_dropout],
        #     givens={
        #         self.x:
        #             test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
        #     }
        #     )

        # vis_layer3 = theano.function(
        #     [i], [self.layers[_index+3].output,
        #           self.layers[_index+3].output_dropout],
        #     givens={
        #         self.x:
        #             test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
        #     }
        #     )

        # print("issue happens@@@@@@")
        # output_L0 = vis_layer0(0)
        # self.plot_image(0, output_L0[0][0], 'layer_{}'.format(0))

        # self.layers[1].set_inpt(output_L0[0], output_L0[1], num_test_batches)
        # output_L1 = vis_layer1(0)
        # self.plot_image(1, output_L1[0][0], 'layer_{}'.format(1))
    
        # self.layers[2].set_inpt(output_L1[0], output_L1[1], num_test_batches)
        # output_L2 = vis_layer2(0)
        # self.plot_image(2, output_L2[0][0], 'layer_{}'.format(2))

        # self.layers[3].set_inpt(output_L2[0], output_L2[1], num_test_batches)
        # output_L3 = vis_layer3(0)
        # self.plot_image(3, output_L3[0][0], 'layer_{}'.format(3))
        


        # test_mb_accuracy = theano.function(
        #     [i], self.layers[-1].accuracy(self.y),
        #     givens={
        #         self.x:
        #             test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
        #         self.y:
        #             test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
        #     })

        # test_mb_predictions = theano.function(
        #     [i], self.layers[-1].y_out,
        #     givens={
        #         self.x:
        #             test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
        #     })



        test_accuracy = 0


        # if test_data:
        #     test_accuracy = np.mean(
        #         [test_mb_accuracy(j) for j in range(num_test_batches)])
        #     print('corresponding test accuracy is {0:.2%}'.format(
        #         test_accuracy))

        # test_predictions = [test_mb_predictions(
        #     j) for j in range(num_test_batches)]

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
