import numpy as np
import theano
import theano.tensor as T

pandas.set_option('display.max_columns', None)
theano.config.exception_verbosity = 'high'


def theano_inner(input):
    output = T.sum(input)

    return output


def theano_example(input1, input2):

    output1 = T.set_subtensor(input1[0], [5])
    output2 = theano_inner(input2)

    return output1, output2


input_1 = K.placeholder(shape=(None, 5))  # one example, of an 120x80 RGB image
input_2 = K.placeholder(shape=(None, 5))  # one example, of an 120x80 RGB image

theano_fn = theano_example(input_1, input_2)
fn = K.function(inputs=[input_1, input_2], outputs=theano_fn)

input_1 = ([0], [0.3], [0.5], [0.8], [0.2])
input_2 = np.ones((1, 5))

out1, out2 = fn([input_1, input_2])
print('Input 1:')
print('shape: ', np.shape(input_1))
print(input_1)
print()
print('Input 2:')
print('shape: ', np.shape(input_2))
print(input_2)
print()
print('Output 1:')
print('shape: ', T.shape(out1).eval())
print(out1)
print()
print('Output 2:')
print('shape: ', T.shape(out2).eval())
print(out2)
