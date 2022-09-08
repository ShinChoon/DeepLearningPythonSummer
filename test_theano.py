import numpy
import theano
import theano.tensor as T
rng = numpy.random
N = 400
feats = 784
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
training_steps = 10000
x = T.matrix('x')
y = T.vector('y')
w_1 = theano.shared(rng.randn(784, 300), name='w1')
b_1 = theano.shared(numpy.zeros((300,)), name='b1')
w_2 = theano.shared(rng.randn(300), name='w2')
b_2 = theano.shared(0., name='b2')
from theano.tensor.nnet import sigmoid
p_1 = sigmoid(-T.dot(sigmoid(-T.dot(x, w_1)-b_1), w_2)-b_2)
prediction = p_1 > 0.5
xent = -y*T.log(p_1) - (1-y)*T.log(1-p_1)
cost = xent.mean() + 0.01 * (w_2 ** 2).sum()
gw_1, gb_1, gw_2, gb_2 = T.grad(cost, [w_1, b_1, w_2, b_2])
train = theano.function(
                        inputs=[x, y],
                        outputs=[prediction, xent],
                        updates={w_1: w_1-0.1*gw_1, b_1: b_1-0.1*gb_1,
                        w_2: w_2-0.1*gw_2, b_2: b_2-0.1*gb_2})
predict = theano.function(inputs=[x], outputs=prediction)
for i in range(training_steps):
   pred, err = train(D[0], D[1])

print("target values for D:\n{}".format(D[1]))
