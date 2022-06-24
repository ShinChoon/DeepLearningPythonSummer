import mnist_loader
import network2
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)


net = network2.Network([784, 10], cost=network2.CrossEntropyCost)
#net.large_weight_initializer()
net.SGD(training_data[:1000], 30, 10, 0.1, 
        lmbda = 1000.0,evaluation_data=validation_data,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True)