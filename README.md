## Overview

### neuralnetworksanddeeplearning.com integrated scripts for Python 3.5.2 and Theano with CUDA support

These scrips are updated ones from the **neuralnetworksanddeeplearning.com** gitHub repository in order to work with Python 3.5.2

The testing file (**test.py**) contains all three networks (network.py, network2.py, network3.py) from the book and it is the starting point to run (i.e. *train and evaluate*) them.

# DeepLearningPythonSummer
## Test script
The network structure is LetNet5 with zero-padding on the input image.
It does the quantization during the training and inference test. The training process is organized in **network3.py** and inference test is organized in **inferenceNetwork.py**.
When the training process finished, the model is saved in csv file param_decoded.csv. 
The inference test read the param_decoded.csv to deploy the weights and bias into test.

## Structure
* The neurons and image sizes are modified in the **i_f_map**:
```python3
i_f_map = ((image_scale, image_scale, CHNL1, CHNLIn),  # same padding
           (28, 28, CHNL1, CHNL1),  # Pool
           (14, 14, CHNL2, CHNL1),  # valid
           (12, 12, CHNL2, CHNL2),  # Pool
           (6, 6, CHNL3, CHNL2),    # Convol # excluded
           (6, 6, CHNL3, 32), # MLP 6*6*8*32
           (1, 1, 1, 10) # SoftMax # replaced by FullyConnected in test
           )
```
Number of Epochs can be set in **epoch_index**.


## Run the training and test
```python
python3 test.py
```
## Do inference test only
comment these in test.py
```python
compiler = CNN_compiler()
accuracy_trained, cost, _params = compiler.training_network()
```

