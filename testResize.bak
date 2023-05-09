from pickletools import uint8
import cv2
import pickle
import gzip
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt

f = gzip.open("mnist.pkl.gz", 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
f.close()

print("size:training_data ", len(training_data[0]))
print("size:validation_data ", len(validation_data[0]))

print("size:test_data ", len(test_data[0]))



# test_image = test_data[0][1]
# resized_image = np.reshape(test_image,(28,28))
# resized_image_pad = np.pad(resized_image,(2,2))
# resized_image_new = resized_image_pad.flatten()
# backstore_image = np.reshape(resized_image_new, (32,32))

# # resized_image = resize(test_image, (28,28))

# print(np.array_equal(test_image, resized_image))
# print(resized_image)
# resized_image_padded = np.pad(resized_image, (1,1))
# resized_image_padded_reshape = resize(resized_image_padded, (resized_image_padded.shape[0], resized_image_padded.shape[1]))

# print(resized_image_new)
# print(np.array_equal(test_image, resized_image_new))

def resize_images(data):
    _result = [[],[]]
    for h in data[0]:
        _reshaped = np.reshape(h, (28, 28))
        _padded = np.pad(_reshaped, (1, 1))
        _result[0].append(_padded.flatten())
    _result[1] = data[1]
    return _result

_new_set = resize_images(test_data)


# def plot_image(index, data=[], image_shape=(30,30), name=''):
#     plt.figure(index)
#     plt.title('{}: {}'.format(name, data[1][0]))
#     plt.imshow(np.reshape(data[0][0], image_shape), cmap='gray')
#     plt.show()


# plot_image(1, test_data, image_shape=(28,28), name="Initial image")

# plot_image(2, _new_set, image_shape=(32, 32), name="After padding")
_reshaped = np.reshape(_new_set[0][0], (30,30))#[0][0] is the picture with number 7
print(np.shape(_reshaped))

print(_reshaped)

