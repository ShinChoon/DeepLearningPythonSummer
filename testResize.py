from pickletools import uint8
import cv2
import pickle
import gzip
import numpy as np
from skimage.transform import resize

f = gzip.open("mnist.pkl.gz", 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
f.close()


# training_data, validation_data, test_data = np.array(images, dtype=object)

# for i in test_data:
#     for h in i:
#         resized_image = resize(h, (1024))
#         print(resized_image.shape)

test_image = test_data[0][1]
resized_image = np.reshape(test_image,(28,28))
# resized_image = np.pad(resized_image,(2,2))
# # resized_image = resize(test_image, (28,28))

# print(np.array_equal(test_image, resized_image))
# print(resized_image)
# resized_image_padded = np.pad(resized_image, (1,1))
# resized_image_padded_reshape = resize(resized_image_padded, (resized_image_padded.shape[0], resized_image_padded.shape[1]))
resized_image_new = resized_image.flatten()

print(resized_image_new)
print(np.array_equal(test_image, resized_image_new))



