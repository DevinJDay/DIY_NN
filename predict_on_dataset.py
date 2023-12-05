## to handle image data
## pip install opencv-python
import cv2 
import numpy as np
from Model import Model
from dataset_preparation import create_data_mnist

'''
import nnfs
nnfs.init()
# The nnfs.init() does three things: it sets the random seed to 0 (by the default), creates a float32 dtype default, and overrides the original dot product from NumPy. 
# All of these are meant to ensure repeatable results for following along.
'''


# Label index to label name relation
fashion_mnist_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

# Load the model
model = Model.load('fashion_mnist.model')
# Create dataset
X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')



# Shuffle the test dataset
keys = np.array(range(X_test.shape[0]))
np.random.shuffle(keys)
X_test = X_test[keys]
y_test = y_test[keys]
# Scale and reshape samples
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) -
127.5) / 127.5

print("\n Model Evaluation:")
model.evaluate(X_test, y_test)


# Predict on samples from validation dataset
# and print the result
confidences = model.predict(X_test[995:1005])


# and print the result of predictions
predictions = model.output_layer_activation.predictions(confidences)
print("\n Results of predicting on samples from validation dataset:")
print(predictions)
predict_labels =[]
for prediction in predictions:
    predict_labels.append(fashion_mnist_labels[prediction])
print(predict_labels)

# Print their true labels
print("\n True labels of predicting on samples from validation dataset:")
print(y_test[995:1005])
true_labels = []
for true_label in y_test[995:1005]:
    true_labels.append(fashion_mnist_labels[true_label])
print(true_labels)



