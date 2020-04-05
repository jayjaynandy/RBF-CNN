from keras.datasets import cifar10
import keras

class Cifar:
    def __init__(self):
        (train_data, train_labels), (test_data, test_labels) = cifar10.load_data()
        
        self.train_data = train_data.astype('float32')
        self.test_data = test_data.astype('float32')
        self.train_data /= 255
        self.test_data /= 255

        self.train_labels = keras.utils.to_categorical(train_labels, 10)
        self.test_labels = keras.utils.to_categorical(test_labels, 10)

    @staticmethod
    def print():
        return "cifar"

