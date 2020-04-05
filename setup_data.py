import numpy as np
import os
import gzip
import urllib.request

def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(num_images*28*28)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data / 255) - 0.5
        data = data.reshape(num_images, 28, 28, 1)
        return data

def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return (np.arange(10) == labels[:, None]).astype(np.float32)

class MNIST:
    def __init__(self):
        _path = "./extras/mnist_data/"
        print(_path)        
        if not os.path.exists(_path):
            os.mkdir(_path)
            files = ["train-images-idx3-ubyte.gz",
                     "t10k-images-idx3-ubyte.gz",
                     "train-labels-idx1-ubyte.gz",
                     "t10k-labels-idx1-ubyte.gz"]
            for name in files:
                urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/' + name, "data/"+name)

        self.train_data = extract_data(_path+"train-images-idx3-ubyte.gz", 60000)+0.5
        self.train_labels = extract_labels(_path+"train-labels-idx1-ubyte.gz", 60000)
        self.test_data = extract_data(_path+"t10k-images-idx3-ubyte.gz", 10000)+0.5
        self.test_labels = extract_labels(_path+"t10k-labels-idx1-ubyte.gz", 10000)        

    @staticmethod
    def print():
        print('mnist')        
        return "mnist"