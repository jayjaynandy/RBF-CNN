import setup_data
import tensorflow as tf
import numpy as np
from keras.layers import Lambda, Conv2D, MaxPooling2D, Activation, Dense
from keras.layers import Flatten, BatchNormalization
from keras.models import Sequential
from keras.utils import get_custom_objects
import os

os.environ['PYTHONHASHSEED'] = '0'
tf.set_random_seed(9999)
np.random.seed(9999)


'''Loading the RBF filter weights.'''
def loadParam(layerDir):
    means = np.load(layerDir + "mean0.npy")
    means = np.transpose(means, axes= (1,2,3,0))
    convs = np.load(layerDir + "conv0.npy")
    
    means = means.astype(np.float32)
    convs = convs.astype(np.float32)
    return means, convs 

''' 
Output to be used in the reconstruction layer for stitching 
the patches of the reconstructed images.
'''
def get_divider(_shape):
    tf_inp_ones = tf.ones((1, _shape-2, _shape-2, 1))
    tf_ones_2 = tf.ones((3,3,1,1)) 
    out2 = tf.nn.conv2d_transpose(tf_inp_ones, tf_ones_2, output_shape= [1, _shape, _shape,1], strides=(1,1,1,1), padding="VALID")
    sess = tf.Session()

    with sess.as_default():
        np_divider = out2.eval(feed_dict={})
        np_divider = np.transpose(np_divider, (1,2,3,0))
        return np_divider

''' RBF layer and Reconstruction layer ''' 
def recon_lambda(x, means, convs, divider, p, randMat):    
    ch = means.shape[2]
    tf_ones = tf.ones((3,3,ch,1))    
    tensor1 = tf.nn.conv2d(x, means, strides=(1,1,1,1), padding="VALID")
    tensor2 = tf.multiply(x, x, name=None)
    tensor2 = tf.nn.conv2d(tensor2, tf_ones, strides=(1,1,1,1), padding="VALID")    
    tensor3 = tf.multiply(means, means, name=None)
    tensor3 = tf.reduce_sum(tensor3, [0,1,2]) 
    out = 2*tensor1 - tensor2 - tensor3
    out = out/(2*convs) - tf.log(convs)*0.5
    out = tf.nn.sigmoid(out)
    
    '''Reconstruction Layer'''
    out = tf.pow(2.0, out*p)
    out = tf.nn.relu(out)+1e-10
    out = out / tf.expand_dims(tf.reduce_sum(out, [3]), axis=-1)
    input_shape = tf.shape(x)
    out = tf.nn.conv2d_transpose(out, means+randMat, output_shape= input_shape, strides=(1,1,1,1), padding="VALID")
    
    out = tf.transpose(out, (1,2,3,0))
    out = out/divider
    out = tf.transpose(out, (3,0,1,2))
    out = tf.clip_by_value(out, 0, 1)
    return out


def get_classmodel(data, model_path, layer_path):
    
    means, convs = loadParam(layerDir=layer_path)
    shp = means.shape

    randMat = np.random.normal(0, 1.5,[shp[0],shp[1],shp[2],shp[3]]).astype(np.float32)
    randMat = 0
    randMat = randMat*convs

    _args = {'means':means, 'convs': convs, 
             'divider': get_divider(data.train_data.shape[1]),  'p': 25,
             'randMat': randMat}

    model = Sequential()
    model.add(Lambda(recon_lambda, arguments=_args,  output_shape=data.train_data.shape[1:], input_shape=data.train_data.shape[1:]))    
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=data.train_data.shape[1:], activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(10))
    model.add(Activation("softmax"))
    
    model.load_weights(model_path)

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])    
    return model


def get_normal_acc():
    recovered = get_acc(data.test_data, data.test_labels)*10000
    recovered_acc = recovered/data.test_data.shape[0]
    print('   recovered : ', recovered, '  recovered_acc : ', recovered_acc )


def get_attack_acc(fileName):
    dataList = np.load(fileName)
    test_data = dataList['x_test']
    test_labels = dataList['y_test']
    
    recovered = get_acc(test_data, test_labels)

    print(fileName, '   recovered : ', recovered)

def get_acc(npData, npLabel):
    iter = len(classModelList)
    data_size = npData.shape[0]
    final_preds = np.zeros([data_size, num_class])
    for k in range(iter):
        final_preds += classModelList[k].predict(npData)
        
    acc = np.mean(np.equal(np.argmax(npLabel, axis=1), np.argmax(final_preds, axis=1)))
    return acc    

def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct, logits=predicted)
get_custom_objects().update({"fn": fn})


if __name__ == '__main__': 

    data = setup_data.Cifar()
    data_name = data.print()

    iter = 10
    num_class = 10
    classModelList = []
    for i in range(iter):
        print(i)
        classModelList.append(get_classmodel(data, './trained_models/rCNN', 
                                             './rbf_layer/'))

    get_normal_acc()

    ''' 
    Provide Appropriate attack path with the following format.
    dataList = np.load(fileName)
    test_data = dataList['x_test']
    test_labels = dataList['y_test']
    
    '''
    adv_path = './adv_images/adv.npz'  # example path
    get_attack_acc(adv_path)
