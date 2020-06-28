## source : https://github.com/geifmany/cifar-vgg
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
import setup_data
import tensorflow as tf
import os

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
def recon_lambda(x, means, convs, divider):
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
    
    ####################### reconstruction #######################
    out = tf.pow(2.0, out*125)
    out = tf.nn.relu(out)+1e-10
    out = out / tf.expand_dims(tf.reduce_sum(out, [3]), axis=-1)
    input_shape = tf.shape(x)
    out = tf.nn.conv2d_transpose(out, means, output_shape= input_shape, strides=(1,1,1,1), padding="VALID")
    
    out = tf.transpose(out, (1,2,3,0))
    out = out/divider
    out = tf.transpose(out, (3,0,1,2))   
    return out

#%%
class cifar10vgg:
    def __init__(self):
        self.num_classes = 10
        self.weight_decay = 0.0005
        self.x_shape = [32,32,3]
        self.model = self.build_model()
        
#%%
    def build_model(self, layer_path = './rbf_layer/'):
        model = Sequential()
        weight_decay = self.weight_decay

        means, convs = loadParam(layerDir=layer_path)
        _args = {'means':means, 'convs': convs, 'divider': get_divider(data.train_data.shape[1])}

        model.add(Lambda(recon_lambda, arguments=_args,  output_shape=data.train_data.shape[1:], input_shape=data.train_data.shape[1:]))    
        model.add(Conv2D(64, (3, 3), padding='same', input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))

        model.compile(loss=fn, optimizer='sgd', metrics=['accuracy'])
        model.summary()
        return model

#%%
    def train(self, model, data, model_path, existing_path = None):
        
        label_smooth = 0.1
        y_train = data.train_labels.clip(label_smooth / 9., 1. - label_smooth)

        #training parameters
        batch_size = 128
        maxepoches = 50
        learning_rate = 0.1
        lr_decay = 1e-6
        lr_drop = 10

        def lr_scheduler(epoch):
            lr = learning_rate * (0.5 ** (epoch // lr_drop))
            if lr < 0.000001:
                lr = 0.000001
            return lr
        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        #data augmentation
        datagen = ImageDataGenerator(horizontal_flip=True)
        datagen.fit(data.train_data)

        model_checkpoint= keras.callbacks.ModelCheckpoint(
                model_path, monitor="val_acc", save_best_only=False,
                save_weights_only=True, verbose=1)

        #optimization details
        if existing_path !=None:
            model.load_weights(existing_path)
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss=fn, optimizer=sgd, metrics=['accuracy'])

        print(model.evaluate(data.test_data, data.test_labels))

        model.fit_generator(datagen.flow(data.train_data, y_train, batch_size=batch_size),
                            steps_per_epoch=int(np.ceil(data.train_data.shape[0] / float(batch_size))),
                            epochs=maxepoches,
                            validation_data=(data.test_data, data.test_labels), 
                            callbacks=[reduce_lr, model_checkpoint], 
                            verbose=1)

        return model

def fn(correct, predicted):
    return tf.nn.softmax_cross_entropy_with_logits(labels=correct, logits=predicted)

if __name__ == '__main__':
    data = setup_data.Cifar()
    print(data.train_data.shape, data.test_data.shape)
    model = cifar10vgg()
    model_path = './trained_models/rCNN_2'
    vgg = model.train(model= model.model, data= data, model_path= model_path, existing_path= None)
