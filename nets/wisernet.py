import keras
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.engine import Layer
import glob
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose, Lambda , Input , Reshape , merge, concatenate, Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard 
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.layers.core import RepeatVector, Permute
from keras.layers import MaxPooling2D,AveragePooling2D,UpSampling2D,SeparableConv2D,LeakyReLU,Concatenate
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import tensorflow
import numpy as np
from PIL import Image
import os
from keras.optimizers import SGD,Adam
import random
from skimage.color import rgb2lab,lab2rgb
from skimage.transform import resize
from skimage.measure import compare_ssim,compare_psnr
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#Dis
def srm_init(shape,dtype=None):

     
     hpf = np.zeros(shape)
     hpf[1:4,1:4,0,0] = np.array(  [[0,0,0],[1,-1,0],[0,0,0]]  ,dtype=np.float32)
     hpf[1:4,1:4,0,1] = np.array(  [[0,0,0],[0,-1,1],[0,0,0]]  ,dtype=np.float32)
     hpf[1:4,1:4,0,2] = np.array(  [[1,0,0],[0,-1,0],[0,0,0]]  ,dtype=np.float32)
     hpf[1:4,1:4,0,3] = np.array(  [[0,1,0],[0,-1,0],[0,0,0]]  ,dtype=np.float32)
     hpf[1:4,1:4,0,4] = np.array(  [[0,0,1],[0,-1,0],[0,0,0]]  ,dtype=np.float32)
     hpf[1:4,1:4,0,5] = np.array(  [[0,0,0],[0,-1,0],[1,0,0]]  ,dtype=np.float32)
     hpf[1:4,1:4,0,6] = np.array(  [[0,0,0],[0,-1,0],[0,1,0]]  ,dtype=np.float32)
     hpf[1:4,1:4,0,7] = np.array(  [[0,0,0],[0,-1,0],[0,0,1]]  ,dtype=np.float32)

     hpf[1:4,1:4,0,8] = np.array(  [[0,0,0],[1,-2,1],[0,0,0]]  ,dtype=np.float32)
     hpf[1:4,1:4,0,9] = np.array(  [[0,1,0],[0,-2,0],[0,1,0]]  ,dtype=np.float32)
     hpf[1:4,1:4,0,10] = np.array( [[1,0,0],[0,-2,0],[0,0,1]]  ,dtype=np.float32)
     hpf[1:4,1:4,0,11] = np.array( [[0,0,1],[0,-2,0],[1,0,0]]  ,dtype=np.float32)

     hpf[:,:,0,12] = np.array( [ [0,0,0,0,0],[0,0,0,0,0],[0,1,-3,3,-1],[0,0,0,0,0],[0,0,0,0,0] ]  ,dtype=np.float32)
     hpf[:,:,0,13] = np.array( [ [0,0,0,0,0],[0,1,0,0,0],[0,0,-3,0,0],[0,0,0,3,0],[0,0,0,0,-1] ]  ,dtype=np.float32)
     hpf[:,:,0,14] = np.array( [ [0,0,0,0,0],[0,0,1,0,0],[0,0,-3,0,0],[0,0,3,0,0],[0,0,-1,0,0] ]  ,dtype=np.float32)
     hpf[:,:,0,15] = np.array( [ [0,0,0,0,0],[0,0,0,1,0],[0,0,-3,0,0],[0,3,0,0,0],[-1,0,0,0,0] ]  ,dtype=np.float32)
     hpf[:,:,0,16] = np.array( [ [0,0,0,0,0],[0,0,0,0,0],[-1,3,-3,1,0],[0,0,0,0,0],[0,0,0,0,0] ]  ,dtype=np.float32)
     hpf[:,:,0,17] = np.array( [ [-1,0,0,0,0],[0,3,0,0,0],[0,0,-3,0,0],[0,0,0,1,0],[0,0,0,0,0] ]  ,dtype=np.float32)
     hpf[:,:,0,18] = np.array( [ [0,0,-1,0,0],[0,0,3,0,0],[0,0,-3,0,0],[0,0,1,0,0],[0,0,0,0,0] ]  ,dtype=np.float32)
     hpf[:,:,0,19] = np.array( [ [0,0,0,0,-1],[0,0,0,3,0],[0,0,-3,0,0],[0,1,0,0,0],[0,0,0,0,0] ]  ,dtype=np.float32)

     hpf[1:4,1:4,0,20] = np.array( [[-1,2,-1],[2,-4,2],[-1,2,-1]]  ,dtype=np.float32)

     hpf[:,:,0,21] = np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]],dtype=np.float32)

     hpf[1:4,1:4,0,22] = np.array( [[-1,2,-1],[2,-4,2],[0,0,0]]  ,dtype=np.float32)
     hpf[1:4,1:4,0,23] = np.array( [[-1,2,0],[2,-4,0],[-1,2,0]]  ,dtype=np.float32)
     hpf[1:4,1:4,0,24] = np.array( [[0,0,0],[2,-4,2],[-1,2,-1]]  ,dtype=np.float32)
     hpf[1:4,1:4,0,25] = np.array( [[0,2,-1],[0,-4,2],[0,2,-1]]  ,dtype=np.float32)


     hpf[:,:,0,26] = np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[0,0,0,0,0],[0,0,0,0,0]],dtype=np.float32)
     hpf[:,:,0,27] = np.array([[0,0,0,0,0],[0,0,0,0,0],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]],dtype=np.float32)
     hpf[:,:,0,28] = np.array([[0,0,-2,2,-1],[0,0,8,-6,2],[0,0,-12,8,-2],[0,0,8,-6,2],[0,0,-2,2,-1]],dtype=np.float32)
     hpf[:,:,0,29] = np.array([[-1,2,-2,0,0],[2,-6,8,0,0],[-2,8,-12,0,0],[2,-6,8,0,0],[-1,2,-2,0,0]],dtype=np.float32)
     

     return hpf
     

    
    

def D():
    n = 9
    rgb = Input(shape=(256,256,3))

    red_input = Lambda(lambda x:x[:,:,:,:1])(rgb)
    x_red = Conv2D(filters=30,kernel_size=(5,5),strides=1,kernel_initializer=srm_init,padding='same',input_shape=(256,256,1))(red_input)
    x_red.trainable = False	

    green_input = Lambda(lambda x:x[:,:,:,1:2])(rgb)
    x_green = Conv2D(filters=30,kernel_size=(5,5),strides=1,kernel_initializer=srm_init,padding='same',input_shape=(256,256,1))(green_input)
    x_green.trainable = False		

    blue_input = Lambda(lambda x:x[:,:,:,2:])(rgb)
    x_blue = Conv2D(filters=30,kernel_size=(5,5),strides=1,kernel_initializer=srm_init,padding='same',input_shape=(256,256,1))(blue_input)
    x_blue.trainable = False		
    
    #my_concat = Lambda(lambda x: K.concatenate([x[0],x[1],x[2]]))
    #x = my_concat([x_red,x_green,x_blue])
    x = keras.layers.concatenate([x_red,x_green,x_blue])
    x = Conv2D(filters=8*n,kernel_size=(5,5),strides=2,padding='same')(x)
    x = Lambda(lambda x:abs(x))(x)
    x = BatchNormalization(axis=-1,scale=False)(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=(5,5),strides=2)(x)

    x = Conv2D(filters=32*n,kernel_size=(3,3),strides=1,padding='same')(x)
    x = BatchNormalization(axis=-1,scale=False)(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=(5,5),strides=4,padding='same')(x)

    x = Conv2D(filters=128*n,kernel_size=(3,3),strides=1,padding='same')(x)
    x = BatchNormalization(axis=-1,scale=False)(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=(16,16),strides=32,padding='valid')(x)
    x = Flatten()(x)


    x = Dense(800)(x)
    x = Activation('relu')(x)
    x = Dense(400)(x)
    x = Activation('relu')(x)
    x = Dense(200)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    
    output = Activation('linear')(x)
    
    model = Model(inputs=[rgb],outputs=[x])
    print "++++++DECTOR++++++++"
    print(model.summary())
    print '++++++DECTOR++++++++'
    #plot_model(model, to_file='model.png',show_shapes=True)
    sgd = SGD(lr=0.01, momentum=0.9, decay=0.001,nesterov=True)

    #model = multi_gpu_model(model,4)
    
    model.compile(loss='mse',#'binary_crossentropy',
                      optimizer='sgd',
                      metrics=['accuracy'])
    
    return model

 

