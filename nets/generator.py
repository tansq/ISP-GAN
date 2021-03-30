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

def conv_stack(data, filters, s=1,short_cut=False):
    output = Conv2D(filters, (3, 3), strides=s, activation='relu', padding='same')(data)
    output = Conv2D(filters, (3, 3), strides=1, activation='relu', padding='same')(output)
    if short_cut:
        data =  Conv2D(filters, (3, 3), strides=s, activation='relu', padding='same')(data)
    output = keras.layers.add([output,data])
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    return output
def deconv(data, filters, s=1):
    output = Lambda(lambda x: K.resize_images(x,2,2,"channels_last"))(data)

    #output = Lambda(lambda x:tf.image.resize_image_with_pad(x,h,h))(data)
    output = Conv2D(filters, (3,3), strides=s, activation='relu', padding='same')(output)
    return output

def deconv_stack(data, filters, s=1,short_cut=True):
    output = Conv2D(filters, (3, 3), strides=s, activation='relu', padding='same')(data)
    output = Conv2D(filters, (3, 3), strides=1, activation='relu', padding='same')(output)
    if short_cut:
        data =  Conv2D(filters, (3, 3), strides=s, activation='relu', padding='same')(data)
    output = keras.layers.concatenate([output,data],axis=-1)
    #output = BatchNormalization()(output)
    output = Activation('relu')(output)
    return output
    
def deconv_sr(x,filters,s=1):
    x = deconv_stack(x,filters,short_cut=True)
    out = Lambda(lambda x: tf.depth_to_space(x, 2))(x)
    return out
def G():

    #Encoder
    encoder_input = Input(shape=(256, 256, 1,))
    encoder_output0 = conv_stack(encoder_input, 16, 1,short_cut=True)
    encoder_output1 = conv_stack(encoder_output0, 32, 1,short_cut=True)
    encoder_output2 = conv_stack(encoder_output1, 64, 2,short_cut=True)
    encoder_output3 = conv_stack(encoder_output2, 128, 1,short_cut=True)
    encoder_output4 = conv_stack(encoder_output3, 128, 2,short_cut=True)
    encoder_output5 = conv_stack(encoder_output4, 256, 1,short_cut=True)
    encoder_output6 = conv_stack(encoder_output5, 256, 2,short_cut=True)
    encoder_output7 = conv_stack(encoder_output6, 512, 1,short_cut=True)
    encoder_output8 = conv_stack(encoder_output7, 512, 1)

    encoder_output9 = conv_stack(encoder_output8, 256, 1,short_cut=True)

#Decoder
    decoder_output0 = conv_stack(encoder_output9, 128, 1,short_cut=True)
    decoder_output1 = deconv_sr(decoder_output0,128)
    decoder_output1 = Concatenate()([decoder_output1,encoder_output4])
    decoder_output2 = conv_stack(decoder_output1, 64, 1,short_cut=True)
    decoder_output3 = deconv_sr(decoder_output2,64)
    decoder_output3 = Concatenate()([decoder_output3,encoder_output2])
    decoder_output4 = conv_stack(decoder_output3, 32, 1,short_cut=True)
    decoder_output5 = conv_stack(decoder_output4, 16, 1,short_cut=True)
    decoder_output6 = deconv_sr(decoder_output5,16)
    decoder_output6 = Concatenate()([decoder_output6,encoder_output0])
    decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output6)


    model = Model(inputs=[encoder_input], outputs=decoder_output)
    print model.summary()

    return model
    



