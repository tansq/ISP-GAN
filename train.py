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
from nets.generator import G
from nets.wisernet import D
from utils import evaluate_quality
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    
def GAN(g,d):
   
    L_input = Input(shape=(256,256,1),name='gan_input_2')
    tf.get_variable_scope().reuse_variables()
    pre_ab = g([L_input])
    d.trainable = False    
    #concatenate
    X = Lambda(lambda x:K.concatenate([x[0],x[1]*128],axis=-1))([L_input,pre_ab])
    out = d(X)
    
    
    model = Model(inputs=[L_input],outputs=[out,pre_ab])
    print(model.summary()) 
    optimizer = Adam(0.0002, 0.5)    
    model.compile(loss='mse',
                  loss_weights=[1,1000],
					  optimizer=optimizer,
                     metrics=['acc'])
    print model.metrics_names
    
    return model
     
     


        
    
if __name__=='__main__':
    epoch_num = 7000
    
    batch_size = 30

    
    gen = G()
    dis = D()

    gan = GAN(gen,dis)
    x_train = []
    
    train_file = glob.glob('') 
                 
    print len(train_file)

    x_val = []
    val_origin = []
    val_file = glob.glob('')
    j=0
    standard_size = (256,256,3)
    for f in val_file:
        j+=1
        print j,f
        try:
            tmp = np.array(Image.open(f),dtype=np.float32)/255.
            #print tmp.shape
            if tmp.shape!= standard_size:
                tmp = resize(tmp,standard_size)     
            x_val.append(rgb2lab(tmp))
            val_origin.append(tmp*255.)
        except: print 'error'
    x_val = np.array(x_val,dtype=np.float32)    
    val_origin = np.array(val_origin,dtype=np.float32)
    i = 0
    
    batch_num = int(len(train_file)/batch_size)
    

    print '\n success load data....'	
    file = open('record.log','w')

    best_psnr = 0
    best_ssim = 0
    valid_val = np.ones((x_val.shape[0],1))
    for epoch in range(epoch_num):
        print 'Epoch :',epoch
        file.write('=========epoch======'+str(epoch)+'\n')

        for i in range(batch_num):
            #print 'process image...'
            origin = []
            for f in train_file[i*batch_size:(i+1)*batch_size]:
                try:
                    tmp = np.array(Image.open(f),dtype=np.float32)/255.
                    if tmp.shape!= standard_size:
                        tmp = resize(tmp,standard_size) 
                    origin.append(rgb2lab(tmp))
                except:  print f
            origin = np.array(origin,dtype=np.float32)
            valid = np.ones((origin.shape[0],1))
            fake = np.zeros((origin.shape[0],1))            
            origin_L = origin[:,:,:,:1]


            gen_ab = gen.predict([origin_L],verbose=0)
            #print gen_ab
            gen_ab *= 128
            lab_out = np.concatenate((origin_L,gen_ab),axis=-1)
            
            d_loss1 = dis.train_on_batch([origin],valid)            
            d_loss2 = dis.train_on_batch([lab_out],fake)
            d_loss = (d_loss1+d_loss2)
            
            #Y = np_utils.to_categorical(np.array(Y), 2)
            #d_loss[0] means acc
            s = 'epoch:'+str(epoch)+'  batch '+str(i)+'  d_loss :  '+str(d_loss)
            print s
            file.write(s+'\n')

            #fake label

            gan_loss = gan.train_on_batch([origin_L],[valid,origin[:,:,:,1:]/128])
            s = 'gan_loss : '+str(gan_loss)
            print s
            file.write(s+'\n')

            

        save_flag = False    
        result = gen.predict([x_val[:,:,:,:1]],verbose=1)
        result *= 128
        result = np.concatenate((x_val[:,:,:,:1],result),axis=-1)
        psnr,ssim = evaluate_quality(val_origin,result)
        print (psnr,ssim)
        if psnr>best_psnr or ssim>best_ssim:
            print ('============save==============')
            save_flag = True
            if psnr>best_psnr:
                best_psnr = psnr
            if ssim>best_ssim:
                best_ssim = ssim
        
        if save_flag:
            file.close()

            gen.save('model/'+str(epoch)+'psnr:'+str(psnr)+'ssim:'+str(ssim)+'.h5') 
            #dis.save('model/'+str(epoch)+'dis'+str(d_loss)+'.hdf5')
            file = open('record.log','r+')
            file.readlines()
    file.close()



