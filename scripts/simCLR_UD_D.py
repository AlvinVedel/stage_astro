import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from contrastiv_model import simCLR, ContrastivLoss
from simCLR_generator import Gen
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
import time

def inception_block(input):
    c1 = layers.Conv2D(101, activation='relu', kernel_size=1, strides=1, padding='same')(input)
    c2 = layers.Conv2D(101, activation='relu', kernel_size=1, strides=1, padding='same')(input)
    c3 = layers.Conv2D(101, activation='relu', kernel_size=1, strides=1, padding='same')(input)

    c4 = layers.Conv2D(156, activation='relu', kernel_size=3, strides=1, padding='same')(c1)
    c5 = layers.Conv2D(156, activation='relu', kernel_size=5, strides=1, padding='same')(c2)
    c6 = layers.AveragePooling2D((2, 2), strides=1, padding='same')(c3)

    c7 = layers.Conv2D(109, activation='relu', kernel_size=1, strides=1, padding='same')(input)

    conc = layers.Concatenate(axis=-1)([c4, c5, c6, c7])   # 156 + 156 + 101 + 109 = 522
    return conc

def treyer_backbone(bn=True) :

    input_img = keras.Input((64, 64, 5))
    #input_ebv = keras.Input((1,))
    conv1 = layers.Conv2D(96, kernel_size=5, activation='relu', strides=1, padding='same', name='c1')(input_img)
    conv2 = layers.Conv2D(96, kernel_size=3, activation='tanh', strides=1, padding='same', name='c2')(conv1)
    avg_pool = layers.AveragePooling2D(pool_size=(2, 2), strides=2, name='p1')(conv2)  # batch, 32, 32, 96

    incep1 = inception_block(avg_pool)
    incep2 = inception_block(incep1)
    incep3 = inception_block(incep2)

    avg_pool2 = layers.AveragePooling2D((2, 2), strides=2, name='p2')(incep3) # 16, 16, 522

    incep4 = inception_block(avg_pool2)
    incep5 = inception_block(incep4)

    avg_pool3 = layers.AveragePooling2D((2, 2), strides=2, name='p3')(incep5)  # 8, 8, 522

    incep6 = inception_block(avg_pool3) 

    conv3 = layers.Conv2D(96, kernel_size=3, activation='relu', padding='valid', strides=1, name='c3')(incep6)  # 6, 6, 96
    conv4 = layers.Conv2D(96, kernel_size=3, activation='relu', padding='valid', strides=1, name='c4')(conv3)  # 4, 4, 96
    conv5 = layers.Conv2D(96, kernel_size=3, activation='relu', padding='valid', strides=1, name='c5')(conv4)   # 2, 2, 96
    avg_pool4 = layers.AveragePooling2D((2, 2), strides=1, name='p4')(conv5)   # 1, 1, 96
    resh = layers.Reshape(target_shape=(96,))(avg_pool4) # batch, 96
    
    #conc = layers.Concatenate()([resh, input_ebv])
    l1 = layers.Dense(1024, activation='relu', name='l1')(resh)
    if bn :
        l1 = layers.BatchNormalization()(l1)

    
    model = keras.Model(inputs=[input_img], outputs=[l1])
    return model

def backbone(bn=True) :
    inp = keras.Input((64, 64, 5))
    c1 = layers.Conv2D(96, padding='same', strides=1, kernel_size=3)(inp) # 64
    c1 = layers.PReLU()(c1) 
    c2 = layers.Conv2D(96, padding='same', kernel_size=3, strides=1, activation='tanh')(c1)  #64
    p1 = layers.AveragePooling2D((2, 2))(c2)  # 32
    c3 = layers.Conv2D(128, padding='same', strides=1, kernel_size=3)(p1)
    c3 = layers.PReLU()(c3)
    c4 = layers.Conv2D(128, padding='same', kernel_size=3, strides=1)(c3)  #32
    c4 = layers.PReLU(name='c4')(c4) 
    p2 = layers.AveragePooling2D((2, 2))(c4)  # 16
    c5 = layers.Conv2D(256, padding='same', strides=1, kernel_size=3)(p2) #16
    c5 = layers.PReLU()(c5)
    c6 = layers.Conv2D(256, padding='same', kernel_size=3, strides=1)(c5)  #16
    c6 = layers.PReLU()(c6)
    p3 = layers.AveragePooling2D((2, 2))(c6) # 8
    c7 = layers.Conv2D(256, kernel_size=3, strides=1, padding='valid')(p3) # 6
    c7 = layers.PReLU()(c7)
    c8 = layers.Conv2D(256, kernel_size=3, strides=1, padding='valid')(c7) # 4
    c8 = layers.PReLU()(c8)
    c9 = layers.Conv2D(256, padding='valid', kernel_size=3, strides=1)(c8) # 2, 2, 256
    c9 = layers.PReLU()(c9)
    
    flat = layers.Flatten(name='flatten')(c9) # 2, 2, 256 = 1024 

    l1 = layers.Dense(1024)(flat) 
    l1 = layers.PReLU()(l1)
    if bn :
        l1 = layers.BatchNormalization()(l1)

    return keras.Model(inputs=inp, outputs=l1)


def mlp(input_shape=100):
    latent_input = keras.Input((input_shape))
    x = layers.Dense(512, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(5e-7), bias_regularizer=tf.keras.regularizers.l2(5e-7))(latent_input)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)
    x = layers.Dense(256, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(5e-7), bias_regularizer=tf.keras.regularizers.l2(5e-7))(x)
    return keras.Model(latent_input, x)


bn=True

model = simCLR(treyer_backbone(bn=False), mlp(1024))
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=ContrastivLoss())
model(np.random.random((32, 64, 64, 5)))
#model.load_weights("simCLR_cosmos100.weights.h5")

batch_size=256
data_gen = Gen(["/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/spec/", "/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/phot/"], 
               batch_size=batch_size, extensions=["UD.npz", "_D.npz"])

iter = 1
while iter <= 1000 :
    model.fit(data_gen, epochs=10)  # normalement 4mn max par epoch = 400mn 
    data_gen._load_data()
    if iter % 10 == 0 :
        filename = "../model_save/checkpoints_simCLR_UD_D/simCLR_treyerback_cosmos_bn"+str(bn)+"_"+str(iter*10)+".weights.h5"
        model.save_weights(filename)  # 6000 minutes   ==> 15 fois 100 épochs
    iter+=1
