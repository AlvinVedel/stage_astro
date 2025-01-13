import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from contrastiv_model import simCLRcolor1, ContrastivLoss, simCLRcolor2
from simCLR_generator import ColorGen
from regularizers import VarRegularizer, TripletCosineRegularizer, CosineDistRegularizer
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
import time


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

def color_head(input_shape=1024) :
    latent_input = keras.Input((input_shape))
    x = layers.Dense(256)(latent_input)
    x = layers.PReLU()(x)
    x = layers.Dense(256)(x)
    x = layers.PReLU()(x)
    output = layers.Dense(4, activation='linear')(x)
    return keras.Model(latent_input, output)


bn=True
kind="ColorHead_CosineDistRegu"

#model = simCLRcolor1(backbone(bn), mlp(1024), color_head(1024), regularizer=VarRegularizer())
#model = simCLRcolor1(backbone(bn), mlp(1024), color_head(1024), regularizer=TripletCosineRegularizer())
model = simCLRcolor1(backbone(bn), mlp(1024), color_head(1024), regularizer=CosineDistRegularizer())
#model = simCLRcolor2(backbone(bn), mlp(1024))
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=ContrastivLoss())
model(np.random.random((32, 64, 64, 5)))
#model.load_weights("../model_save/checkpoints_simCLR_UD_D/simCLR_cosmos_bnTrue_400_ColorHead.weights.h5")


data_gen = ColorGen(["/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/spec/", "/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/phot/"], batch_size=256, extensions=["UD.npz", '_D.npz'])

iter = 1
while iter <= 100 :
    model.fit(data_gen, epochs=10)  # normalement 4mn max par epoch = 400mn 
    data_gen._load_data()
    if iter % 10 == 0 :
        filename = "../model_save/checkpoints_simCLR_UD_D/simCLR_cosmos_bn"+str(bn)+"_"+str(iter*10)+"_"+kind+".weights.h5"
        model.save_weights(filename)  # 6000 minutes   ==> 15 fois 100 épochs
    iter+=1
