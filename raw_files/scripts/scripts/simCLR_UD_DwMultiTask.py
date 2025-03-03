import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from contrastiv_model import simCLRcolor1, ContrastivLoss, simCLRcolor2, simCLRmultitask
from simCLR_generator import MultiGen
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

    return keras.Model(inputs=inp, outputs=[l1, flat, c4])

def segmentor(input_shape1=1024, input_shape2=(32, 32, 128)) :

    inp = keras.Input(input_shape1)
    deflat = layers.Reshape((8, 8, 16))(inp)
    c1 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(deflat)  # 8 8 256
    c1 = layers.PReLU()(c1)
    c1_r = layers.Reshape((16, 16, 64))(c1)  # 16 16 64
    c2 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(c1_r)
    c2 = layers.PReLU()(c2)
    c2_r = layers.Reshape((32, 32, 64))(c2)  #32 32 64
    c3 = layers.Conv2D(128, strides=1, kernel_size=3, padding='same')(c2_r)
    c3 = layers.PReLU()(c3)
    inp2 = keras.Input(input_shape2)
    conc = layers.Concatenate()([c3, inp2])
    c4 = layers.Conv2D(256, kernel_size=3, padding='same', strides=1)(conc)  # 32 32 256
    c4 = layers.PReLU()(c4)
    c4_r = layers.Reshape((64, 64, 64))(c4)
    segmentation = layers.Conv2D(1, kernel_size=3, padding='same', strides=1, activation='sigmoid')(c4_r)
    return keras.Model([inp, inp2], segmentation)



def deconvolutor(input_shape1=1024) :

    inp = keras.Input(input_shape1)
    l1 = layers.Dense(1024)(inp)
    l1 = layers.PReLU()(l1) 

    deflat = layers.Reshape((8, 8, 16))(inp)
    c1 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(deflat)  # 8 8 256
    c1 = layers.PReLU()(c1)
    c1 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(c1)  # 8 8 256
    c1 = layers.PReLU()(c1)
    c1_r = layers.Reshape((16, 16, 64))(c1)  # 16 16 64
    c2 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(c1_r)
    c2 = layers.PReLU()(c2)
    c2 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(c2)
    c2 = layers.PReLU()(c2)
    c2_r = layers.Reshape((32, 32, 64))(c2)  #32 32 64
    c3 = layers.Conv2D(256, strides=1, kernel_size=3, padding='same')(c2_r)
    c3 = layers.PReLU()(c3)
    c4 = layers.Conv2D(256, kernel_size=3, padding='same', strides=1)(c3)  # 32 32 256
    c4 = layers.PReLU()(c4)
    c4_r = layers.Reshape((64, 64, 64))(c4)
    c5 = layers.Conv2D(256, strides=1, kernel_size=3, padding='same')(c4_r)
    c5 = layers.PReLU()(c5)
    reconstruction = layers.Conv2D(5, kernel_size=3, padding='same', strides=1, activation='linear')(c5)
    return keras.Model(inp, reconstruction)



def mlp(input_shape=100):
    latent_input = keras.Input((input_shape))
    x = layers.Dense(512, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(5e-7), bias_regularizer=tf.keras.regularizers.l2(5e-7))(latent_input)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)
    x = layers.Dense(256, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(5e-7), bias_regularizer=tf.keras.regularizers.l2(5e-7))(x)
    return keras.Model(latent_input, x)

def regression_head(input_shape=1024, out_shape=4) :
    latent_input = keras.Input((input_shape))
    x = layers.Dense(256)(latent_input)
    x = layers.PReLU()(x)
    x = layers.Dense(256)(x)
    x = layers.PReLU()(x)
    output = layers.Dense(out_shape, activation='linear')(x)
    return keras.Model(latent_input, output)


bn=True
kind="Multi_TTTF_"

model = simCLRmultitask(backbone(bn), mlp(1024), do_color=True, color_head=regression_head(1024, 4), do_seg=True, segmentor=segmentor(),
                        do_reco=True, deconvolutor=deconvolutor())
#model = simCLRcolor2(backbone(bn), mlp(1024))
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=ContrastivLoss())
model(np.random.random((32, 64, 64, 5)))
#model.load_weights("simCLR_cosmos100.weights.h5")


data_gen = MultiGen(["/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/spec/", "/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/phot/"], do_color=True, do_seg=True, do_mask_band=True,
                     batch_size=256, extensions=["UD.npz", '_D.npz'])

iter = 1
while iter <= 100 :
    model.fit(data_gen, epochs=10)  # normalement 4mn max par epoch = 400mn 
    data_gen._load_data()
    if iter % 10 == 0 :
        filename = "../model_save/checkpoints_simCLR_UD_D/simCLR_cosmos_bn"+str(bn)+"_"+str(iter*10)+"_"+kind+".weights.h5"
        model.save_weights(filename)  # 6000 minutes   ==> 15 fois 100 épochs
    iter+=1
