import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from contrastiv_model import simCLRcolor1, ContrastivLoss, simCLRcolor2
from simCLR_generator import ColorGen
from regularizers import VarRegularizer, TripletCosineRegularizer, CosineDistRegularizer
import os 
from vit_layers import Block, ViT_backbone
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
import time



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
kind="ViTback_ColorHead"

model = simCLRcolor1(ViT_backbone(embed_dim=256, num_blocks=4, num_heads=8, gp='average'), mlp(1024), color_head(1024), regularizer=None)
#model = simCLRcolor1(backbone(bn), mlp(1024), color_head(1024), regularizer=TripletCosineRegularizer())
#model = simCLRcolor1(backbone(bn), mlp(1024), color_head(1024), regularizer=CosineDistRegularizer())
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
