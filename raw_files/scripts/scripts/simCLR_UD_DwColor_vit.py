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
    x = layers.Dense(512, activation='linear')(latent_input)
    x = layers.PReLU()(x)
    x = layers.Dense(256, activation='linear', activity_regularizer=tf.keras.regularizers.L1L2(l1=1e-3, l2=1e-2))(x)
    return keras.Model(latent_input, x)



def color_head(input_shape=1024) :
    latent_input = keras.Input((input_shape))
    x = layers.Dense(256)(latent_input)
    x = layers.PReLU()(x)
    x = layers.Dense(256)(x)
    x = layers.PReLU()(x)
    output = layers.Dense(4, activation='linear')(x)
    return keras.Model(latent_input, output)


class LearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr, target_lr, max_epochs, ep_counter = 0, decay_factor=1.0):
        super().__init__()
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self.max_epochs = max_epochs
        self.decay_factor = decay_factor
        self.epoch_counter = ep_counter  # Compteur pour le nombre d'époques

    def cosine_lr_schedule(self):
        """Calcul de la décroissance cosinus du learning rate"""
        epoch = self.epoch_counter
        decayed_lr = self.initial_lr + (self.target_lr - self.initial_lr)*(0.5 * (1 + np.cos(np.pi * epoch / self.max_epochs)))
        return decayed_lr

    def on_epoch_begin(self, epoch, logs=None):
        """Mise à jour du learning rate au début de chaque époque"""
        self.epoch_counter += 1
        current_lr = self.model.optimizer.learning_rate - self.cosine_lr_schedule()
        tf.keras.backend.set_value(self.model.optimizer.lr, current_lr)
        print(f"\nEpoch {self.epoch_counter}: Learning rate is set to {current_lr}")

bn=True
kind="ViTback_ColorHead"

model = simCLRcolor1(ViT_backbone(embed_dim=256, num_blocks=4, num_heads=8, patch_size=4, gp='average'), mlp(256), color_head(256), regularizer=None)
#model = simCLRcolor1(backbone(bn), mlp(1024), color_head(1024), regularizer=TripletCosineRegularizer())
#model = simCLRcolor1(backbone(bn), mlp(1024), color_head(1024), regularizer=CosineDistRegularizer())
#model = simCLRcolor2(backbone(bn), mlp(1024))
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=ContrastivLoss())
model(np.random.random((32, 64, 64, 5)))
#model.load_weights("../model_save/checkpoints_simCLR_UD_D/simCLR_cosmos_bnTrue_100_ViTback_ColorHead.weights.h5")

lr = LearningRateScheduler(1e-3, 1e-6, 1000, ep_counter=100)

data_gen = ColorGen(["/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/spec/", "/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/phot/"], batch_size=128, extensions=["UD.npz", '_D.npz'])

iter = 1
while iter <= 100 :
    model.fit(data_gen, epochs=10, callbacks=[lr])  # normalement 4mn max par epoch = 400mn 
    data_gen._load_data()
    if iter % 10 == 0 :
        filename = "../model_save/checkpoints_simCLR_UD_D/simCLR_cosmos_bn"+str(bn)+"_"+str(iter*10)+"_"+kind+".weights.h5"
        model.save_weights(filename)  # 6000 minutes   ==> 15 fois 100 épochs
    iter+=1
