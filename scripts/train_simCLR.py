import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from contrastiv_model import simCLR, NTXent as ContrastivLoss
from generator import ColorGen, Gen, MultiGen, AdversarialGen
from regularizers import VarRegularizer, TripletCosineRegularizer, CosineDistRegularizer
from deep_models import basic_backbone, projection_mlp, color_mlp, treyer_backbone, segmentor, deconvolutor, classif_mlp
from vit_layers import Block, ViT_backbone
from schedulers import CosineDecay

import os 
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
import time


model_save = 'checkpoints_simCLR_UD_D/simCLR_regu_color_cosmos_'
iter_suffixe=""
allowed_extensions = ["UD.npz", "_D.npz"]
batch_size=256
lr = 1e-3
callbacks = [CosineDecay(lr, 1e-6, 800)]

#### PARAMS  générateur
do_color = True
do_seg = False
do_drop_band = False
do_adversarial = False

load_model = False
iter = 1

intermediate_outputs = []
color = {"do":True, "network":color_mlp(1024), "need":[0]}
segment = {"do":False, "network":segmentor(1024), "need":[0]}
reconstr = {"do":False, "network":deconvolutor(1024), "need":[0]}
adverse = {"do":False, "network":classif_mlp(1024), "need":[0], "metric":tf.keras.metrics.BinaryAccuracy()}
sup_regu = {"do":False}


model = simCLR(backbone=basic_backbone(), head=projection_mlp(1024, False),
                regularization=sup_regu, color_head=color, segmentor=segment, deconvolutor=reconstr, adversarial=adverse)


model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=ContrastivLoss())
model(np.random.random((32, 64, 64, 5)))

if load_model :
    model.load_weights("../model_save/"+model_save+str(iter*10)+iter_suffixe+".weights.h5")



data_gen = MultiGen(["/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/spec/", "/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/phot/"], 
               batch_size=batch_size, extensions=allowed_extensions, do_color=do_color, do_seg=do_seg, do_mask_band=do_drop_band)




while iter <= 1000 :
    model.fit(data_gen, epochs=10, callbacks=callbacks)  # normalement 4mn max par epoch = 400mn 
    data_gen._load_data()
    if iter % 10 == 0 :
        filename = "../model_save/"+model_save+str(iter*10)+iter_suffixe+".weights.h5"
        model.save_weights(filename)  # 6000 minutes   ==> 15 fois 100 épochs
    iter+=1
