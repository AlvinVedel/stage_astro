import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from models.contrastiv_model import simCLR, NTXent as ContrastivLoss, simCLRcolor1
from generators.generator import MultiGen
from utils.regularizers import VarRegularizer, TripletCosineRegularizer, CosineDistRegularizer
from models.deep_models import basic_backbone, projection_mlp, color_mlp, treyer_backbone, segmentor, deconvolutor, classif_mlp, noregu_projection_mlp
from vit_layers import Block, ViT_backbone
from utils.schedulers import CosineDecay, LinearDecay
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
import time


"""

ENTRAINEMENT DE SIMCLR : A Simple Framework for Contrastive Learning of Visual Representations https://arxiv.org/abs/2002.05709



"""



model_save = 'checkpoints_new_simCLR/simCLR_UD_D_norm'
iter_suffixe="_ColorHead_NotRegularized_fullBN_minus1"
allowed_extensions = ["UD.npz", "_D.npz"]
batch_size=256
lr = 1e-4
callbacks = [LinearDecay(0, 2, 40)]

#### PARAMS  générateur
do_color = True
do_seg = False
do_drop_band = False
do_adversarial = False

load_model = False
iter = 0

#intermediate_outputs = []
#color = {"do":True, "network":color_mlp(1024), "need":[0], "weight":1}
#segment = {"do":False, "network":segmentor(1024), "need":[0], "weight":1}
#reconstr = {"do":False, "network":deconvolutor(1024), "need":[0], "weight":1}
#adverse = {"do":False, "network":classif_mlp(1024), "need":[0], "weight":1, "metric":tf.keras.metrics.BinaryAccuracy()}
#sup_regu = {"do":False, "weight":0.1}


#model = simCLR(backbone=basic_backbone(), head=projection_mlp(1024, False),
#                regularization=sup_regu, color_head=color, segmentor=segment, deconvolutor=reconstr, adversarial=adverse)
#model = simCLRcolor1(basic_backbone(), projection_mlp(1024, False), color_mlp(1024))
model = simCLRcolor1(basic_backbone(full_bn=True, all_bn=False), noregu_projection_mlp(1024, True), color_mlp(1024))
model.compile(optimizer=keras.optimizers.Adam(lr), loss=ContrastivLoss(normalize=True))
model(np.random.random((32, 64, 64, 6)))





if load_model :
    model.load_weights("../model_save/"+model_save+str(iter*10)+iter_suffixe+".weights.h5")



data_gen = MultiGen(["/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/cleaned_spec/", "/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/cleaned_phot/"], 
               batch_size=batch_size, extensions=allowed_extensions, do_color=do_color, do_seg=do_seg, do_mask_band=do_drop_band)


while iter <= 1000 :
    iter+=1
    model.fit(data_gen, epochs=10, callbacks=callbacks)  # normalement 4mn max par epoch = 400mn 
    data_gen._load_data()

    if iter % 5 == 0 :
        filename = "../model_save/"+model_save+str(iter*10)+iter_suffixe+".weights.h5"
        model.save_weights(filename)  # 6000 minutes   ==> 15 fois 100 épochs
