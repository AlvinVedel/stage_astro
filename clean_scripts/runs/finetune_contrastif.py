
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from models.contrastiv_model import simCLR, simCLRcolor1, simCLRcolor1_adversarial, simCLR1
from models.deep_models import basic_backbone, projection_mlp, color_mlp, segmentor, deconvolutor, astro_head, astro_model, classif_mlp, AstroFinetune, ContrastivAstroFinetune
import os 
from vit_layers import ViT_backbone
from generators.generator import SupervisedGenerator
from utils.schedulers import TreyerScheduler
from utils.astro_metrics import Bias, SigmaMAD, OutlierFraction
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import time

base_path = "/lustre/fswork/projects/rech/dnz/ull82ct/astro/"


"""
SIMPLE SCRIPT DE FINETUNING SSL AVEC PERTE CONTRASTIVE

"""



model = simCLRcolor1(basic_backbone(full_bn=True), projection_mlp(1024, True), color_mlp(1024))
#model = simCLRcolor1(basic_backbone(full_bn=True), projection_mlp(1024, True), color_mlp(1024))
#model = simCLRcolor1_adversarial(basic_backbone(full_bn=True), projection_mlp(1024, True), color_mlp(1024), classif_mlp())
#model = simCLRcolor1(ViT_backbone(), projection_mlp(256), color_mlp(256))
model(np.random.random((32, 64, 64, 6)))



load_w_path = "model_save/checkpoints_new_simCLR/simCLR_UD_D_"
model_name = "norm300_Color_NotRegularized_fullBN_minus1"
save_w_path = "model_save/checkpoints_simCLR_finetune/simCLR_finetune_UD_D_"
plots_path = "plots/simCLR/simCLR_finetune/"


n_epochs = 50

for base in ["base1", "base2", "base3"] :

    data_gen = SupervisedGenerator("/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/finetune/"+base+".npz", batch_size=32, nbins=400, contrast=True)

    
    model(np.random.random((32, 64, 64, 6)))
    model.load_weights(base_path+load_w_path+model_name+".weights.h5")

    extracteur = model.backbone
    head = model.head
    
    model1 = ContrastivAstroFinetune(back=extracteur, head=astro_head(1024, 400), projection_head=head, train_back=False)
    model1.compile(optimizer=keras.optimizers.Adam(1e-4))

    history = model1.fit(data_gen, epochs=5, callbacks=[TreyerScheduler()])

    
    model1.train_back=True
    model1.compile(optimizer=keras.optimizers.Adam(1e-4))

    history = model1.fit(data_gen, epochs=n_epochs, callbacks=[TreyerScheduler()])
    model1.save_weights(base_path+save_w_path+"_ALL_base="+base+"_model="+model_name+".weights.h5")

    print("TRAIN ENDED FOR ", base_path+save_w_path+"_ALL_base="+base+"_model="+model_name+".weights.h5")
   
