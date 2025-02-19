
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from contrastiv_model import simCLR, simCLRcolor1
from deep_models import basic_backbone, projection_mlp, color_mlp, segmentor, deconvolutor, astro_head, astro_model, classif_mlp, AstroFinetune
import os 
from vit_layers import ViT_backbone
from generator import SupervisedGenerator
from schedulers import TreyerScheduler
from astro_metrics import Bias, SigmaMAD, OutlierFraction
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import time
import random

base_path = "/lustre/fswork/projects/rech/dnz/ull82ct/astro/"





#model = simCLR(basic_backbone(), projection_mlp(1024))
model = simCLRcolor1(basic_backbone(), projection_mlp(256), color_mlp(256))
model(np.random.random((32, 64, 64, 5)))



load_w_path = "model_save/checkpoints_new_simCLR/simCLR_UD_D_"
model_name = "norm300_ColorHead_NotRegularized_resnet50"
save_w_path = "model_save/checkpoints_simCLR_finetune_progress/"
plots_path = "plots/simCLR/simCLR_finetune/"

supervised_paths = ["cube1_UD.npz", "cube2_UD.npz", "cube3_UD.npz"]

for max_img in [500, 1000, 10000, 50000, 100000, 150000]  :
    data_paths = []
    count = 0
    index = 0
    while count < max_img :
        data_paths.append(base_path+"data/cleaned_spec/"+supervised_paths[index])
        count+=50000

    random.seed(2)
    np.random.seed(2)

    data_gen = SupervisedGenerator(data_path=data_paths, batch_size=32, nbins=400)
    data_gen.images = data_gen.images[:max_img]
    data_gen.z_bins = data_gen.z_bins[:max_img]
    data_gen.z_values = data_gen.z_values[:max_img]
   
    
    model(np.random.random((32, 64, 64, 5)))
    model.load_weights(base_path+load_w_path+model_name+".weights.h5")

    extracteur = model.backbone
    #extracteur = keras.Model(extracteur.input, extracteur.layers[-1].output)
    model1 = AstroFinetune(extracteur, astro_head(256, 400), train_back=False)
    model1.compile(optimizer=keras.optimizers.Adam(1e-4))
    model1.fit(data_gen, epochs=5)

    model1.train_back = True
    n_epochs = 50
    history = model1.fit(data_gen, epochs=n_epochs, callbacks=[TreyerScheduler()])
    model1.save_weights(base_path+save_w_path+model_name+"_img="+max_img+".weights.h5")

   
