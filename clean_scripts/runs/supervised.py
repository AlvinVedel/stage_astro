import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from utils.astro_metrics import Bias, SigmaMAD, OutlierFraction
import matplotlib.pyplot as plt
from models.deep_models import basic_backbone, treyer_backbone, astro_head, astro_model, AstroModel, adv_network, AstroFinetune
from tensorflow.keras.applications import ResNet50
from vit_layers import ViT_backbone
from generator import SupervisedGenerator
from utils.schedulers import TreyerScheduler, AlternateTreyerScheduler
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'


###

# ENTRAINEMENT SUPERVISE D'UN MODELE 

###


base_path = "/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/finetune/"
adv_path = "/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/cleaned_spec/"
base_names = ["base1", "base2", "base3"]
save_name = "resnet50_baseline_hayat2"



for base in base_names :
    #model = astro_model(ViT_backbone(256, 4, 8, 4, 'average'), astro_head())
    #model = astro_model(basic_backbone(), astro_head())
    #model = AstroModel(back=basic_backbone(), head=astro_head(), is_adv=False, adv_network=adv_network())
    model = AstroFinetune(ResNet50(include_top=False, weights=None, input_shape=(64, 64, 6), pooling='avg'), astro_head(2048, 400), train_back=True)
    #model = AstroFinetune(basic_backbone(full_bn=False, all_bn=False), astro_head(1024, 400))
    model(np.random.random((32, 64, 64, 6)))
    gen = SupervisedGenerator(base_path+base+".npz", batch_size=256, adversarial=False, adversarial_dir=adv_path, adv_extensions=["_D.npz"], nbins=400)

    n_epochs = 100
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
    #model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss={"pdf" : tf.keras.losses.SparseCategoricalCrossentropy(), "reg":tf.keras.losses.MeanAbsoluteError()}, 
    #              metrics= {"pdf":["accuracy"], "reg" :[Bias(name='global_bias'), SigmaMAD(name='global_smad'), OutlierFraction(name='global_outl'),Bias(inf=0, sup=0.4, name='bias1'), Bias(inf=0.4, sup=2, name='bias2'), Bias(inf=2, sup=4, name='bias3'), Bias(inf=4, sup=6, name='bias4'), 
    #              SigmaMAD(inf=0, sup=0.4, name='smad1'), SigmaMAD(inf=0.4, sup=2, name='smad2'), SigmaMAD(inf=2, sup=4, name='smad3'), SigmaMAD(inf=4, sup=6, name='smad4'), OutlierFraction(inf=0, sup=0.4, name='outl1'), OutlierFraction(inf=0.4, sup=2, name='outl2'), OutlierFraction(inf=2, sup=4, name='outl3'), OutlierFraction(inf=4, sup=6, name='outl4')]})
    history = model.fit(gen, epochs=n_epochs, callbacks=[AlternateTreyerScheduler([70, 90])])

    model.save_weights("/lustre/fswork/projects/rech/dnz/ull82ct/astro/model_save/checkpoints_supervised/"+save_name+"_"+base+".weights.h5")

    print("history keys :", history.history.keys())
   