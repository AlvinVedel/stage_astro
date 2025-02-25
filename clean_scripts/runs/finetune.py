
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from models.contrastiv_model import simCLR, simCLRcolor1, simCLRcolor1_adversarial, simCLR1
from models.deep_models import basic_backbone, projection_mlp, color_mlp, segmentor, deconvolutor, astro_head, astro_model, classif_mlp, AstroFinetune, noregu_projection_mlp
import os 
from vit_layers import ViT_backbone
from generators.generator import SupervisedGenerator
from utils.schedulers import TreyerScheduler, AlternateTreyerScheduler
from utils.astro_metrics import Bias, SigmaMAD, OutlierFraction
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import time
from keras.applications import ResNet50

base_path = "/lustre/fswork/projects/rech/dnz/ull82ct/astro/"


"""
SIMPLE SCRIPT DE FINETUNE : 
PRE ENTRAINEMENT DE LA TETE INITIALISE ALEATOIREMENT PUIS FINETUNE COMPLET


"""



#model = simCLR1(basic_backbone(full_bn=False), projection_mlp(1024, False))
#model = simCLR1(ResNet50(include_top=False, weights=None, input_shape=(64, 64, 6), pooling='avg'), noregu_projection_mlp(2048, True))
#model = simCLRcolor1(basic_backbone(full_bn=False), projection_mlp(1024, False), color_mlp(1024))
model = simCLRcolor1(ResNet50(include_top=False, weights=None, input_shape=(64, 64, 6), pooling='avg'), noregu_projection_mlp(2048, True), color_mlp(2048))
#model = simCLRcolor1_adversarial(basic_backbone(full_bn=True), projection_mlp(1024, True), color_mlp(1024), classif_mlp())
#model = simCLRcolor1(ViT_backbone(), projection_mlp(256), color_mlp(256))
model(np.random.random((32, 64, 64, 6)))



load_w_path = "model_save/checkpoints_new_simCLR/simCLR_UD_D_"
model_name = "norm300_ColorHead_NotRegularized_resnet50"
save_w_path = "model_save/checkpoints_simCLR_finetune2/simCLR_finetune_UD_D_"
plots_path = "plots/simCLR/simCLR_finetune/"

for base in ["base1", "base2", "base3"] :

    data_gen = SupervisedGenerator("/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/finetune/"+base+".npz", batch_size=256, nbins=400)

   
    
   
    n_epochs = 100
   
    

    ##### PARTIE 2

    for pre in [0, 5, 10] :
        model(np.random.random((32, 64, 64, 6)))
        model.load_weights(base_path+load_w_path+model_name+".weights.h5")
    
        extracteur = model.backbone
        #extracteur = keras.Model(extracteur.input, extracteur.layers[-1].output)
        #model1 = astro_model(extracteur, astro_head(1024, 400))
        model1 = AstroFinetune(extracteur, astro_head(2048, 400), train_back=False)
        model1.compile(optimizer=keras.optimizers.Adam(1e-4))
        model1.fit(data_gen, epochs=pre)   #### ON PRE ENTRAINE LA TETE
        ## RE ENTRAINEEMNT DU MODEL EN ENTRAINANT LE BACK CETTE FOIS
        model1.train_back=True
        model1.compile(optimizer=keras.optimizers.Adam(1e-4))#, clipnorm=1e-3))

        #model1.compile(optimizer=keras.optimizers.Adam(1e-4), loss={"pdf" : tf.keras.losses.SparseCategoricalCrossentropy(), "reg":tf.keras.losses.MeanAbsoluteError()}, metrics= {"pdf":["accuracy"], "reg" :[Bias(name='global_bias'), SigmaMAD(name='global_smad'), OutlierFraction(name='global_outl'),Bias(inf=0, sup=0.4, name='bias1'), Bias(inf=0.4, sup=2, name='bias2'), Bias(inf=2, sup=4, name='bias3'), Bias(inf=4, sup=6, name='bias4'), 
        #              SigmaMAD(inf=0, sup=0.4, name='smad1'), SigmaMAD(inf=0.4, sup=2, name='smad2'), SigmaMAD(inf=2, sup=4, name='smad3'), SigmaMAD(inf=4, sup=6, name='smad4'), OutlierFraction(inf=0, sup=0.4, name='outl1'), OutlierFraction(inf=0.4, sup=2, name='outl2'), OutlierFraction(inf=2, sup=4, name='outl3'), OutlierFraction(inf=4, sup=6, name='outl4')]})
        
        history = model1.fit(data_gen, epochs=n_epochs, callbacks=[AlternateTreyerScheduler([70, 90])])
        model1.save_weights(base_path+save_w_path+"_ALL_base="+base+"_model="+model_name+"_HeadPre"+str(pre)+"_clip3_hayat2.weights.h5")

        print("TRAIN ENDED FOR ", base_path+save_w_path+"_ALL_base="+base+"_model="+model_name+".weights.h5", "pre=", pre)
    


