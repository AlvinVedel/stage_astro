
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from contrastiv_model import simCLR, simCLRcolor1
from deep_models import basic_backbone, projection_mlp, color_mlp, segmentor, deconvolutor, astro_head, astro_model, classif_mlp
import os 
from vit_layers import ViT_backbone
from generator import SupervisedGenerator
from schedulers import TreyerScheduler
from astro_metrics import Bias, SigmaMAD, OutlierFraction
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import time

base_path = "/lustre/fswork/projects/rech/dnz/ull82ct/astro/"





#model = simCLR(basic_backbone(), projection_mlp(1024))
model = simCLRcolor1(ViT_backbone(), projection_mlp(256), color_mlp(256))
model(np.random.random((32, 64, 64, 5)))



load_w_path = "model_save/checkpoints_simCLR_UD_D/simCLR_cosmos_bnTrue_"
model_name = "280_ViTback_ColorHead"
save_w_path = "model_save/checkpoints_simCLR_finetune/simCLR_finetune_UD_D_"
plots_path = "plots/simCLR/simCLR_finetune/"

for base in ["b1_1", "b2_1", "b3_1"] :

    data_gen = SupervisedGenerator("/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/finetune/"+base+"_v2.npz", batch_size=32, nbins=400)

   
    
    model(np.random.random((32, 64, 64, 5)))
    model.load_weights(base_path+load_w_path+model_name+".weights.h5")

    extracteur = model.backbone
    #extracteur = keras.Model(extracteur.input, extracteur.layers[-1].output)
    model1 = astro_model(extracteur, astro_head(256, 400))

    model1.compile(optimizer=keras.optimizers.Adam(1e-4), loss={"pdf" : tf.keras.losses.SparseCategoricalCrossentropy(), "reg":tf.keras.losses.MeanAbsoluteError()}, metrics= {"pdf":["accuracy"], "reg" :[Bias(name='global_bias'), SigmaMAD(name='global_smad'), OutlierFraction(name='global_outl'),Bias(inf=0, sup=0.4, name='bias1'), Bias(inf=0.4, sup=2, name='bias2'), Bias(inf=2, sup=4, name='bias3'), Bias(inf=4, sup=6, name='bias4'), 
                  SigmaMAD(inf=0, sup=0.4, name='smad1'), SigmaMAD(inf=0.4, sup=2, name='smad2'), SigmaMAD(inf=2, sup=4, name='smad3'), SigmaMAD(inf=4, sup=6, name='smad4'), OutlierFraction(inf=0, sup=0.4, name='outl1'), OutlierFraction(inf=0.4, sup=2, name='outl2'), OutlierFraction(inf=2, sup=4, name='outl3'), OutlierFraction(inf=4, sup=6, name='outl4')]})
    
    n_epochs = 50
    history = model1.fit(data_gen, epochs=n_epochs, callbacks=[TreyerScheduler()])
    model1.save_weights(base_path+save_w_path+"_HeadOnly_base="+base+"_model="+model_name+".weights.h5")

    print("TRAIN ENDED FOR ", base_path+save_w_path+"_HeadOnly_base="+base+"_model="+model_name+".weights.h5")
    """
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_loss"])
    plt.xlabel("epochs")
    plt.ylabel("loss (mae)")
    plt.title("finetuning loss")
    plt.savefig(base_path+plots_path+"loss_HeadOnly_base="+base+"_model="+model_name+".png")
    plt.close()

    plt.plot(np.arange(1, n_epochs+1), history.history["reg_global_bias"], label='biais moyen')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_bias1"], label='[0, 0.4[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_bias2"], label='[0.4, 2[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_bias3"], label='[2, 4[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_bias4"], label='[4, 6[')
    plt.xlabel("epochs")
    plt.ylabel("Bias")
    plt.legend()
    plt.title("finetuning bias")
    plt.savefig(base_path+plots_path+"bias_HeadOnly_base="+base+"_model="+model_name+".png")
    plt.close()

    plt.plot(np.arange(1, n_epochs+1), history.history["reg_global_smad"], label='smad moyen')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_smad1"], label='[0, 0.4[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_smad2"], label='[0.4, 2[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_smad3"], label='[2, 4[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_smad4"], label='[4, 6[')
    plt.xlabel("epochs")
    plt.ylabel("Sigma MAD")
    plt.legend()
    plt.title("finetuning smad")
    plt.savefig(base_path+plots_path+"smad_HeadOnly_base="+base+"_model="+model_name+".png")
    plt.close()


    plt.plot(np.arange(1, n_epochs+1), history.history["reg_global_outl"], label='outl moyen')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_outl1"], label='[0, 0.4[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_outl2"], label='[0.4, 2[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_outl3"], label='[2, 4[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_outl4"], label='[4, 6[')
    plt.xlabel("epochs")
    plt.ylabel("Outlier Fraction")
    plt.legend()
    plt.title("finetuning outl")
    plt.savefig(base_path+plots_path+"outl_HeadOnly_base="+base+"_model="+model_name+".png")
    plt.close()
    """
    

    ##### PARTIE 2

    
    model(np.random.random((32, 64, 64, 5)))
    model.load_weights(base_path+load_w_path+model_name+".weights.h5")

    extracteur = model.backbone
    #extracteur = keras.Model(extracteur.input, extracteur.layers[-1].output)
    model1 = astro_model(extracteur, astro_head(256, 400))

    model1.compile(optimizer=keras.optimizers.Adam(1e-4), loss={"pdf" : tf.keras.losses.SparseCategoricalCrossentropy(), "reg":tf.keras.losses.MeanAbsoluteError()}, metrics= {"pdf":["accuracy"], "reg" :[Bias(name='global_bias'), SigmaMAD(name='global_smad'), OutlierFraction(name='global_outl'),Bias(inf=0, sup=0.4, name='bias1'), Bias(inf=0.4, sup=2, name='bias2'), Bias(inf=2, sup=4, name='bias3'), Bias(inf=4, sup=6, name='bias4'), 
                  SigmaMAD(inf=0, sup=0.4, name='smad1'), SigmaMAD(inf=0.4, sup=2, name='smad2'), SigmaMAD(inf=2, sup=4, name='smad3'), SigmaMAD(inf=4, sup=6, name='smad4'), OutlierFraction(inf=0, sup=0.4, name='outl1'), OutlierFraction(inf=0.4, sup=2, name='outl2'), OutlierFraction(inf=2, sup=4, name='outl3'), OutlierFraction(inf=4, sup=6, name='outl4')]})
    
    history = model1.fit(data_gen, epochs=n_epochs, callbacks=[TreyerScheduler()])
    model1.save_weights(base_path+save_w_path+"_ALL_base="+base+"_model="+model_name+".weights.h5")

    print("TRAIN ENDED FOR ", base_path+save_w_path+"_ALL_base="+base+"_model="+model_name+".weights.h5")
    """
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_loss"])
    plt.xlabel("epochs")
    plt.ylabel("loss (mae)")
    plt.title("finetuning loss")
    plt.savefig(base_path+plots_path+"loss_ALL_base="+base+"_model="+model_name+".png")
    plt.close()



    plt.plot(np.arange(1, n_epochs+1), history.history["reg_global_bias"], label='biais moyen')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_bias1"], label='[0, 0.4[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_bias2"], label='[0.4, 2[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_bias3"], label='[2, 4[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_bias4"], label='[4, 6[')
    plt.xlabel("epochs")
    plt.ylabel("Bias")
    plt.legend()
    plt.title("finetuning bias")
    plt.savefig(base_path+plots_path+"bias_ALL_base="+base+"_model="+model_name+".png")
    plt.close()

    plt.plot(np.arange(1, n_epochs+1), history.history["reg_global_smad"], label='smad moyen')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_smad1"], label='[0, 0.4[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_smad2"], label='[0.4, 2[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_smad3"], label='[2, 4[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_smad4"], label='[4, 6[')
    plt.xlabel("epochs")
    plt.ylabel("Sigma MAD")
    plt.legend()
    plt.title("finetuning smad")
    plt.savefig(base_path+plots_path+"smad_ALL_base="+base+"_model="+model_name+".png")
    plt.close()


    plt.plot(np.arange(1, n_epochs+1), history.history["reg_global_outl"], label='outl moyen')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_outl1"], label='[0, 0.4[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_outl2"], label='[0.4, 2[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_outl3"], label='[2, 4[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_outl4"], label='[4, 6[')
    plt.xlabel("epochs")
    plt.ylabel("Outlier Fraction")
    plt.legend()
    plt.title("finetuning outl")
    plt.savefig(base_path+plots_path+"outl_ALL_base="+base+"_model="+model_name+".png")
    plt.close()
    """

