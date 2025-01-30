import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from astro_metrics import Bias, SigmaMAD, OutlierFraction
import matplotlib.pyplot as plt
from deep_models import basic_backbone, treyer_backbone, astro_head, astro_model, AstroModel, adv_network
from vit_layers import ViT_backbone
from generator import SupervisedGenerator
from schedulers import TreyerScheduler
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'




base_path = "/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/finetune/"
adv_path = "/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/cleaned_spec/"
base_names = ["base1", "base2", "base3"]
save_name = "cleaned_cnn_supervised_noadv"


#x_val = np.load("/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/cleaned_spec/cube_1_UD.npz", allow_pickle=True)["cube"][:10000, :, :, :6]
#y_val = np.load("/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/cleaned_spec/cube_1_UD.npz", allow_pickle=True)["info"][:10000]["ZSPEC"]
#x_val2 = np.load("/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/cleaned_spec/cube_1_D.npz", allow_pickle=True)["cube"][:10000, :, :, :6]

#val_data = ((x_val, x_val2), y_val)


for base in base_names :
    #model = astro_model(ViT_backbone(256, 4, 8, 4, 'average'), astro_head())
    #model = astro_model(basic_backbone(), astro_head())
    model = AstroModel(back=basic_backbone(), head=astro_head(), is_adv=False, adv_network=adv_network())
    model(np.random.random((32, 64, 64, 6)))
    gen = SupervisedGenerator(base_path+base+".npz", batch_size=32, adversarial=False, adversarial_dir=adv_path, adv_extensions=["_D.npz"])

    n_epochs = 50
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
    #model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss={"pdf" : tf.keras.losses.SparseCategoricalCrossentropy(), "reg":tf.keras.losses.MeanAbsoluteError()}, 
    #              metrics= {"pdf":["accuracy"], "reg" :[Bias(name='global_bias'), SigmaMAD(name='global_smad'), OutlierFraction(name='global_outl'),Bias(inf=0, sup=0.4, name='bias1'), Bias(inf=0.4, sup=2, name='bias2'), Bias(inf=2, sup=4, name='bias3'), Bias(inf=4, sup=6, name='bias4'), 
    #              SigmaMAD(inf=0, sup=0.4, name='smad1'), SigmaMAD(inf=0.4, sup=2, name='smad2'), SigmaMAD(inf=2, sup=4, name='smad3'), SigmaMAD(inf=4, sup=6, name='smad4'), OutlierFraction(inf=0, sup=0.4, name='outl1'), OutlierFraction(inf=0.4, sup=2, name='outl2'), OutlierFraction(inf=2, sup=4, name='outl3'), OutlierFraction(inf=4, sup=6, name='outl4')]})
    history = model.fit(gen, epochs=n_epochs, callbacks=[TreyerScheduler()])

    model.save_weights("/lustre/fswork/projects/rech/dnz/ull82ct/astro/model_save/checkpoints_supervised/"+save_name+"_"+base+".weights.h5")

    print("history keys :", history.history.keys())
    """
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_loss"])
    plt.xlabel("epochs")
    plt.ylabel("loss (mae)")
    plt.title("supervised loss")
    plt.savefig("/lustre/fswork/projects/rech/dnz/ull82ct/astro/plots/simCLR/simCLR_finetune/loss_"+save_name+"_base="+base+".png")
    plt.close()


    plt.plot(np.arange(1, n_epochs+1), history.history["reg_global_bias"], label='biais moyen')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_bias1"], label='[0, 0.4[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_bias2"], label='[0.4, 2[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_bias3"], label='[2, 4[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_bias4"], label='[4, 6[')
    plt.xlabel("epochs")
    plt.ylabel("Bias")
    plt.legend()
    plt.title("supervised bias")
    plt.savefig("/lustre/fswork/projects/rech/dnz/ull82ct/astro/plots/simCLR/simCLR_finetune/bias_"+save_name+"_base="+base+".png")
    plt.close()

    plt.plot(np.arange(1, n_epochs+1), history.history["reg_global_smad"], label='smad moyen')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_smad1"], label='[0, 0.4[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_smad2"], label='[0.4, 2[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_smad3"], label='[2, 4[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_smad4"], label='[4, 6[')
    plt.xlabel("epochs")
    plt.ylabel("Sigma MAD")
    plt.legend()
    plt.title("supervised smad")
    plt.savefig("/lustre/fswork/projects/rech/dnz/ull82ct/astro/plots/simCLR/simCLR_finetune/smad_"+save_name+"_base="+base+".png")
    plt.close()


    plt.plot(np.arange(1, n_epochs+1), history.history["reg_global_outl"], label='outl moyen')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_outl1"], label='[0, 0.4[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_outl2"], label='[0.4, 2[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_outl3"], label='[2, 4[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_outl4"], label='[4, 6[')
    plt.xlabel("epochs")
    plt.ylabel("Outlier Fraction")
    plt.legend()
    plt.title("supervised outl")
    plt.savefig("/lustre/fswork/projects/rech/dnz/ull82ct/astro/plots/simCLR/simCLR_finetune/outl_"+save_name+"_base="+base+".png")
    plt.close()
    """






#model.backbone.save_weights("sdss_backbone.weights.h5")
#model.classifier.save_weights("sdss_classifier.weights.h5")





