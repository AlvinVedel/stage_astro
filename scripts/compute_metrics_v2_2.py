import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from contrastiv_model import simCLR
import matplotlib.pyplot as plt
from vit_layers import ViT_backbone
from deep_models import basic_backbone, astro_head, astro_model, AstroModel, adv_network, AstroFinetune
import os
from keras.applications import ResNet50
import gc
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"]='0'

base_path = "/lustre/fswork/projects/rech/dnz/ull82ct/astro/"
model_dir = "model_save/checkpoints_simCLR_finetune2/"



## si supervisé, ne pas inclure weights.h5
model_name = "norm300_ColorHead_NotRegularized_resnet50_HeadPre10_clip3_hayat2.weights.h5"
#model_name = "basic_baseline_log"
plots_name = "resnet_color_h10_c3_b256"
with_plots = False
supervised = False
log_z = False

if supervised :
    model_dir = "model_save/checkpoints_supervised/"


#model = AstroFinetune(basic_backbone(full_bn=False, all_bn=False), head=astro_head(1024, 400))
model = AstroFinetune(ResNet50(include_top=False, weights=None, input_shape=(64, 64, 6), pooling='avg'), head=astro_head(2048, 400))
model(np.random.random((32, 64, 64, 6)))


bigbins_edges = np.linspace(0, 6, 13)  # de 0 à 6 en 0.5
bigbins_centres = (bigbins_edges[1:] + bigbins_edges[:-1])/2

data_frame = {"metric":[], "finetune_base":[], "value":[], "plage":[], "inference_base":[]}










metric_save = np.zeros((2, 3, 12, 3))   # pour le modèle on a 3 bases de finetune  => on considère que le finetune de tout   mais print inférence tête ?
# sur 12 plages et 3 métriques   et surtout 2 bases d'inférence

def z_med(probas, bin_central_values) :
    cdf = np.cumsum(probas)
    index = np.argmax(cdf>=0.5)
    return bin_central_values[index]

if log_z :
    bins_edges = np.linspace(0, np.log(6+1), 401)
    bins_centres = (bins_edges[1:] + bins_edges[:-1])/2
else :
    bins_edges = np.concatenate([np.linspace(0, 4, 381), np.linspace(4, 6, 21)[1:]], axis=0)
    bins_centres = (bins_edges[1:] + bins_edges[:-1])/2

def extract_z(tup) :
    return tup[40]
def delta_z(z_pred, z_spec) :
    return (z_pred - z_spec) / (1 + z_spec)








path_memory = {}
inf_bases = ["_UD", "_D"]
inf_bases_aliases = ["UD", "D"]
train_bases = ["base1", "base2", "base3"]
metrics_list = ["bias", "smad", 'outl']


for inf_base in inf_bases :
    dir_path = Path(base_path+"data/cleaned_spec/")
    extension = inf_base+'.npz'
    npz_files = [file for file in dir_path.rglob(f"*{extension}") if "4" not in str(file)]
    print("npz files :", npz_files)
    path_memory[inf_base] = npz_files

### pour chacune des bases d'inférence on a les paths 


if with_plots :
    fig1, axes1 = plt.subplots(nrows=3, ncols=4, figsize=(30, 40))
    fig2, axes2 = plt.subplots(nrows=3, ncols=4, figsize=(30, 40))

    figs = [(fig1, axes1), (fig2, axes2)]

for i, inf_base in enumerate(inf_bases) :
    npz_files = path_memory[inf_base]


    true_zs = []
    pred_zs = []

    for file in npz_files :
                        
        data = np.load(str(file), allow_pickle=True)
        images = data["cube"][..., :6]
        print("images loaded")
        meta = data["info"]
        print("meta loaded")
        z = np.array([extract_z(m) for m in meta])
        true_zs.append(z)
        file_preds = []   # liste de taille      3, 50k
 

        for j, tb in enumerate(train_bases) :
                        
            if supervised :
                model.load_weights(base_path+model_dir+model_name+"_"+tb+".weights.h5")
            else :
                model.load_weights(base_path+model_dir+"simCLR_finetune_UD_D__ALL_base="+tb+"_model="+model_name)
            output = model.predict(images)
            probas = output["pdf"]
            z_meds = np.array([z_med(p, bins_centres) for p in probas])
            if log_z :
                z_meds = np.exp(z_meds) - 1
            file_preds.append(z_meds)

        pred_zs.append(file_preds)   # liste de taille     nb_files,  nb_finetune_bases ,    nb_z (indéterminé)

                        
        del images           
        gc.collect()


    true_zs = np.concatenate(true_zs, axis=0)   # était nb_files, nb_z  => devient nb_z total





    ## On passe au calcul des métriques par plages

    for j, tb in enumerate(train_bases) :
         
        predictions = np.concatenate([pred_zs[a][j] for a in range(len(npz_files))], axis=0)
        deltas_z = (predictions - true_zs) / (1 + true_zs) 



        for bin_ in range(len(bigbins_edges)-1) :
            
            ### BIAS
            inds = np.where((true_zs>=bigbins_edges[bin_]) & (true_zs<bigbins_edges[bin_+1]))
            selected_deltas = deltas_z[inds]
            metric_save[i, j, bin_, 0] = np.mean(selected_deltas)


            ### SMAD
            median_delta_z_norm = np.median(selected_deltas)
            mad = np.median(np.abs(selected_deltas - median_delta_z_norm))
            sigma_mad = 1.4826 * mad
            metric_save[i, j, bin_, 1] = sigma_mad


            ### OUTL
            outliers = np.abs(selected_deltas) > 0.05
            fraction_outliers = np.sum(outliers) / (len(selected_deltas)+1e-6)
            metric_save[i, j, bin_, 2] = fraction_outliers


        if with_plots :

            current_fig = figs[i]
            fig, ax = current_fig

            xy = np.vstack([true_zs, predictions])
            density = gaussian_kde(xy)(xy)

            ax[j, 0].scatter(true_zs, predictions, c=density, cmap='hot', s=2)
            ax[j, 0].set_ylim((-0.5, 6.5))
            ax[j, 0].set_xlim((-0.5, 6.5))
            ax[j, 0].set_xlabel("true Z")
            ax[j, 0].set_ylabel("pred Z")


            for p in range(1, 4) :
                ax[j, p].plot(bigbins_centres, metric_save[i, j, :, p])
                ax[j, p].set_xlabel("redshift (z)")
                ax[j, p].set_ylabel(metrics_list[p-1])
                 




            


if with_plots :
     plt.savefig(base_path+"plots/raw_plots/"+plots_name+".png")
     plt.close()


                        



for i, inf_base in enumerate(inf_bases) :


    for j, tb in enumerate(train_bases) :


        for bin_ in range(len(bigbins_edges)-1) :


            for k, metr in enumerate(metrics_list) :

                data_frame["metric"].append(metr)
                data_frame["finetune_base"].append(train_bases[j])
                data_frame["plage"].append(bigbins_centres[bin_])
                data_frame["inference_base"].append(inf_bases_aliases[i])
                data_frame["value"].append(metric_save[i, j, bin_, k])

                

 

import pandas as pd

df = pd.DataFrame(data_frame)
df.to_csv(base_path+"data/metrics_save/resultats_"+plots_name+".csv", index=False)




    





