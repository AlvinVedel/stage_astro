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
model_dir = "model_save/checkpoints_simCLR_finetune_progress/"
model_name = "norm300_ColorHead_NotRegularized_resnet50"



#model = AstroFinetune(basic_backbone(full_bn=False, all_bn=False), head=astro_head(1024, 400))
model = AstroFinetune(ResNet50(include_top=False, weights=None, input_shape=(64, 64, 6), pooling='avg'), head=astro_head(2048, 400))
model(np.random.random((32, 64, 64, 6)))


bigbins_edges = np.linspace(0, 6, 13)  # de 0 à 6 en 0.5
bigbins_centres = (bigbins_edges[1:] + bigbins_edges[:-1])/2




def z_med(probas, bin_central_values) :
    cdf = np.cumsum(probas)
    index = np.argmax(cdf>=0.5)
    return bin_central_values[index]

bins_edges = np.concatenate([np.linspace(0, 4, 381), np.linspace(4, 6, 21)[1:]], axis=0)
bins_centres = (bins_edges[1:] + bins_edges[:-1])/2

def extract_z(tup) :
    return tup[40]
def delta_z(z_pred, z_spec) :
    return (z_pred - z_spec) / (1 + z_spec)






path_memory = {}
inf_bases = ["_UD", "_D"]
inf_bases_aliases = ["UD", "D"]
max_images = [500, 1000, 10000, 50000, 100000, 150000]
metrics_list = ["bias", "smad", 'outl']


inf_base_ud = np.load(base_path+"data/cleaned_spec/cube_4_UD.npz", allow_pickle=True)
inf_base_d = np.load(base_path+"data/cleaned_spec/cube_1_D.npz", allow_pickle=True)
### pour chacune des bases d'inférence on a les paths 



metric_save = np.zeros((len(max_images), 2, len(bigbins_centres), 3))   


for i, mx in enumerate(max_images) :

    model.load_weights(base_path+model_dir+model_name+"_img="+str(mx)+".weights.h5")

    for j, base in [inf_base_ud, inf_base_d] :

        images = base["cube"][..., :6]
        meta = base["info"]
        z = np.array([extract_z(m) for m in meta])
        output = model.predict(images)
        probas = output["pdf"]
        z_meds = np.array([z_med(p, bins_centres) for p in probas])

        deltas_z = (z_meds - z) / (1 + z)
        for bin_ in range(len(bigbins_edges)-1) :
            
            ### BIAS
            inds = np.where((z>=bigbins_edges[bin_]) & (z_meds<bigbins_edges[bin_+1]))
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




data_frame = {"metric":[], "nb_images":[], "value":[], "plage":[], "inference_base":[]}




   

for i, mx in enumerate(max_images) :


    for j, ib in enumerate(["UD", "D"]) :


        for bin_, centre in enumerate(bigbins_centres) :


            for k, metr in enumerate(metrics_list) :

                data_frame["metric"].append(metr)
                data_frame["nb_images"].append(mx)
                data_frame["plage"].append(centre)
                data_frame["inference_base"].append(ib)
                data_frame["value"].append(metric_save[i, j, bin_, k])

                

 

import pandas as pd

df = pd.DataFrame(data_frame)
df.to_csv(base_path+"data/metrics_save/resultats_progressiv_finetune_resnet50.csv", index=False)




    





