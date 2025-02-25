import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
from tensorflow.keras import layers
import tensorflow.keras as keras
from models.contrastiv_model import simCLRcolor1, simCLRcolor1_adversarial, simCLR1
from models.deep_models import basic_backbone, projection_mlp, color_mlp, classif_mlp, noregu_projection_mlp
import matplotlib.pyplot as plt
from vit_layers import ViT_backbone
#from scipy.stats import gaussian_kde
from matplotlib import cm
import os
from keras.applications import ResNet50
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


"""

POUR UN MODELE DONNE, RECUPERE LES ESPACES LATENTS DES BASES LABELLISEES
REDUIT LA DIMENSION PAR UNE TSNE ET PRODUIT DES VISUALISATION 2D 
AVEC LA MISE EN RELATION DE CERTAINES VARIABLES (EBV / REDSHIFT)

"""





#model = simCLR1(basic_backbone(full_bn=True), projection_mlp(1024, True))
#model = simCLRcolor1(basic_backbone(full_bn=True, all_bn=False), projection_mlp(1024, True), color_mlp(1024))
#model = AstroFinetune(basic_backbone(full_bn=False, all_bn=False), astro_head(1024, 400))
model = simCLR1(ResNet50(include_top=False, weights=None, input_shape=(64, 64, 6), pooling='avg'), noregu_projection_mlp(2048, bn=True))
#model = simCLRcolor1(ResNet50(include_top=False, weights=None, input_shape=(64, 64, 6), pooling='avg'), noregu_projection_mlp(2048, bn=True), color_mlp(2048))
#model = simCLRcolor1_adversarial(basic_backbone(full_bn=True), projection_mlp(1024, True), color_mlp(1024), classif_mlp())
#model = simCLRcolor1(ViT_backbone(), projection_mlp(256), color_mlp(256))
#model = simCLR(basic_backbone(), projection_mlp(1024))
base_path = "../model_save/checkpoints_new_simCLR/"
#model_name = "simCLR_UD_D_nonorm350_ColorHead_Regularized.weights.h5"
model_name = "simCLR_UD_D_norm300_NoColorHead_NotRegularized_resnet50.weights.h5"
#model_name = "basic_baseline_base1.weights.h5"
supervised = False
if supervised :
    base_path = "../model_save/checkpoints_supervised/"



code_save = "resnet50_nocolor_noreg300"

model(np.random.random((32, 64, 64, 6)))
model.load_weights(base_path+model_name)
if supervised :
    extracteur = model.back
else :
    extracteur = model.backbone


folder_path2 = "/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/cleaned_spec/"



file_paths2 = {'d':[], 'ud':[]}
for file_name in os.listdir(folder_path2):
    if file_name.endswith('_D.npz'):
        file_path = os.path.join(folder_path2, file_name)
        file_paths2['d'].append(file_path)
    elif file_name.endswith('_UD.npz'):
        file_path = os.path.join(folder_path2, file_name)
        file_paths2['ud'].append(file_path)
    

indices2 = np.arange(len(file_paths2["d"]))
indices3 = np.arange(len(file_paths2["ud"]))
import random
random.shuffle(indices2)
random.shuffle(indices3)

import gc

all_latents = []
all_metas = []
origin_label = []

def extract_meta(tup) :
    #                  RA      DEC    EB_V         ZPHOT          EBV
    return np.array([tup[1], tup[2], tup[7], min(max(tup[40], 1e-5), 7), tup[35]])

for i in range(len(indices2)) :
    

    ind = indices2[i]
    print("FILE ", i, file_paths2['d'][ind])
    data = np.load(file_paths2['d'][ind], allow_pickle=True)
    images = data["cube"][..., :6]
    meta = data["info"]
    
    #n = images.shape[0]
    #sub_ind = np.arange(n)
    #random.shuffle(sub_ind)
    #takes = sub_ind[:250]
    #images = images[takes]
    #meta = meta[takes]
    
    #images = np.sign(images)*(np.sqrt(np.abs(images+1))-1) 
    latents = []
    a = 0
    while a < len(images) :
        if a + 500 < len(images) :
            feat = extracteur(images[a:a+500])
        else :
            feat = extracteur(images[a:])
        latents.append(feat)
        a+=500
    latents = np.concatenate(latents, axis=0)
    all_latents.append(latents)
    metas = np.array([extract_meta(met) for met in meta])
    all_metas.append(metas)
    label = ['cosmos_d' for _ in range(images.shape[0])]
    origin_label.append(label)
    gc.collect()



for i in range(len(indices3)) :

    ind = indices3[i]
    print("FILE ", i, file_paths2['ud'][ind])
    data = np.load(file_paths2['ud'][ind], allow_pickle=True)
    images = data["cube"][..., :6]
    meta = data["info"]

    #n = images.shape[0]
    #sub_ind = np.arange(n)
    #random.shuffle(sub_ind)
    #takes = sub_ind[:10000]
    #images = images[takes]
    #meta = meta[takes]
    latents = []
    a = 0
    while a < len(images) :
        if a + 500 < len(images) :
            feat = extracteur(images[a:a+500])
        else :
            feat = extracteur(images[a:])
        latents.append(feat)
        a+=500
    latents = np.concatenate(latents, axis=0)
    all_latents.append(latents)
    #images = np.sign(images)*(np.sqrt(np.abs(images+1))-1)
    #all_images.append(images)
    metas = np.array([extract_meta(met) for met in meta]) # shape 20k, 5
    all_metas.append(metas)
    label = ['cosmos_ud' for _ in range(images.shape[0])]
    origin_label.append(label)
    gc.collect()


features = np.concatenate(all_latents, axis=0)
metas = np.concatenate(all_metas, axis=0)  # 12*20k, 5

ra = metas[:, 0]
dec = metas[:, 1]
ebv = metas[:, 2]
z = np.log(metas[:, 3]+1)
print(np.max(z), np.min(z), np.max(metas[:, 3]), np.min(metas[:, 3]))

print("IL Y A", len(images), "IMAGES DANS COSMO UD ET D")
print("STATS =", np.max(features), np.min(features), np.var(features))
var_per_features = np.var(features, axis=0)
print(np.mean(var_per_features), var_per_features)


origin_label = np.concatenate(origin_label, axis=0)
cat1 = origin_label == 'cosmos_d'
cat2 = origin_label == "cosmos_ud"

colors_dict = {'cosmos_d':'red', 'cosmos_ud':'yellow'}
colors = []
labels = []
for s in origin_label :
    colors.append(colors_dict[s])
    labels.append(s)



    

if True :    
    #extractor = model.backbone
    if False :
        features, flatten = extractor.predict(images)
    elif False :
        features = []
        i = 0
        while i < len(images) :
            if i+200 > len(images) :
                f = extractor(images[i:])
            else :
                f = extractor(images[i:i+200])
            i+=200
            features.append(f)
        features = np.concatenate(features, axis=0)
        #features = extractor.predict(images)
    if np.isnan(features).any():
        print("Found NaN values in features, replacing them with 0...")
        features = np.nan_to_num(features, nan=0.0)
    print(features.shape)
    print(features)
    tsne = TSNE(n_components=2, random_state=42)
    data_tsne = tsne.fit_transform(features)
    print("tsne ended")
    np.save("tsne_coord_"+code_save+".npy", np.concatenate([data_tsne, np.expand_dims(z, axis=1)], axis=1))
    width_x = np.max(data_tsne[:, 0]) - np.min(data_tsne[:, 0])
    width_y = np.max(data_tsne[:, 1]) - np.min(data_tsne[:, 1])


    xlimits = (np.min(data_tsne[:, 0]) - 0.05*width_x, np.max(data_tsne[:, 0]+ 0.05*width_x))
    ylimits = (np.min(data_tsne[:, 1])-0.05*width_y, np.max(data_tsne[:, 1])+0.05*width_y)
    vmin = np.min(z)
    vmax = np.max(z)
    ebvvmin = np.min(ebv)
    ebvvmax = np.max(ebv)

    print(data_tsne.shape, z.shape, ra.shape, dec.shape, ebv.shape)


    #### Z UD + D
    plt.figure(figsize=(10, 8))
    scatter1 = plt.scatter(
        data_tsne[cat1, 0], data_tsne[cat1, 1], 
        c=z[cat1], cmap='viridis', marker='^', alpha=0.6, label='D', vmin=vmin, vmax=vmax
    )
    scatter2 = plt.scatter(
        data_tsne[cat2, 0], data_tsne[cat2, 1], 
        c=z[cat2], cmap='viridis', marker='o', alpha=0.6, label='UD', vmin=vmin, vmax=vmax
    )
    plt.colorbar(scatter1, label='Redshift (z)')
    plt.legend()
    plt.title("t-SNE features colorées par Z")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.ylim(ylimits)
    plt.xlim(xlimits)
    plt.savefig("../plots/simCLR/tsne_redshift_"+code_save+".png")
    plt.close()



    ##### Z D
    plt.figure(figsize=(10, 8))
    scatter1 = plt.scatter(
        data_tsne[cat1, 0], data_tsne[cat1, 1], 
        c=z[cat1], cmap='viridis', marker='^', alpha=0.6, label='D', vmin=vmin, vmax=vmax
    )

    plt.colorbar(scatter1, label='Redshift (z)')
    plt.legend()
    plt.title("t-SNE features colorées par Z")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.ylim(ylimits)
    plt.xlim(xlimits)
    plt.savefig("../plots/simCLR/tsne_redshift_D_"+code_save+".png")
    plt.close()


    #### Z UD
    plt.figure(figsize=(10, 8))
    scatter2 = plt.scatter(
        data_tsne[cat2, 0], data_tsne[cat2, 1], 
        c=z[cat2], cmap='viridis', marker='o', alpha=0.6, label='UD', vmin=vmin, vmax=vmax
    )
    plt.colorbar(scatter2, label='Redshift (z)')
    plt.legend()
    plt.title("t-SNE features colorées par Z")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.ylim(ylimits)
    plt.xlim(xlimits)
    plt.savefig("../plots/simCLR/tsne_redshift_UD_"+code_save+".png")
    plt.close()





    #### EBV
    plt.figure(figsize=(10, 8))
    scatter1 = plt.scatter(
        data_tsne[cat1, 0], data_tsne[cat1, 1], 
        c=ebv[cat1], cmap='viridis', marker='^', alpha=0.6, label='D', vmin=ebvvmin, vmax=ebvvmax
    )
    scatter2 = plt.scatter(
        data_tsne[cat2, 0], data_tsne[cat2, 1], 
        c=ebv[cat2], cmap='viridis', marker='o', alpha=0.6, label='UD', vmin=ebvvmin, vmax=ebvvmax
    )
    plt.colorbar(scatter1, label='EBV')
    plt.legend()
    plt.title("t-SNE features colorées par EBV")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.ylim(ylimits)
    plt.xlim(xlimits)
    plt.savefig("../plots/simCLR/tsne_ebv_base"+code_save+".png")
    plt.close()

    plt.figure(figsize=(10, 8))
    scatter1 = plt.scatter(
        data_tsne[cat1, 0], data_tsne[cat1, 1], 
        c=ebv[cat1], cmap='viridis', marker='^', alpha=0.6, label='D', vmin=ebvvmin, vmax=ebvvmax
    )
    plt.colorbar(scatter1, label='EBV')
    plt.legend()
    plt.title("t-SNE features colorées par EBV")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.ylim(ylimits)
    plt.xlim(xlimits)
    plt.savefig("../plots/simCLR/tsne_ebv_D_base"+code_save+".png")
    plt.close()


    plt.figure(figsize=(10, 8))

    scatter2 = plt.scatter(
        data_tsne[cat2, 0], data_tsne[cat2, 1], 
        c=ebv[cat2], cmap='viridis', marker='o', alpha=0.6, label='UD', vmin=ebvvmin, vmax=ebvvmax
    )
    plt.colorbar(scatter2, label='EBV')
    plt.legend()
    plt.title("t-SNE features colorées par EBV")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.ylim(ylimits)
    plt.xlim(xlimits)
    plt.savefig("../plots/simCLR/tsne_ebv_UD_base"+code_save+".png")
    plt.close()





