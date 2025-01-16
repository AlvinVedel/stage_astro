import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
from tensorflow.keras import layers
import tensorflow.keras as keras
from contrastiv_model import simCLR
from deep_models import basic_backbone, projection_mlp, color_mlp, classif_mlp
import matplotlib.pyplot as plt
#from scipy.stats import gaussian_kde
from matplotlib import cm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'




model = simCLR(basic_backbone(), projection_mlp(1024))
base_path = "../model_save/checkpoints_simCLR_UD/"
model_name = "simCLR_cosmos_bnTrue_400_ColorHead.weights.h5"

code_save = "ColorHead_UD400"

model(np.random.random((32, 64, 64, 5)))
model.load_weights(base_path+model_name)




folder_path2 = "/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/spec/"



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

all_images = []
all_metas = []
origin_label = []

def extract_meta(tup) :
    #                  RA      DEC    EB_V         ZPHOT          EBV
    return np.array([tup[1], tup[2], tup[7], min(max(tup[40], 1e-5), 7), tup[35]])

for i in range(len(indices2)) :
    

    ind = indices2[i]
    print("FILE ", i, file_paths2['d'][ind])
    data = np.load(file_paths2['d'][ind], allow_pickle=True)
    images = data["cube"][..., :5]
    meta = data["info"]
    
    n = images.shape[0]
    sub_ind = np.arange(n)
    random.shuffle(sub_ind)
    takes = sub_ind[:250]
    images = images[takes]
    meta = meta[takes]
    
    images = np.sign(images)*(np.sqrt(np.abs(images+1))-1) 
    all_images.append(images)
    metas = np.array([extract_meta(met) for met in meta])
    all_metas.append(metas)
    label = ['cosmos_d' for _ in range(images.shape[0])]
    origin_label.append(label)
    gc.collect()



for i in range(len(indices3)) :

    ind = indices3[i]
    print("FILE ", i, file_paths2['ud'][ind])
    data = np.load(file_paths2['ud'][ind], allow_pickle=True)
    images = data["cube"][..., :5]
    meta = data["info"]

    n = images.shape[0]
    sub_ind = np.arange(n)
    random.shuffle(sub_ind)
    takes = sub_ind[:250]
    images = images[takes]
    meta = meta[takes]

    images = np.sign(images)*(np.sqrt(np.abs(images+1))-1)
    all_images.append(images)
    metas = np.array([extract_meta(met) for met in meta]) # shape 20k, 5
    all_metas.append(metas)
    label = ['cosmos_ud' for _ in range(images.shape[0])]
    origin_label.append(label)
    gc.collect()


images = np.concatenate(all_images, axis=0)
metas = np.concatenate(all_metas, axis=0)  # 12*20k, 5

ra = metas[:, 0]
dec = metas[:, 1]
ebv = metas[:, 2]
z = np.log(metas[:, 3]+1)
print(np.max(z), np.min(z), np.max(metas[:, 3]), np.min(metas[:, 3]))

print("IL Y A", len(images), "IMAGES DANS COSMO UD ET D")



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
    extractor = model.backbone
    if i ==2 :
        features, flatten = extractor.predict(images)
    else :
        features = extractor.predict(images)
    if np.isnan(features).any():
        print("Found NaN values in features, replacing them with 0...")
        features = np.nan_to_num(features, nan=0.0)
    print(features.shape)
    print(features)
    tsne = TSNE(n_components=2, random_state=42)
    data_tsne = tsne.fit_transform(features)
    print("tsne ended")

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





