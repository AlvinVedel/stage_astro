import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
from tensorflow.keras import layers
import tensorflow.keras as keras
from contrastiv_model import simCLRcolor1
from deep_models import basic_backbone, projection_mlp, color_mlp, classif_mlp
import matplotlib.pyplot as plt
from vit_layers import ViT_backbone
#from scipy.stats import gaussian_kde
from matplotlib import cm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'



model = simCLRcolor1(basic_backbone(), projection_mlp(1024), color_mlp(1024))
#model = simCLRcolor1(ViT_backbone(), projection_mlp(256), color_mlp(256))
#model = simCLR(basic_backbone(), projection_mlp(1024))
base_path = "../model_save/checkpoints_simCLR_UD_D/"
model_name = "simCLR_cosmos_bnTrue_200_ColorHead_Regularized.weights.h5"

code_save = "knn"

model(np.random.random((32, 64, 64, 5)))
model.load_weights(base_path+model_name)
extractor = model.backbone



#folder_path2 = "/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/spec/"
folder_path2 = "/lustre/fswork/projects/rech/kof/uve94ap/CUBES_HSC/SPEC/COSMOS/" 


file_paths2 = {'d':[], 'ud':[]}
for file_name in os.listdir(folder_path2):
    if file_name.endswith('_D.npz'):
        file_path = os.path.join(folder_path2, file_name)
        file_paths2['d'].append(file_path)
    elif file_name.endswith('_UD.npz'):
        file_path = os.path.join(folder_path2, file_name)
        file_paths2['ud'].append(file_path)
    

z_key = "ZPHOT"

ud_latent = []
ud_z = []

print("There is ", len(file_paths2["ud"]), "files in ud")

for i, path in enumerate(file_paths2["ud"]) :
    print(i)

    info_i = np.load(path, allow_pickle=True)["info"]
    info_i.dtype.names = tuple([x.lower() for x in info_i.dtype.names])
    mask = (info_i["i"] >= 18) & (info_i["i"] <= 25)
    if "flag" in np.load(path).files :
        flags = np.load(path)["flag"][:,[0,  1,  2,  3,  4,  5]]
        mask = mask & (np.sum(flags,axis=1)==0)
    if z_key in info_i.dtype.names:
        mask = mask & (info_i[z_key] >= 0.001) & (info_i[z_key] <= 6)

    images = np.load(path, allow_pickle=True)["cube"][mask][..., :5]
    images = np.sign(images) * (np.sqrt(np.abs(images) + 1) - 1)
    latent = extractor(images)
    z_assoc = np.load(path, allow_pickle=True)["info"][mask][z_key]
    ud_latent.append(latent)
    ud_z.append(z_assoc)

ud_latent = np.concatenate(ud_latent, axis=0)
ud_z = np.concatenate(ud_z, axis=0)


d_latent = []
d_z = []

print("There is ", len(file_paths2["d"]), "files in d")


for i, path in enumerate(file_paths2["d"]) :
    print(i)

    info_i = np.load(path, allow_pickle=True)["info"]
    info_i.dtype.names = tuple([x.lower() for x in info_i.dtype.names])
    mask = (info_i["i"] >= 18) & (info_i["i"] <= 25)
    if "flag" in np.load(path).files :
        flags = np.load(path)["flag"][:,[0,  1,  2,  3,  4,  5]]
        mask = mask & (np.sum(flags,axis=1)==0)
    if z_key in info_i.dtype.names:
        mask = mask & (info_i[z_key] >= 0.001) & (info_i[z_key] <= 6)

    images = np.load(path, allow_pickle=True)["cube"][mask][..., :5]
    images = np.sign(images) * (np.sqrt(np.abs(images) + 1) - 1)
    latent = extractor(images)
    z_assoc = np.load(path, allow_pickle=True)["info"][mask][z_key]
    d_latent.append(latent)
    d_z.append(z_assoc)

d_latent = np.concatenate(d_latent, axis=0)
d_z = np.concatenate(d_z, axis=0)


#deltas = np.zeros(d_z.shape[0])

from sklearn.neighbors import KNeighborsRegressor

n_ud = len(ud_latent)
inferences_edges = np.linspace(0, 6, 20)
inferences_edges_mid = (inferences_edges[1:]+inferences_edges[:-1])/2



for d, n_dim in enumerate([2, 10, 200, 1000]) :
    print("D = ", n_dim)

    tsne = TSNE(n_components=n_dim, random_state=42)
    all_latent = np.concatenate([ud_latent, d_latent], axis=0)
    projections = tsne.fit_transform(all_latent)

    fig, axes = plt.subplots(nrows=4, ncols=4)

    for n, n_neigh in enumerate([1, 10, 100, 200]) :
        print("N =", n_neigh)

        neigh = KNeighborsRegressor(n_neighbors=n_neigh, weights='distance')
        neigh.fit(projections[:n_ud], ud_z)
        predictions = neigh.predict(projections[n_ud:])


        metrics = np.zeros((len(inferences_edges)-1, 3))
        deltas = (predictions - d_z) / (1+d_z)
        

        axes[n, 0].set_xlabel("redshift")
        axes[n, 0].set_ylabel("bias")

        axes[n, 1].set_xlabel("redshift")
        axes[n, 1].set_ylabel("smad")

        axes[n, 2].set_xlabel("redshift")
        axes[n, 2].set_ylabel("outl frac")

        axes[n, 3].set_xlabel("true z")
        axes[n, 3].set_ylabel("pred z")


        axes[n, 0].set_title("n neighbor ="+str(n_neigh))
        


        for i in range(len(inferences_edges)-1) :
            inds = np.where((d_z>=inferences_edges[i]) & (d_z<inferences_edges[i+1]))
            selected_deltas = deltas[inds]

            metrics[i, 0] = np.mean(selected_deltas)

            median_delta_z_norm = np.median(selected_deltas)
            mad = np.median(np.abs(selected_deltas - median_delta_z_norm))
            sigma_mad = 1.4826 * mad
            metrics[d, n, i, 1] = sigma_mad

            outliers = np.abs(selected_deltas) > 0.05
            fraction_outliers = np.sum(outliers) / (len(selected_deltas)+1e-6)
            metrics[d, n, i, 2] = fraction_outliers


        print("métriques :", metrics)


        axes[n, 0].plot(inferences_edges_mid, metrics[:, 0])
        axes[n, 1].plot(inferences_edges_mid, metrics[:, 1])
        axes[n, 2].plot(inferences_edges_mid, metrics[:, 2])
        axes[n, 3].plot(d_z, predictions)



    fig.savfig("../plots/knn_results_dim_tsne="+str(n_dim)+".png")


        










if True :    
    
    features = np.concatenate([ud_latent, d_latent])
    z = np.log(1+np.concatenate([ud_z,d_z], axis=0))

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




    #### Z UD + D
    plt.figure(figsize=(10, 8))
    scatter1 = plt.scatter(
        data_tsne[n_ud:, 0], data_tsne[n_ud:, 1], 
        c=z[n_ud:], cmap='viridis', marker='^', alpha=0.6, label='D', vmin=vmin, vmax=vmax
    )
    scatter2 = plt.scatter(
        data_tsne[:n_ud, 0], data_tsne[:n_ud, 1], 
        c=z[:n_ud], cmap='viridis', marker='o', alpha=0.6, label='UD', vmin=vmin, vmax=vmax
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
        data_tsne[n_ud:, 0], data_tsne[n_ud:, 1], 
        c=z[n_ud:], cmap='viridis', marker='^', alpha=0.6, label='D', vmin=vmin, vmax=vmax
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
        data_tsne[:n_ud, 0], data_tsne[:n_ud, 1], 
        c=z[:n_ud], cmap='viridis', marker='o', alpha=0.6, label='UD', vmin=vmin, vmax=vmax
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











