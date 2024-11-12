import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
from tensorflow.keras import layers
import tensorflow.keras as keras
from vit_layers import Backbone, BackboneAstro
import matplotlib.pyplot as plt
#from scipy.stats import gaussian_kde
from matplotlib import cm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'






folder_path = "/lustre/fswork/projects/rech/kof/uve94ap/CUBES_HSC/PHOT/DEEP2"
folder_path2 = "/lustre/fswork/projects/rech/kof/uve94ap/CUBES_HSC/PHOT/COSMOS"


file_paths = []

for file_name in os.listdir(folder_path):
    if file_name.endswith('.npz'):
        file_path = os.path.join(folder_path, file_name)
        file_paths.append(file_path)

file_paths2 = {'d':[], 'ud':[]}
for file_name in os.listdir(folder_path2):
    if file_name.endswith('_D.npz'):
        file_path = os.path.join(folder_path2, file_name)
        file_paths2['d'].append(file_path)
    elif file_name.endswith('_UD.npz'):
        file_path = os.path.join(folder_path2, file_name)
        file_paths2['ud'].append(file_path)
    

indices = np.arange(len(file_paths))
indices2 = np.arange(len(file_paths2["d"]))
indices3 = np.arange(len(file_paths2["ud"]))
import random
random.shuffle(indices)
random.shuffle(indices2)
random.shuffle(indices3)

import gc

all_images = []
all_metas = []
origin_label = []

def extract_meta(tup) :
    # RA   DEC   EB_V   ZPHOT   EBV
    return np.array([tup[1], tup[2], tup[7], max(tup[29], 1e-4), tup[35]])


for i in range(28) :
    ind = indices[i]
    print("FILE ", i, file_paths[ind])
    data = np.load(file_paths[ind], allow_pickle=True)
    images = data["cube"]
    meta = data["info"]

    si = np.arange(0, images.shape[0])
    random.shuffle(si)
    si = si[:500]
    images = images[si]
    meta = meta[si]

    images = np.sign(images)*(np.sqrt(np.abs(images+1))-1)   
    all_images.append(images)
    metas = np.array([extract_meta(met) for met in meta])
    all_metas.append(metas)
    label = ['deep2' for _ in range(images.shape[0])]
    origin_label.append(label)
    gc.collect()



    ind = indices2[i]
    print("FILE ", i, file_paths2['d'][ind])
    data = np.load(file_paths2['d'][ind], allow_pickle=True)
    images = data["cube"]
    meta = data["info"]

    si = np.arange(0, images.shape[0])
    random.shuffle(si)
    si = si[:500]
    images = images[si]
    meta = meta[si]

    images = np.sign(images)*(np.sqrt(np.abs(images+1))-1)   
    all_images.append(images)
    metas = np.array([extract_meta(met) for met in meta])
    all_metas.append(metas)
    label = ['cosmos_d' for _ in range(images.shape[0])]
    origin_label.append(label)
    gc.collect()



    ind = indices3[i]
    print("FILE ", i, file_paths2['ud'][ind])
    data = np.load(file_paths2['ud'][ind], allow_pickle=True)
    images = data["cube"]
    meta = data["info"]

    si = np.arange(0, images.shape[0])
    random.shuffle(si)
    si = si[:500]
    images = images[si]
    meta = meta[si]

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

import multiprocessing as mp
from scipy.ndimage import center_of_mass
from scipy import ndimage

def get_mask(image, threshold=0.2, center_window_fraction=0.6) :
            mean_image = np.mean(image, axis=-1)
            #print("dans le get mask")
            binary_image = mean_image > threshold * np.max(mean_image)        
            labeled_image, num_labels = ndimage.label(binary_image)        
            h, w = mean_image.shape
            center_h, center_w = h // 2, w // 2
            window_size_h, window_size_w = int(h * center_window_fraction), int(w * center_window_fraction)        
            min_h, max_h = max(0, center_h - window_size_h // 2), min(h, center_h + window_size_h // 2)
            min_w, max_w = max(0, center_w - window_size_w // 2), min(w, center_w + window_size_w // 2)
            best_label = None
            min_distance = float('inf')
            for label in range(1, num_labels + 1):
                mask = labeled_image == label
                obj_center = center_of_mass(mask)
                if (min_h <= obj_center[0] <= max_h) and (min_w <= obj_center[1] <= max_w):
                    distance = np.linalg.norm(np.array(obj_center) - np.array([center_h, center_w]))
                    if distance < min_distance:
                        min_distance = distance
                        best_label = label
            mask = labeled_image == best_label if best_label is not None else np.zeros_like(mean_image)
            
            mask[image.shape[0]//2-2:image.shape[0]//2+2, image.shape[1]//2-2:image.shape[1]//2+2] = np.ones((4, 4))
            return mask

with mp.Pool(processes=mp.cpu_count()) as pool:
    masques = pool.map(get_mask, images)

print(images.shape)
masques = np.array(masques)
print(masques.shape)

origin_label = np.concatenate(origin_label, axis=0)

colors_dict = {'deep2':'blue', 'cosmos_d':'red', 'cosmos_ud':'yellow'}
colors = []
labels = []
for s in origin_label :
    colors.append(colors_dict[s])
    labels.append(s)


# COSMOS D ET UD,                                        COSMOS D ET DEEP 2,                           COSMO UD + DEEP2                     DEEP2
weights_paths = ["./teacher_backbone.weights.h5", "./checkpoints_dino_astro/teacher_backbone.weights.h5", "./checkpoints_dino_color/teacher_backbone.weights.h5"]
code_w = ['Base', 'DA', 'color']



for i, w in enumerate(weights_paths) :
    if i == 1 or i==2 : 

        model = BackboneAstro()
        model(np.random.random((32, 64, 64, 10)))
    else :
        model = Backbone()
        model(np.random.random((32, 64, 64, 9)))

    model.load_weights(w)

    #model.load_weights("./checkpoints_d2/byol_d2_epoch_120.h5")
    k = 0 
    features = []
    while k < images.shape[0] :
        print("k =", k)
        
            
        if k+200 > images.shape[0] :
            im_to_pred = images[k:] 
            if i>=1 :
                im_to_pred = np.concatenate([im_to_pred, np.expand_dims(masques[k:], axis=1)], axis=-1) # batch, 64, 64, 9   batch, 64, 64, +1
            f = model.predict(im_to_pred)
        else :
            im_to_pred = images[k:k+200]
            if i>=1 :
                im_to_pred = np.concatenate([im_to_pred, np.expand_dims(masques[k:k+200], axis=1)], axis=-1)
            f = model.predict(images[k:k+200])
        features.append(f) 
        k+=200
    features_cls = [f["cls_token"] for f in features]
    features = np.concatenate(features_cls, axis=0)
    #tokens = model.predict(images)  # 
    #features = tokens["cls_token"]
    #features = extractor.predict(images)
    print(features.shape)

    tsne = TSNE(n_components=2, random_state=42)
    data_tsne = tsne.fit_transform(features)
    print("tsne ended")


    print(data_tsne.shape, z.shape, ra.shape, dec.shape, ebv.shape)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=z, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Redshift (z)') 
    plt.title("t-SNE features DINO colorées par Z")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig("plots_dino/tsne_redshift"+code_w[i]+".png")
    plt.close()



    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=ra, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Ra') 
    plt.title("t-SNE features DINO colorées par RA")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig("plots_dino/tsne_ra"+code_w[i]+".png")
    plt.close()

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=dec, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Dec') 
    plt.title("t-SNE features DINO colorées par DEC")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig("plots_dino/tsne_dec"+code_w[i]+".png")
    plt.close()


    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=ebv, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='ebv') 
    plt.title("t-SNE features DINO colorées par EBV")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig("plots_dino/tsne_ebv"+code_w[i]+".png")
    plt.close()

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=colors, cmap='viridis', alpha=0.6)
    for category, color in colors_dict.items():
        plt.scatter([], [], color=color, label=category)
    plt.legend(title='survey')
    plt.title("t-SNE features DINO colorées par survey")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig("plots_dino/tsne_survey"+code_w[i]+".png")
    plt.close()



