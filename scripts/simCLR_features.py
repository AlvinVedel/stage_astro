import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
from tensorflow.keras import layers
import tensorflow.keras as keras
from contrastiv_model import simCLR, simCLR_adversarial as simCLR_adv
import matplotlib.pyplot as plt
#from scipy.stats import gaussian_kde
from matplotlib import cm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def backbone(bn=True) :
    inp = keras.Input((64, 64, 5))
    c1 = layers.Conv2D(96, padding='same', strides=1, kernel_size=3)(inp) # 64
    c1 = layers.PReLU()(c1) 
    c2 = layers.Conv2D(96, padding='same', kernel_size=3, strides=1, activation='tanh')(c1)  #64
    p1 = layers.AveragePooling2D((2, 2))(c2)  # 32
    c3 = layers.Conv2D(128, padding='same', strides=1, kernel_size=3)(p1)
    c3 = layers.PReLU()(c3)
    c4 = layers.Conv2D(128, padding='same', kernel_size=3, strides=1)(c3)  #32
    c4 = layers.PReLU(name='c4')(c4) 
    p2 = layers.AveragePooling2D((2, 2))(c4)  # 16
    c5 = layers.Conv2D(256, padding='same', strides=1, kernel_size=3)(p2) #16
    c5 = layers.PReLU()(c5)
    c6 = layers.Conv2D(256, padding='same', kernel_size=3, strides=1)(c5)  #16
    c6 = layers.PReLU()(c6)
    p3 = layers.AveragePooling2D((2, 2))(c6) # 8
    c7 = layers.Conv2D(256, kernel_size=3, strides=1, padding='valid')(p3) # 6
    c7 = layers.PReLU()(c7)
    c8 = layers.Conv2D(256, kernel_size=3, strides=1, padding='valid')(c7) # 4
    c8 = layers.PReLU()(c8)
    c9 = layers.Conv2D(256, padding='valid', kernel_size=3, strides=1)(c8) # 2, 2, 256
    c9 = layers.PReLU()(c9)
    
    flat = layers.Flatten(name='flatten')(c9) # 2, 2, 256 = 1024 

    l1 = layers.Dense(1024)(flat) 
    l1 = layers.PReLU()(l1)
    if bn :
        l1 = layers.BatchNormalization()(l1)

    return keras.Model(inputs=inp, outputs=l1)

def backbone_adv(bn=True) :
    inp = keras.Input((64, 64, 5))
    c1 = layers.Conv2D(96, padding='same', strides=1, kernel_size=3)(inp) # 64
    c1 = layers.PReLU()(c1)
    c2 = layers.Conv2D(96, padding='same', kernel_size=3, strides=1, activation='tanh')(c1)  #64
    p1 = layers.AveragePooling2D((2, 2))(c2)  # 32
    c3 = layers.Conv2D(128, padding='same', strides=1, kernel_size=3)(p1)
    c3 = layers.PReLU()(c3)
    c4 = layers.Conv2D(128, padding='same', kernel_size=3, strides=1)(c3)  #32
    c4 = layers.PReLU(name='c4')(c4)
    p2 = layers.AveragePooling2D((2, 2))(c4)  # 16
    c5 = layers.Conv2D(256, padding='same', strides=1, kernel_size=3)(p2) #16
    c5 = layers.PReLU()(c5)
    c6 = layers.Conv2D(256, padding='same', kernel_size=3, strides=1)(c5)  #16
    c6 = layers.PReLU()(c6)
    p3 = layers.AveragePooling2D((2, 2))(c6) # 8
    c7 = layers.Conv2D(256, kernel_size=3, strides=1, padding='valid')(p3) # 6
    c7 = layers.PReLU()(c7)
    c8 = layers.Conv2D(256, kernel_size=3, strides=1, padding='valid')(c7) # 4
    c8 = layers.PReLU()(c8)
    c9 = layers.Conv2D(256, padding='valid', kernel_size=3, strides=1)(c8) # 2, 2, 256
    c9 = layers.PReLU()(c9)

    flat = layers.Flatten(name='flatten')(c9) # 2, 2, 256 = 1024

    l1 = layers.Dense(1024)(flat)
    l1 = layers.PReLU()(l1)
    if bn :
        l1 = layers.BatchNormalization()(l1)

    return keras.Model(inputs=inp, outputs=[l1, flat])

def mlp(input_shape=100):
    latent_input = keras.Input((input_shape))
    x = layers.Dense(512, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(5e-7), bias_regularizer=tf.keras.regularizers.l2(5e-7))(latent_input)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)
    x = layers.Dense(256, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(5e-7), bias_regularizer=tf.keras.regularizers.l2(5e-7))(x)
    return keras.Model(latent_input, x)

def mlp_adv(input_shape=1024) :
    latent_inp = keras.Input((input_shape))
    x = layers.BatchNormalization()(latent_inp)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(2, activation='softmax')(x)
    return keras.Model(latent_inp, x)

bn=True


model = simCLR(backbone(bn), mlp(1024))


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
    # RA   DEC   EB_V   ZPHOT   EBV
    return np.array([tup[1], tup[2], tup[7], max(tup[40], 1e-5), tup[35]])

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


#print(all_metas)
images = np.concatenate(all_images, axis=0)
metas = np.concatenate(all_metas, axis=0)  # 12*20k, 5

ra = metas[:, 0]
dec = metas[:, 1]
ebv = metas[:, 2]
z = np.log(metas[:, 3]+1)


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


weights_paths = ["../model_save/checkpoints_simCLR_UD/simCLR_cosmos_bnTrue_400.weights.h5", "../model_save/checkpoints_simCLR_UD_D/simCLR_cosmos_bnTrue_400.weights.h5", '../model_save/checkpoints_simCLR_UD_D_adv/simCLR_cosmos_bnTrue_400.weights.h5']
code_w = ['UD400', 'UD_D400', 'UD_D_adv400']

model = simCLR(backbone=backbone(), head=mlp(1024))
model(np.random.random((32, 64, 64, 5)))

for i, w in enumerate(weights_paths) :
    if i == 2 :
        model = simCLR_adv(backbone_adv(), mlp(1024), mlp_adv(1024))
        model(np.random.random((32, 64, 64, 5)))

    model.load_weights(w)

    
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
    plt.savefig("../plots/simCLR/tsne_redshift_base"+code_w[i]+".png")
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
    plt.savefig("../plots/simCLR/tsne_redshift_D_base"+code_w[i]+".png")
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
    plt.savefig("../plots/simCLR/tsne_redshift_UD_base"+code_w[i]+".png")
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
    plt.savefig("../plots/simCLR/tsne_ebv_base"+code_w[i]+".png")
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
    plt.savefig("../plots/simCLR/tsne_ebv_D_base"+code_w[i]+".png")
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
    plt.savefig("../plots/simCLR/tsne_ebv_UD_base"+code_w[i]+".png")
    plt.close()





