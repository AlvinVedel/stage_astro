import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
from tensorflow.keras import layers
import tensorflow.keras as keras
from byol_model import BYOL
import matplotlib.pyplot as plt
#from scipy.stats import gaussian_kde
from matplotlib import cm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def mlp(input_shape=2048):
    latent_input = keras.Input((input_shape))
    x = layers.Dense(4096, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(5e-7))(latent_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(256, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(5e-7))(x)
    return keras.Model(latent_input, x)

def ResNet50():
    inp = tf.keras.Input((64, 64, 9))
    c1 = layers.Conv2D(64, kernel_size=7, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-7))(inp)  # 32, 32
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.ReLU()(c1)
    c1 = layers.MaxPooling2D((2, 2))(c1)  # 16, 16

    r1 = bottleneck_block(c1, 64, downsample=True)
    r1 = bottleneck_block(r1, 64)
    r1 = bottleneck_block(r1, 64)

    r2 = bottleneck_block(r1, 128, 2, True)   #8, 8
    r2 = bottleneck_block(r2, 128)
    r2 = bottleneck_block(r2, 128)
    r2 = bottleneck_block(r2, 128)

    r3 = bottleneck_block(r2, 256, 2, True) # 4, 4
    r3 = bottleneck_block(r3, 256)
    r3 = bottleneck_block(r3, 256)
    r3 = bottleneck_block(r3, 256)
    r3 = bottleneck_block(r3, 256)
    r3 = bottleneck_block(r3, 256)

    r4 = bottleneck_block(r3, 512, 2, True)  #2, 2
    r4 = bottleneck_block(r4, 512)
    r4 = bottleneck_block(r4, 512)

    x = layers.GlobalAveragePooling2D()(r4)  # 512
    return tf.keras.Model(inp, x)

def bottleneck_block(x, filters, stride=1, downsample=False) :
    identity = x
    if downsample :
        identity = layers.Conv2D(filters*4, (1, 1), strides=stride, padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(5e-7))(identity)
        identity = layers.BatchNormalization()(identity)

    x = conv_block(x, filters, kernel_size=1, stride=stride)
    x = conv_block(x, filters)

    x = layers.Conv2D(filters*4, (1, 1), use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(5e-7))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, identity])
    x = layers.ReLU()(x)
    return x

def conv_block(x, filters, kernel_size=3, stride=1, padding='same'):
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding=padding, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(5e-7))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


folder_path = "/lustre/fswork/projects/rech/kof/uve94ap/CUBES_HSC/PHOT/DEEP2"

file_paths = []

for file_name in os.listdir(folder_path):
    if file_name.endswith('.npz'):
        file_path = os.path.join(folder_path, file_name)
        file_paths.append(file_path)
    

indices = np.arange(len(file_paths))
import random
random.shuffle(indices)

indices = indices[:8]

all_images = []
all_metas = []
for i in range(8) :
    ind = indices[i]
    print("FILE ", i, file_paths[ind])
    data = np.load(file_paths[ind], allow_pickle=True)
    images = np.sign(data['cube'])*np.sqrt(np.abs(data["cube"]+1))-1 
    meta = data["info"].item()
    for key in meta :
        print(key, meta[key][:10])
    all_images.append(images)
    all_metas.append(meta)

images = np.concatenate(all_images, axis=0)

all_z = []
all_ra = []
all_dec = []
all_ebv = []
for meta in all_metas :
    all_z.append(meta["ZSPEC"])
    all_ra.append(meta["RA"])
    all_dec.append(meta["DEC"])
    all_ebv.append(meta["EBV"])

z = np.concatenate(all_z, axis=0)
ra = np.concatenate(all_ra, axis=0)
dec = np.concatenate(all_dec, axis=0)
ebv = np.concatenate(all_ebv, axis=0)


"""
data = np.load("/home/barrage/HSC_READY_CUBES/XMM_SHALLOW_SPECTRO.npz", allow_pickle=True)
images = np.sign(data['cube'])*np.sqrt(np.abs(data["cube"]+1))-1 
meta = data["info"].item()

z = meta["ZSPEC"]
ra = meta["RA"]
dec = meta["DEC"]
ebv = meta["EBV"]
survey = meta["SURVEY"]

surveys = ["VIPERS"]
surveys = [s.encode("utf-8") for s in surveys]

indexes = np.where(np.isin(survey, surveys))[0]
ebv = ebv[indexes]
z = z[indexes]
ra = ra[indexes]
dec = dec[indexes]
images = images[indexes]

dist_radec = np.sqrt(ra**2 + dec**2)
print(dist_radec)
print(dist_radec.shape)

"""

model = BYOL(ResNet50(), ResNet50(), mlp(2048), mlp(2048), mlp(256))
model(np.random.random((32, 64, 64, 9)))
model.load_weights("byol_d2_epoch120.h5")

extractor = model.online_backbone

features = extractor.predict(images)
print(features.shape)

tsne = TSNE(n_components=2, random_state=42)
data_tsne = tsne.fit_transform(features)
print("tsne ended")


print(data_tsne.shape, z.shape, ra.shape, dec.shape, ebv.shape)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=z, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Redshift (z)') 
plt.title("t-SNE features byol colorées par Z")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.savefig("tsne_redshift.png")
plt.close()



plt.figure(figsize=(10, 8))
scatter = plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=ra, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Ra') 
plt.title("t-SNE features byol colorées par RA")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.savefig("tsne_ra.png")
plt.close()

plt.figure(figsize=(10, 8))
scatter = plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=dec, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Dec') 
plt.title("t-SNE features byol colorées par DEC")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.savefig("tsne_dec.png")
plt.close()


plt.figure(figsize=(10, 8))
scatter = plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=ebv, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='ebv') 
plt.title("t-SNE features byol colorées par EBV")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.savefig("tsne_ebv.png")
plt.close()



