import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras as keras
from contrastiv_model import simCLR
import os
import gc
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
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

def mlp(input_shape=100):
    latent_input = keras.Input((input_shape))
    x = layers.Dense(512, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(5e-7), bias_regularizer=tf.keras.regularizers.l2(5e-7))(latent_input)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)
    x = layers.Dense(256, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(5e-7), bias_regularizer=tf.keras.regularizers.l2(5e-7))(x)
    return keras.Model(latent_input, x)

model = simCLR(backbone(), mlp(1024))
model(np.random.random((32, 64, 64, 5)))

model.load_weights("/lustre/fswork/projects/rech/dnz/ull82ct/astro/model_save/checkpoints_simCLR_UD_D/simCLR_cosmos_bnTrue_400.weights.h5")

extracteur = model.backbone


names = ["conv2d", "conv2d_1","conv2d_2","conv2d_3","conv2d_4","conv2d_5","conv2d_6","conv2d_7","conv2d_8"]


for l, name in enumerate(names) :
    weights = extracteur.get_layer(name).get_weights()  # SHAPE NB filters x H x W
    filters = weights[0]
    filters = np.transpose(filters, axes=(3, 0, 1, 2))
    filters = np.reshape(filters, (filters.shape[0], -1))
    angle_ref = np.ones((filters.shape[0], filters.shape[1]))
    angle_ref[:, -1] = 0
    direction_ref = np.zeros((filters.shape[0], filters.shape[1]))
    direction_ref[:, -1] = 1


    cosine_sim = np.sum(angle_ref * filters, axis=1) / (
            np.linalg.norm(angle_ref, axis=1) * np.linalg.norm(filters, axis=1) + 1e-8
    )

    signe_ref_sim = np.sign(np.sum(direction_ref * filters, axis=1) / (
            np.linalg.norm(direction_ref, axis=1) * np.linalg.norm(filters, axis=1) + 1e-8
    ))

    angles = np.arccos(cosine_sim) 
   
    x = np.cos(angles)
    y = np.sin(angles) * signe_ref_sim

    
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)

    kde = gaussian_kde(np.vstack([x, y]))
    densities = kde(np.vstack([x, y]))

    # Tracé de la distribution des points
    plt.figure(figsize=(8, 8))
    plt.plot(circle_x, circle_y, 'k--', label='Cercle unitaire')  # Cercle
    sc = plt.scatter(x, y, c=densities, cmap='viridis', label='filtres', s=100)  # Points colorés
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)

    plt.title("Distribution des angles des filtres sur le cercle unitaire")
    plt.xlabel("cos(θ)")
    plt.ylabel("sin(θ)")
    plt.axis('equal')
    plt.legend()
    plt.colorbar(sc, label='Densité')
    plt.grid(True)

    # Afficher
    plt.savefig("/lustre/fswork/projects/rech/dnz/ull82ct/astro/plots/simCLR/orthog_simCLR_filtres_"+str(l+1)+".png")
    plt.close()





    print(filters.shape)
