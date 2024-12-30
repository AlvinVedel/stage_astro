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

def backbone() :
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
model.load_weights("simCLR_final_cosmos.weights.h5")

extracteur = model.backbone

all_layer_model = keras.Model(extracteur.input, [extracteur.get_layer("p_re_lu").output, extracteur.get_layer("conv2d_1").output, extracteur.get_layer("p_re_lu_1").output, 
                                                 extracteur.get_layer("c4").output, extracteur.get_layer("p_re_lu_2").output, extracteur.get_layer("p_re_lu_3").output,
                                                 extracteur.get_layer("p_re_lu_4").output, extracteur.get_layer("p_re_lu_5").output, extracteur.get_layer("p_re_lu_6").output,
                                                 extracteur.get_layer("flatten").output, extracteur.get_layer("p_re_lu_7").output,])




dir_path = "/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/spec/"
name='spec_UD'
extension = "spec_UD.npz"

paths = []

for dire in dir_path :
    for root, dirs, files in os.walk(dire):
                for file in files:
                    if file.endswith(tuple(extension)):
                        filepath = os.path.join(root, file)
                        paths.append(filepath)


all_cosine_sim = []
all_z_vals = []
for path in paths :

    data = np.load("my_cube.npz", allow_pickle=True)
    images = data["cube"][:, :, :, :5]
    meta = data["info"]
    z_vals = np.array([m[40] for m in meta])
    all_z_vals.append(z_vals)
    images = np.sign(images)*(np.sqrt(np.abs(images)+1)-1 )

    o1s, o2s, o3s, o4s, o5s, o6s, o7s, o8s, o9s, o10s, o11s = [], [], [], [], [], [], [], [], [], [], []

    i=0
    inf_batch = 128
    while i < len(images) :
        if i +inf_batch < len(images) :
            o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11 = all_layer_model(images[i:i+inf_batch])
        else : 
            o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11 = all_layer_model(images[i:])
        #print(o1)
        print(i)
        i+=inf_batch
        o1s.append(o1.numpy())
        o2s.append(o2.numpy())
        o3s.append(o3.numpy())
        o4s.append(o4.numpy())
        o5s.append(o5.numpy())
        o6s.append(o6.numpy())
        o7s.append(o7.numpy())
        o8s.append(o8.numpy())
        o9s.append(o9.numpy())
        o10s.append(o10.numpy())
        o11s.append(o11.numpy())

        del o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11
        gc.collect()

        #gc.collect()

    o1 = np.concatenate(o1s, axis=0)
    o2 = np.concatenate(o2s, axis=0)
    o3 = np.concatenate(o3s, axis=0)
    o4 = np.concatenate(o4s, axis=0)
    o5 = np.concatenate(o5s, axis=0)
    o6 = np.concatenate(o6s, axis=0)
    o7 = np.concatenate(o7s, axis=0)
    o8 = np.concatenate(o8s, axis=0)
    o9 = np.concatenate(o9s, axis=0)
    o10 = np.concatenate(o10s, axis=0)
    o11 = np.concatenate(o11s, axis=0)


    

    outputs = [o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11]

    def cosine_similarity(a, b) :
        dp = a @ b
        norm_a = np.linalg.norm(a, axis=1)
        norm_b = np.linalg.norm(b, axis=1) 

    for l, o in enumerate(outputs) :
        print(o.shape)
        o = np.reshape(o, (o.shape[0], -1))  # B, X  ==> des grands vecteurs de dimension x
        average_o = np.mean(o, axis=0)
        print(o.shape)

        #o = tf.convert_to_tensor(o)
        #average_o = tf.convert_to_tensor(average_o)
        coss = []
        
        i = 0
        inf_batch = 128
        while i < o.shape[0] :
            if i + inf_batch < o.shape[0] :
                batch = o[i:i+inf_batch]
            else :
                batch = o[i:]


            average_o_expanded = np.tile(average_o, (batch.shape[0], 1))
            cosine_sim = np.sum(average_o_expanded * batch, axis=1) / (
                np.linalg.norm(average_o_expanded, axis=1) * np.linalg.norm(batch, axis=1) + 1e-8
            )
            #print(cosine_sim)
            coss.append(cosine_sim)
            i+=inf_batch
        
        cosine_sim = np.concatenate(coss, axis=0)
        all_cosine_sim.append(cosine_sim)
    #print(cosine_sim)

cosine_sim = np.concatenate(all_cosine_sim, axis=0)
z_vals = np.concatenate(all_z_vals, axis=0)

angles = np.arccos(cosine_sim)
print("ANGLE SHAPE :",angles.shape)

    # Calcul des coordonnées x, y pour chaque point sur le cercle
x = np.cos(angles)
y = np.sin(angles)

origin_angle = 0  # L'angle du point de référence (origine)
origin_x = np.cos(origin_angle)
origin_y = np.sin(origin_angle)
    # Tracé du cercle unitaire
theta = np.linspace(0, 2 * np.pi, 100)
circle_x = np.cos(theta)
circle_y = np.sin(theta)

kde = gaussian_kde(np.vstack([x, y]))
densities = kde(np.vstack([x, y]))

    # Tracé de la distribution des points
plt.figure(figsize=(8, 8))
plt.plot(circle_x, circle_y, 'k--', label='Cercle unitaire')  # Cercle
sc = plt.scatter(x, y, c=densities, cmap='viridis', label='Représentations', s=30)  # Points colorés
plt.scatter(origin_x, origin_y, color='red', label='Point origine', s=50, edgecolors='black', marker='X')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)

    # Personnalisation
plt.title("Distribution des angles sur le cercle unitaire")
plt.xlabel("cos(θ)")
plt.ylabel("sin(θ)")
plt.axis('equal')
plt.legend()
plt.colorbar(sc, label='Densité')
plt.grid(True)

    # Afficher
plt.savefig("/lustre/fswork/projects/rech/dnz/ull82ct/astro/plots/simCLR/orthog_simCLR_"+str(l+1)+".png")
plt.close()



    # Tracé de la distribution des points
plt.figure(figsize=(8, 8))
plt.plot(circle_x, circle_y, 'k--', label='Cercle unitaire')  # Cercle
sc = plt.scatter(x, y, c=z_vals, cmap='viridis', label='Représentations', s=30)  # Points colorés
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)

    # Personnalisation
plt.title("Distribution des angles sur le cercle unitaire")
plt.xlabel("cos(θ)")
plt.ylabel("sin(θ)")
plt.axis('equal')
plt.legend()
plt.colorbar(sc, label='Redshift (Z)')
plt.grid(True)

    # Afficher
plt.savefig("/lustre/fswork/projects/rech/dnz/ull82ct/astro/plots/simCLR/orthog_simCLR_"+str(l+1)+"z.png")
plt.close()












