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


outputs = [extracteur.get_layer("p_re_lu").output, extracteur.get_layer("conv2d_1").output, extracteur.get_layer("p_re_lu_1").output, 
                                                 extracteur.get_layer("c4").output, extracteur.get_layer("p_re_lu_2").output, extracteur.get_layer("p_re_lu_3").output,
                                                 extracteur.get_layer("p_re_lu_4").output, extracteur.get_layer("p_re_lu_5").output, extracteur.get_layer("p_re_lu_6").output,
                                                 extracteur.get_layer("flatten").output, extracteur.get_layer("p_re_lu_7").output]





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


for l, out in enumerate(outputs) :

    all_layer_model = keras.Model(extracteur.input, out)    

    all_cosine_sim = []
    all_directions = []
    all_z_vals = []

    average_vecs = []
    nb_samples = []

    for path in paths :

        data = np.load(path, allow_pickle=True)
        images = data["cube"][:, :, :, :5]
        meta = data["info"]
        z_vals = np.array([m[40] for m in meta])
        all_z_vals.append(z_vals)
        images = np.sign(images)*(np.sqrt(np.abs(images)+1)-1 )

        outs = []
        i=0
        inf_batch = 64
        while i < len(images) :
            
            if i +inf_batch < len(images) :
                o= all_layer_model(images[i:i+inf_batch])
                
            else : 
                o = all_layer_model(images[i:])

            outs.append(o.numpy())
            i+=inf_batch
            print(i)
                
            del o
            gc.collect()


        o = np.concatenate(outs, axis=0)

        o = np.reshape(o, (o.shape[0], -1))  # B, X  ==> des grands vecteurs de dimension x
        average_o = np.mean(o, axis=0)
        
        average_vecs.append(average_o)
        nb_samples.append(len(o))        
        
        
        coss = []
        directions = []
        
        i = 0
        inf_batch = 64
        
        while i < o.shape[0] :
            if i + inf_batch < o.shape[0] :
                batch = o[i:i+inf_batch]
            else :
                batch = o[i:]

            angle_ref = np.ones((batch.shape[0], batch.shape[1]))
            angle_ref[-1] = 0 # 1 1 1 1 1 1 1 0
            direction_ref = np.zeros((batch.shape[0], batch.shape[1]))
            direction_ref[:, -1] = 1 ###  0 0 0 0 0 0 0 0  1


            cosine_sim = np.sum(angle_ref * batch, axis=1) / (
                np.linalg.norm(angle_ref, axis=1) * np.linalg.norm(batch, axis=1) + 1e-8
            )

            signe_ref_sim = np.sum(direction_ref * batch, axis=1) / (
                np.linalg.norm(direction_ref, axis=1) * np.linalg.norm(batch, axis=1) + 1e-8
            )

        
            coss.append(cosine_sim)
            direc = np.sign(signe_ref_sim)
            #print(direc)
            directions.append(direc)
            i+=inf_batch
            print(i)


            del batch, direc, signe_ref_sim#, vec_prod, ref_vec
            gc.collect()

        
        
        cosine_sim = np.concatenate(coss, axis=0)
        direction = np.concatenate(directions, axis=0)
        print(cosine_sim)

        angles = np.arccos(cosine_sim) 
        print(angles)
        print("ANGLE SHAPE :",angles.shape)
        all_cosine_sim.append(cosine_sim)
        all_directions.append(direction)


    average_o = np.sum([average_vecs[i] * nb_samples[i] for i in range(len(nb_samples))]) / np.sum(nb_samples)

    average_sim = np.sum(angle_ref[0] * average_o) / (
                np.linalg.norm(angle_ref[0]) * np.linalg.norm(average_o) + 1e-8
        )
    average_sign = np.sign(np.sum(direction_ref[0] * average_o) / (
                np.linalg.norm(direction_ref[0]) * np.linalg.norm(average_o) + 1e-8
            ))

    # Calcul des coordonnées x, y pour chaque point sur le cercle
    x = np.cos(angles)
    y = np.sin(angles) * direction

    origin_angle = np.arccos(average_sim)   # L'angle du point de référence (origine)
    origin_x = np.cos(origin_angle)
    origin_y = np.sin(origin_angle) * average_sign
    # Tracé du cercle unitaire
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)

    kde = gaussian_kde(np.vstack([x, y]))
    densities = kde(np.vstack([x, y]))

    # Tracé de la distribution des points
    plt.figure(figsize=(8, 8))
    plt.plot(circle_x, circle_y, 'k--', label='Cercle unitaire')  # Cercle
    sc = plt.scatter(x, y, c=densities, cmap='viridis', label='Représentations', s=100)  # Points colorés
    plt.scatter(origin_x, origin_y, color='red', label='Vecteur moyen', s=100, edgecolors='black', marker='X')
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
    sc = plt.scatter(x, y, c=z_vals, cmap='viridis', label='Représentations', s=100)  # Points colorés
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












