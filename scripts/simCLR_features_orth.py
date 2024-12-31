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

name = "simCLR_UD+D_400"

model.load_weights("/lustre/fswork/projects/rech/dnz/ull82ct/astro/model_save/checkpoints_simCLR_UD_D/simCLR_cosmos_bnTrue_400.weights.h5")

extracteur = model.backbone


outputs = [extracteur.get_layer("p_re_lu").output, extracteur.get_layer("conv2d_1").output, extracteur.get_layer("p_re_lu_1").output, 
                                                 extracteur.get_layer("c4").output, extracteur.get_layer("p_re_lu_2").output, extracteur.get_layer("p_re_lu_3").output,
                                                 extracteur.get_layer("p_re_lu_4").output, extracteur.get_layer("p_re_lu_5").output, extracteur.get_layer("p_re_lu_6").output,
                                                 extracteur.get_layer("flatten").output, extracteur.get_layer("p_re_lu_7").output]




fig1, ax1 = plt.subplots(nrows=3, ncols=3, figsize=(10, 10)) ## DENSITY
fig2, ax2 = plt.subplots(nrows=3, ncols=3, figsize=(10, 10)) ## REDSHIFT
fig3, ax3 = plt.subplots(nrows=3, ncols=3, figsize=(10, 10)) ## REDSHIFT


for l, out in enumerate(outputs) :

    row = l % 3
    col = l // 3

    all_layer_model = keras.Model(extracteur.input, out)    

    all_cosine_sim = []
    all_directions = []
    all_z_vals = []
    all_angles = []
    all_surveys = []

    average_vecs = []
    nb_samples = []

    ext_name = ["spec_D", "spec_UD"]
    for m, extension in enumerate(["spec_UD.npz", "spec_D.npz"]) :
        from pathlib import Path

        dir_path = Path("/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/spec/")
        paths = [file for file in dir_path.rglob(f"*{extension}")]

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

            angles = np.arccos(cosine_sim) 
            all_angles.append(angles)
            all_directions.append(direction)
            if m == 0 : 
                all_surveys.append(np.zeros((angles.shape[0])))
            else :
                all_surveys.append(np.ones((angles.shape[0])))
            #all_cosine_sim.append(cosine_sim)
            #all_directions.append(direction)


    average_o = np.sum([average_vecs[i] * nb_samples[i] for i in range(len(nb_samples))]) / np.sum(nb_samples)

    average_sim = np.sum(angle_ref[0] * average_o) / (
        np.linalg.norm(angle_ref[0]) * np.linalg.norm(average_o) + 1e-8
    )
    average_sign = np.sign(np.sum(direction_ref[0] * average_o) / (
        np.linalg.norm(direction_ref[0]) * np.linalg.norm(average_o) + 1e-8
    ))

        # Calcul des coordonnées x, y pour chaque point sur le cercle
    angles = np.concatenate(all_angles, axis=0)
    direction = np.concatenate(all_directions, axis=0)
    surveys = np.concatenate(all_surveys, axis=0)
    
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
    ax1[row, col].plot(circle_x, circle_y, 'k--', label='Cercle unitaire')  # Cercle
    sc = ax1[row, col].scatter(x, y, c=densities, cmap='viridis', label='Représentations', s=100)  # Points colorés
    ax1[row, col].axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax1[row, col].axvline(0, color='gray', linestyle='--', linewidth=0.5)
    ax1[row, col].set_title("conv"+str(l+1))

    ax1[row, col].set_xlabel("cos(θ)")
    ax1[row, col].set_ylabel("sin(θ)")
    ax1[row, col].set_axis('equal')
    ax1[row, col].grid(False)



    ax2[row, col].plot(circle_x, circle_y, 'k--', label='Cercle unitaire')  # Cercle
    sc = ax2[row, col].scatter(x, y, c=z_vals, cmap='viridis', label='Représentations', s=100)  # Points colorés
    ax2[row, col].axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax2[row, col].axvline(0, color='gray', linestyle='--', linewidth=0.5)
    ax2[row, col].set_title("conv"+str(l+1))

    ax2[row, col].set_xlabel("cos(θ)")
    ax2[row, col].set_ylabel("sin(θ)")
    ax2[row, col].set_axis('equal')
    if row == 0 and col == 0 : 
        ax2[0, 0].colorbar(sc, label='Redshift (Z)')
    ax2[row, col].grid(False)





    ax3[row, col].plot(circle_x, circle_y, 'k--', label='Cercle unitaire')  # Cercle
    ax3[row, col].scatter(x, y, c=surveys, label='Représentations', s=100)  # Points colorés
    ax3[row, col].axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax3[row, col].axvline(0, color='gray', linestyle='--', linewidth=0.5)
    ax3[row, col].set_title("conv"+str(l+1))

    ax3[row, col].set_xlabel("cos(θ)")
    ax3[row, col].set_ylabel("sin(θ)")
    ax3[row, col].set_axis('equal')
    if row == 0 and col == 0 : 
        ax3[0, 0].legend()
    ax3[row, col].grid(False)

        # Afficher
fig1.savefig("/lustre/fswork/projects/rech/dnz/ull82ct/astro/plots/simCLR/orthog_"+name+".png")
plt.close(fig1)

        # Afficher
fig2.savefig("/lustre/fswork/projects/rech/dnz/ull82ct/astro/plots/simCLR/orthog_"+name+"_z.png")
plt.close(fig2)

fig3.savefig("/lustre/fswork/projects/rech/dnz/ull82ct/astro/plots/simCLR/orthog_"+name+"_survey.png")
plt.close(fig3)












