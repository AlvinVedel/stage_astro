import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from contrastiv_model import simCLR
import matplotlib.pyplot as plt

import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'



def inception_block(input):
    c1 = layers.Conv2D(101, activation='relu', kernel_size=1, strides=1, padding='same')(input)
    c2 = layers.Conv2D(101, activation='relu', kernel_size=1, strides=1, padding='same')(input)
    c3 = layers.Conv2D(101, activation='relu', kernel_size=1, strides=1, padding='same')(input)

    c4 = layers.Conv2D(156, activation='relu', kernel_size=3, strides=1, padding='same')(c1)
    c5 = layers.Conv2D(156, activation='relu', kernel_size=5, strides=1, padding='same')(c2)
    c6 = layers.AveragePooling2D((2, 2), strides=1, padding='same')(c3)

    c7 = layers.Conv2D(109, activation='relu', kernel_size=1, strides=1, padding='same')(input)

    conc = layers.Concatenate(axis=-1)([c4, c5, c6, c7])   # 156 + 156 + 101 + 109 = 522
    return conc

def create_model(with_ebv=False, nbins=400) :

    input_img = keras.Input((64, 64, 5))
    #input_ebv = keras.Input((1,))
    conv1 = layers.Conv2D(96, kernel_size=5, activation='relu', strides=1, padding='same', name='c1')(input_img)
    conv2 = layers.Conv2D(96, kernel_size=3, activation='tanh', strides=1, padding='same', name='c2')(conv1)
    avg_pool = layers.AveragePooling2D(pool_size=(2, 2), strides=2, name='p1')(conv2)  # batch, 32, 32, 96

    incep1 = inception_block(avg_pool)
    incep2 = inception_block(incep1)
    incep3 = inception_block(incep2)

    avg_pool2 = layers.AveragePooling2D((2, 2), strides=2, name='p2')(incep3) # 16, 16, 522

    incep4 = inception_block(avg_pool2)
    incep5 = inception_block(incep4)

    avg_pool3 = layers.AveragePooling2D((2, 2), strides=2, name='p3')(incep5)  # 8, 8, 522

    incep6 = inception_block(avg_pool3) 

    conv3 = layers.Conv2D(96, kernel_size=3, activation='relu', padding='valid', strides=1, name='c3')(incep6)  # 6, 6, 96
    conv4 = layers.Conv2D(96, kernel_size=3, activation='relu', padding='valid', strides=1, name='c4')(conv3)  # 4, 4, 96
    conv5 = layers.Conv2D(96, kernel_size=3, activation='relu', padding='valid', strides=1, name='c5')(conv4)   # 2, 2, 96
    avg_pool4 = layers.AveragePooling2D((2, 2), strides=1, name='p4')(conv5)   # 1, 1, 96
    resh = layers.Reshape(target_shape=(96,))(avg_pool4) # batch, 96
    if with_ebv : 
        ebv_input = keras.Input((1))
        resh = layers.Concatenate(axis=-1)([resh, ebv_input])
    #conc = layers.Concatenate()([resh, input_ebv])
    l1 = layers.Dense(1024, activation='relu', name='l1')(resh)

    l2 = layers.Dense(1024, activation='relu', name='l2')(l1)
    l3 = layers.Dense(512, activation='tanh', name='l3')(l1)

    pdf = layers.Dense(nbins, activation='softmax', name='pdf')(l2)
    regression = layers.Dense(1, activation='relu', name='reg')(l3)
    if with_ebv :
        model = keras.Model(inputs=[input_img, ebv_input], outputs=[pdf, regression])
    else :
        model = keras.Model(inputs=[input_img], outputs=[pdf, regression])
    return model

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

def regression_head(input_shape=1024) :
    inp = keras.Input((input_shape))
    l1 = layers.Dense(1024)(inp)
    l1 = layers.PReLU()(l1)
    l2 = layers.Dense(1024)(l1)
    l2 = layers.PReLU()(l2)
    reg = layers.Dense(1, activation='linear')(l2)
    return keras.Model(inp, reg)

def output_head(input_shape=1024, nbins=400) :
    inp = keras.Input((input_shape))
    l1 = layers.Dense(1024)(inp)
    l1 = layers.PReLU()(l1)
    l2 = layers.Dense(1024)(l1)
    l2 = layers.PReLU()(l2)
    pdf = layers.Dense(nbins, activation='linear', name='pdf')(l2)
    l3 = layers.Dense(512, activation='tanh')(l2)
    reg = layers.Dense(1, activation='linear')(l3)
    return keras.Model(inp, [pdf, reg])


class FineTuneModel(keras.Model) :
    def __init__(self, back, head, train_back=False, adv=False) :
        super(FineTuneModel, self).__init__()
        self.backbone = back
        self.head = head
        self.adv = adv
        self.train_back = train_back

    def call(self, inputs, training=True) :
        if self.adv :
            latent, flat = self.backbone(inputs, training=self.train_back)
        else :
            latent = self.backbone(inputs, training=self.train_back)
        pred = self.head(latent, training=training)
        return pred


base_path = "/lustre/fswork/projects/rech/dnz/ull82ct/astro/"






data_frame = {"name":[], "inf base":[], "finetune base" : [],
               "bias [0, 0.6[":[], "bias [0.6, 2[":[], 'bias [2, 4[':[], 'bias [4, 6[':[],
               "smad [0, 0.6[":[], "smad [0.6, 2[":[], 'smad [2, 4[':[], 'smad [4, 6[':[],
               "oult [0, 0.6[":[], "oult [0.6, 2[":[], 'oult [2, 4[':[], 'oult [4, 6[':[]}


directory = base_path+"data/spec/"

def z_med(probas, bin_central_values) :
    cdf = np.cumsum(probas)
    index = np.argmax(cdf>=0.5)
    return bin_central_values[index]


path_memory = {}
bins_edges = np.concatenate([np.linspace(0, 4, 381), np.linspace(4, 6, 21)[1:]], axis=0)
# bins_edges = np.linspace(0, 6, 300)
bins_centres = (bins_edges[1:] + bins_edges[:-1])/2

for j, inf_base in enumerate(["spec_UD", "cos2020_UD", "spec_D", "cos2020_D"]) :

    from pathlib import Path

    dir_path = Path("/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/spec/")
    extension = inf_base+'.npz'
    npz_files = [file for file in dir_path.rglob(f"*{extension}")]
    path_memory[inf_base] = npz_files

    
    base_true_z = []
    inf_base_preds = {}

    for file in npz_files :
        def extract_z(tup) :
            return tup[40]
                        

        data = np.load(str(file), allow_pickle=True)
        images = data["cube"][..., :5]
        meta = data["info"]
        z = np.array([extract_z(m) for m in meta])
        base_true_z.append(z)

        for i, finetune_base in enumerate(["b1_1", "b2_1", "b3_1"]) :   #### POUR CHAQUE CONDITION D ENTRAINEMENT ON VA AVOIR UN SUBPLOT
            base_liste = ["spec_UD", "cos2020_UD", "spec_D", "cos2020_D"]
            print("finetune base", finetune_base)

            fig1, ax1 = plt.subplots(nrows=len(base_liste), ncols=7, figsize=(20, 20)) # pour 1 treyer + 3 simCLR dans 2 conditions modèles   ==>   heatmap
            fig2, ax2 = plt.subplots(nrows=len(base_liste), ncols=7, figsize=(20, 20)) # ===> BIAS
            fig3, ax3 = plt.subplots(nrows=len(base_liste), ncols=7, figsize=(20, 20)) # ===> SMAD
            fig4, ax4 = plt.subplots(nrows=len(base_liste), ncols=7, figsize=(20, 20)) # ===> OUTL

            if finetune_base not in inf_base_preds :
                inf_base_preds[finetune_base] = {}
            

                #npz_files = [f for f in os.listdir(directory) if f.endswith(inf_base+'.npz')]   ## Récupère les fichiers sur lesquels inférer

            simbases = ["fullsupervised", "UD400_classif", "UD_D400_classif" "UD800_classif","UD_D800_classif"]
            conds = ["HeadOnly", "ALL"]
            iter = 0
                #model_liste = ["simCLR_finetune/simCLR_finetune_"+cond+"_base="+finetune_base+"_model="+sim_base+".weights.h5"]
            for k in range(len(simbases)*2-1) :
                if k == 0 :
                    #model_name = base_path+"model_save/checkpoints_supervised/treyer_supervised_"+finetune_base+".weights.h5"
                    model_name = "../model_save/checkpoints_supervised/backbone_supervised_"+inf_base+".weights.h5"
                    tag_name = "supervised"
                    treyer = True

                    model = create_model()
                    model(np.random.random((32, 64, 64, 5)))

                else :

                    cond = "HeadOnly" if (iter)%2==0 else "ALL"
                    sim_base = simbases[iter//2]
                    #model_name = base_path+"model_save/checkpoints_simCLR_finetune/simCLR_finetune_"+cond+"_base="+finetune_base+"_model="+sim_base+".weights.h5"
                    model_name = "../model_save/checkpoints_simCLR_finetune/simCLR_finetune_"+cond+"_base="+finetune_base+"_model="+sim_base+".weights.h5"
                    tag_name = 'sim_'+sim_base+"_"+cond
                    treyer = True
                    iter+=1
                    model = FineTuneModel(backbone(True), head=output_head(1024))
                    model(np.random.random((32, 64, 64, 5)))  
                #print("the model name isss :", model_name)
                #print(os.getcwd())
                print(tag_name)
                if tag_name not in inf_base_preds[finetune_base] :
                    inf_base_preds[finetune_base][tag_name] = []

                model.load_weights(model_name)
                                        
                probas, reg = model.predict(images)
                z_meds = np.array([z_med(p, bins_centres) for p in probas])
                       
                inf_base_preds[finetune_base][tag_name] = np.concatenate([inf_base_preds[finetune_base][tag_name] ,z_meds], axis=0)


        
                            

                        

                        #### MATRICE DE CHALEUR
                        from scipy.stats import gaussian_kde
                        import matplotlib.pyplot as plt

                        xy = np.vstack([true_z, pred_z])
                        density = gaussian_kde(xy)(xy)


                        ax1[j, k].scatter(true_z, pred_z, c=density, cmap='hot', s=5)
                        ax1[j, k].set_ylim((-1, 6))
                        ax1[j, k].set_xlim((-1, 6))
                        ax1[j, k].set_xlabel("true Z")
                        ax1[j, k].set_ylabel("pred Z")

                        if j == 0 :
                            ax1[j, k].set_title(tag_name)

                        if k == 0:
                            ax1[j, k].text(-1.5, 0.5, inf_base, fontsize=12, ha='center', va='center', rotation=90)


                        """
                        plt.scatter(true_z, pred_z, c=density, cmap='hot', s=5)
                        plt.colorbar(label='density')
                        plt.xlabel("true Z")
                        plt.ylabel("pred Z")
                        plt.xlim((-1, 6))
                        plt.ylim((-1, 6))
                        plt.title("prediction density heatmap")
                        plt.savefig(base_path+"plots/simCLR/density_heatmap_"+model_name+".png")
                        plt.close()
                        """


                        def delta_z(z_pred, z_spec) :
                            return (z_pred - z_spec) / (1 + z_spec)

                        #### CALCUL DES METRIQUES ASTRO 
                        bigbins_edges = np.linspace(0, 6, 24)
                        megabins_edges = np.array([0, 0.6, 2, 4, 6])

                        deltas_z = (pred_z - true_z) / (1 + true_z) 

                        ### LES PLOTS

                        bias = np.zeros((len(bigbins_edges)-1))
                        smad = np.zeros((len(bigbins_edges)-1))
                        outl = np.zeros((len(bigbins_edges)-1))



                        for ij in range(len(bigbins_edges)-1) :
                            inds = np.where((true_z>=bigbins_edges[ij]) & (true_z<bigbins_edges[ij+1]))

                            selected_deltas = deltas_z[inds]

                            bias[ij] = np.mean(selected_deltas)

                            median_delta_z_norm = np.median(selected_deltas)
                            mad = np.median(np.abs(selected_deltas - median_delta_z_norm))
                            sigma_mad = 1.4826 * mad
                            smad[ij] = sigma_mad

                            outliers = np.abs(selected_deltas) > 0.05
                            fraction_outliers = np.sum(outliers) / (len(selected_deltas)+1e-6)
                            outl[ij] = fraction_outliers
                            


                        bins_centres = (bigbins_edges[1:] + bigbins_edges[:-1])/2



                        ax2[j, k].plot(bins_centres, bias)
                        ax2[j, k].set_xlabel("redshift")
                        ax2[j, k].set_ylabel("prediction bias")

                        if j == 0 :
                            ax2[j, k].set_title(tag_name)

                        if k == 0:
                            ax2[j, k].text(-1.5, 0.5, inf_base, fontsize=12, ha='center', va='center', rotation=90)

                        """
                        plt.plot(bins_centres, bias)
                        plt.xlabel("Z")
                        plt.ylabel("prediction bias")
                        plt.title("prediction bias for "+model_name)
                        plt.savefig(base_path+"plots/simCLR/bias_"+model_name+".png")
                        plt.close()
                        """

                        ax3[j, k].plot(bins_centres, smad)
                        ax3[j, k].set_xlabel("redshift")
                        ax3[j, k].set_ylabel("sigma MAD")

                        if j == 0 :
                            ax3[j, k].set_title(tag_name)

                        if k == 0:
                            ax3[j, k].text(-1.5, 0.5, inf_base, fontsize=12, ha='center', va='center', rotation=90)

                        """
                        plt.plot(bins_centres, smad)
                        plt.xlabel("Z")
                        plt.ylabel("sMAD")
                        plt.title("Sigma MAD for "+model_name)
                        plt.savefig(base_path+"plots/simCLR/smad_"+model_name+".png")
                        plt.close()
                        """

                        ax4[j, k].plot(bins_centres, smad)
                        ax4[j, k].set_xlabel("redshift")
                        ax4[j, k].set_ylabel("outlier fraction")

                        if j == 0 :
                            ax4[j, k].set_title(tag_name)

                        if k == 0:
                            ax4[j, k].text(-1.5, 0.5, inf_base, fontsize=12, ha='center', va='center', rotation=90)

                        """
                        plt.plot(bins_centres, outl)
                        plt.xlabel("Z")
                        plt.ylabel("outlier fraction")
                        plt.title("outlier fraction for "+model_name)
                        plt.savefig(base_path+"plots/simCLR/outl_"+model_name+".png")
                        plt.close()
                        """


                        bias = np.zeros((len(megabins_edges)-1))
                        smad = np.zeros((len(megabins_edges)-1))
                        outl = np.zeros((len(megabins_edges)-1))
                        #### LES VALEURS DE TABLEAU :
                        for ij in range(len(megabins_edges)-1) :
                            inds = np.where((true_z>=megabins_edges[ij]) & (true_z<megabins_edges[ij+1]))

                            selected_deltas = deltas_z[inds]

                            bias[ij] = np.mean(selected_deltas)

                            median_delta_z_norm = np.median(selected_deltas)
                            mad = np.median(np.abs(selected_deltas - median_delta_z_norm))
                            sigma_mad = 1.4826 * mad
                            smad[ij] = sigma_mad

                            outliers = np.abs(selected_deltas) > 0.05
                            fraction_outliers = np.sum(outliers) / (len(selected_deltas)+1e-6)
                            outl[ij] = fraction_outliers

                        print(" ------------------   MODEL = ", tag_name, finetune_base, " --------------")
                        print("RESULTS ON MEGABINS EDGES :")
                        print("PLAGES : [0, 0.6]     [0.6, 2]    [2, 4]     [4, 6]")
                        print("BIAS :", bias)
                        print("SMAD :", smad)
                        print("OUTL :", outl)

                        print("END RESULTS FOR", model_name)
                        #model.backbone.save_weights("sdss_backbone.weights.h5")
                        #model.classifier.save_weights("sdss_classifier.weights.h5")
                        data_frame["name"].append(tag_name)
                        data_frame["finetune base"].append(finetune_base)
                        data_frame['inf base'].append(inf_base)
                        data_frame['bias [0, 0.6['].append(bias[0])
                        data_frame['bias [0.6, 2['].append(bias[1])
                        data_frame['bias [2, 4['].append(bias[2])
                        data_frame['bias [4, 6['].append(bias[3])

                        data_frame['smad [0, 0.6['].append(smad[0])
                        data_frame['smad [0.6, 2['].append(smad[1])
                        data_frame['smad [2, 4['].append(smad[2])
                        data_frame['smad [4, 6['].append(smad[3])

                        data_frame["oult [0, 0.6["].append(outl[0])
                        data_frame["oult [0.6, 2["].append(outl[1])
                        data_frame["oult [2, 4["].append(outl[2])
                        data_frame["oult [4, 6["].append(outl[3])


                except Exception as e :
                        print(e)
                        print("file not found for ", model_name)


    fig1.suptitle("redshift estimation heatmap")
    fig1.savefig(base_path+"plots/simCLR/all_heatmaps_classif_"+finetune_base+".png")
    plt.close(fig1)

    fig2.suptitle("prediction bias")
    fig2.savefig(base_path+"plots/simCLR/all_bias_classif_"+finetune_base+".png")
    plt.close(fig2)

    fig3.suptitle("sigma MAD")
    fig3.savefig(base_path+"plots/simCLR/all_smad_classif_"+finetune_base+".png")
    plt.close(fig3)

    fig4.suptitle("outlier fraction")
    fig4.savefig(base_path+"plots/simCLR/all_outl_classif_"+finetune_base+".png")
    plt.close(fig4)


import pandas as pd

df = pd.DataFrame(data_frame)
df.to_csv("metrics_inference_v2.csv", index=False)







