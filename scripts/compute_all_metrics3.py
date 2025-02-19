import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from contrastiv_model import simCLR
import matplotlib.pyplot as plt
from vit_layers import ViT_backbone
from deep_models import basic_backbone, astro_head, astro_model, AstroModel, adv_network, AstroFinetune
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'

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

plots_name = "COMP_NB_EP_PETITLR_GRADCLIP"
path_memory = {}
import gc

for i, finetune_base in enumerate(["base1", "base2", "base3"]) :   #### POUR CHAQUE CONDITION D ENTRAINEMENT ON VA AVOIR UN SUBPLOT
    base_liste = ["_UD", "_D"]
    print("finetune base", finetune_base)

    fig1, ax1 = plt.subplots(nrows=len(base_liste), ncols=2, figsize=(50, 25)) # pour 1 treyer + 3 simCLR dans 2 conditions modèles   ==>   heatmap
    fig2, ax2 = plt.subplots(nrows=len(base_liste), ncols=2, figsize=(50, 25)) # ===> BIAS
    fig3, ax3 = plt.subplots(nrows=len(base_liste), ncols=2, figsize=(50, 25)) # ===> SMAD
    fig4, ax4 = plt.subplots(nrows=len(base_liste), ncols=2, figsize=(50, 25)) # ===> OUT2



    for j, inf_base in enumerate(base_liste):
        print("inf base", inf_base)
        if i == 0 :
            from pathlib import Path

            dir_path = Path("/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/cleaned_spec/")
            extension = inf_base+'.npz'
            npz_files = [file for file in dir_path.rglob(f"*{extension}") if "4" not in str(file)]
            print("npz files :", npz_files)
            path_memory[inf_base] = npz_files

        else :
            npz_files = path_memory[inf_base]

        #npz_files = [f for f in os.listdir(directory) if f.endswith(inf_base+'.npz')]   ## Récupère les fichiers sur lesquels inférer
        #simbases = ["cleaned_cnn_supervised", "cleaned_cnn_supervised_noadv"]
        
        simbases = ["norm300_ColorHead_Regularized_HYP=lr57_clip3_ep50", "norm300_ColorHead_Regularized_HYP=lr57_clip3_ep100", "norm300_ColorHead_Regularized_HYP=lr57_clip3_ep150"] #    , "UD800_classif","UD_D800_classif"]
        iter=0
        #model_liste = ["simCLR_finetune/simCLR_finetune_"+cond+"_base="+finetune_base+"_model="+sim_base+".weights.h5"]
        for k in range((len(simbases))) :
            if False :
                #model_name = base_path+"model_save/checkpoints_supervised/treyer_supervised_"+finetune_base+".weights.h5"
                #model_name = "../model_save/checkpoints_supervised/cnn_backbone_"+finetune_base+".weights.h5"
                model_name = "../model_save/checkpoints_supervised/"+simbases[k]+"_"+finetune_base+".weights.h5"
                tag_name = "supervised_cleaned"
                treyer = True
                if True :
                    #model = astro_model(basic_backbone(), astro_head())
                    model = AstroFinetune(basic_backbone(full_bn=True), astro_head(1024, 400))
                    print("LE ASTRO_MODEL a ", len(model.layers), "layers")
                    #model = AstroModel(back=basic_backbone(), head=astro_head(), is_adv=False, adv_network=adv_network())
                    #print("AstroModel a ", len(model.layers), "layers")
                    model(np.random.random((32, 64, 64, 6)))
                    model.load_weights(model_name)
                    #inp = keras.Input((64, 64, 6))
                    #x = model.back(inp)
                    #pdf, reg = model.head(x)
                    #model = keras.Model(inp, [pdf, reg])
                    #inp = keras.Input((64, 64, 6))
                    #x = model.back(inp)
                    #pdf, reg = model.head(x)
                    #model = keras.Model(inp, [pdf, reg])
                else :
                    model = AstroModel(back=basic_backbone(), head=astro_head(), is_adv=True, adv_network=adv_network())
                    #inp = keras.Input((64, 64, 6))
                    #x = model.back(inp)
                    #pdf, reg = model.head(x)
                    #model = keras.Model(inp, [pdf, reg])
                    #model = backbone_sup(True)#create_model()
                    #model = FineTuneModel(backbone(True), head=output_head(1024))
                    model(np.random.random((32, 64, 64, 6)))
                    model.load_weights(model_name)
                    backbone = keras.Model(model.back.input, model.back.layers[-1].output)
                    inp = keras.Input((64, 64, 6))
                    x = backbone(inp)
                    pdf, reg = model.head(x)
                    model = keras.Model(inp, [pdf, reg])

            else :

                print("iter=",iter, "donc", str(iter//(len(simbases))), str(iter%(len(simbases))) )
                cond = "ALL"
                sim_base = simbases[(iter%(len(simbases)))]
                if sim_base == "280_ViTback_ColorHead" :
                    back = ViT_backbone()
                    back(np.random.random((32, 64, 64, 5)))
                    #back.summary()
                    model = astro_model(back, astro_head(256, 400))
                else :
                    #back = basic_backbone()
                    #print("SUMMARY BACK")
                    #back.summary()
                    #head = astro_head(1024, 400)
                    #print("SUMMARY HEAD")
                    #head.summary()
                    #model = astro_model(back, head)
                    model = AstroFinetune(basic_backbone(full_bn=False, all_bn=False), astro_head(1024, 400))
                    model(np.random.random((32, 64, 64, 6)))
                #model_name = base_path+"model_save/checkpoints_simCLR_finetune/simCLR_finetune_"+cond+"_base="+finetune_base+"_model="+sim_base+".weights.h5"
                model_name = "../model_save/checkpoints_simCLR_finetune2/simCLR_finetune_UD_D__"+cond+"_base="+finetune_base+"_model="+sim_base+".weights.h5"
                tag_name = 'sim_'+sim_base+"_"+cond
                treyer = True
                iter+=1
                #model = FineTuneModel(backbone(True), head=output_head(1024))
                #model(np.random.random((32, 64, 64, 5)))  
            #print("the model name isss :", model_name)
            #print(os.getcwd())
            print(tag_name)
            try : 
                    model.load_weights(model_name)
                

                    true_z = []
                    pred_z = []
                    

                    #bins_edges = np.linspace(0, 6, 300)
                    bins_edges = np.concatenate([np.linspace(0, 4, 381), np.linspace(4, 6, 21)[1:]], axis=0)
                    bins_centres = (bins_edges[1:] + bins_edges[:-1])/2

                    def extract_z(tup) :
                        return tup[40]
                    
                    counter = 0
                    for file in npz_files :
                        #data = np.load(str(base_path)+"data/spec/"+str(file), allow_pickle=True)
                        data = np.load(str(file), allow_pickle=True)
                        images = data["cube"][..., :6]
                        print("images loaded")
                        meta = data["info"]
                        print("meta loaded")
                        z = np.array([extract_z(m) for m in meta])
                        #print(z.shape)
                        true_z.append(z)
                        #print("file opened")
                        if treyer :
                            output = model.predict(images)
                            probas = output["pdf"]
                            #probas = output[0]
                            #print("nan ?", np.any(np.isnan(probas)), np.any(np.isnan(reg)))
                            z_meds = np.array([z_med(p, bins_centres) for p in probas])
                            reg = z_meds
                            #print(reg.shape)
                        else :
                            reg = model.predict(images)
                            reg = reg[:, 0]
                        #print("ive made through prediction")
                        #reg = model.predict(images)
                        #reg = reg[:, 0]
                        pred_z.append(reg)
                        counter+=1
                        del images, meta
                        gc.collect()
                    true_z = np.concatenate(true_z, axis=0)
                    pred_z = np.concatenate(pred_z, axis=0)

                    #### MATRICE DE CHALEUR
                    from scipy.stats import gaussian_kde
                    import matplotlib.pyplot as plt

                    xy = np.vstack([true_z, pred_z])
                    density = gaussian_kde(xy)(xy)


                    ax1[j, k].scatter(true_z, pred_z, c=density, cmap='hot', s=2)
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
    fig1.savefig(base_path+"plots/simCLR/all_heatmaps_classif_"+plots_name+"_"+finetune_base+".png")
    plt.close(fig1)

    fig2.suptitle("prediction bias")
    fig2.savefig(base_path+"plots/simCLR/all_bias_classif_"+plots_name+"_"+finetune_base+".png")
    plt.close(fig2)

    fig3.suptitle("sigma MAD")
    fig3.savefig(base_path+"plots/simCLR/all_smad_classif_"+plots_name+"_"+finetune_base+".png")
    plt.close(fig3)

    fig4.suptitle("outlier fraction")
    fig4.savefig(base_path+"plots/simCLR/all_outl_classif_"+plots_name+"_"+finetune_base+".png")
    plt.close(fig4)


import pandas as pd

df = pd.DataFrame(data_frame)
df.to_csv("metrics_inference_v2_"+plots_name+".csv", index=False)



