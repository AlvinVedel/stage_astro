import numpy as np

"""

ANALYSE DE LA DISTRIBUTION DES DONNEES ET ECRITURE DANS UN CSV

"""


base_path = "/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/"

redshift_distribution_edges = np.linspace(0, 6, 13)
redshift_distribution_centres = (redshift_distribution_edges[1:] + redshift_distribution_edges[:-1])/2

classif_bins = np.concatenate([np.linspace(0, 4, 381), np.linspace(4, 6, 21)[1:]], axis=0)  
bins_centers = (classif_bins[1:] + classif_bins[:-1])/2

def find_index(value, edges) :
    flag = True 
    index = 0
    while flag and index < len(edges)-1 :
        if value >= edges[index] and value < edges[index+1] :
            flag = False
            return index
        index+=1
    return index


inference_counter = np.zeros((2, 12))
finetune_counter = np.zeros((3, 12))
unsup_counter = np.zeros((2, 12))

bin_inference_counter = np.zeros((2, 400))
bin_finetune_counter = np.zeros((3, 400))
bin_unsup_counter = np.zeros((2, 400))


for file in ["cube_1_D.npz",  "cube_1_UD.npz",  "cube_2_UD.npz",  "cube_3_UD.npz"] :
    data = np.load(base_path+"cleaned_spec/"+file, allow_pickle=True)["info"]
    print(data.dtype)
    type_ = 1 if "_D" in file else 0
    for row in data :
        z_value = row["ZSPEC"]
        count_ind = find_index(z_value, redshift_distribution_edges)
        inference_counter[type_, count_ind] += 1

        bin_count_ind = find_index(z_value, classif_bins)
        bin_inference_counter[type_, bin_count_ind] += 1


print("end for inference base")


for i, file in enumerate(["base1", "base2", "base3"]) :
    data = np.load(base_path+"finetune/"+file+".npz", allow_pickle=True)["info"]
    print(data.dtype)
    
    for row in data :
        z_value = row["ZSPEC"]
        count_ind = find_index(z_value, redshift_distribution_edges)
        finetune_counter[i, count_ind] += 1

        bin_count_ind = find_index(z_value, classif_bins)
        bin_finetune_counter[i, bin_count_ind] += 1


print("finis pour le finetune")




for file in ["cube_10_D.npz" , "cube_12_D.npz",  "cube_1_UD.npz",  "cube_2_UD.npz",  "cube_3_UD.npz",  "cube_5_D.npz",  "cube_7_D.npz",  "cube_9_D.npz",
            "cube_11_D.npz",  "cube_1_D.npz",   "cube_2_D.npz",   "cube_3_D.npz",   "cube_4_D.npz",   "cube_6_D.npz",  "cube_8_D.npz"] :
    data = np.load(base_path+"cleaned_phot/"+file, allow_pickle=True)["info"]
    print(data.dtype)
    type_ = 1 if "_D" in file else 0
    for row in data :
        z_value = row["ZPHOT"]
        count_ind = find_index(z_value, redshift_distribution_edges)
        unsup_counter[type_, count_ind] += 1

        bin_count_ind = find_index(z_value, classif_bins)
        bin_unsup_counter[type_, bin_count_ind] += 1

    print("one unsup file ended")



print("finis pour unsupervised")

data_frame = {"kind":[], "split":[], "plage_value":[], "count":[], "survey":[]}


for i in range(inference_counter.shape[0]) : 
    for j in range(inference_counter.shape[1]) :
        data_frame["kind"].append("inference")
        data_frame["split"].append("uniforme")
        data_frame["plage_value"].append(redshift_distribution_centres[j])
        data_frame["count"].append(inference_counter[i, j])
        surv = "D" if i == 1 else "UD"
        data_frame["survey"].append(surv)


    for j in range(bin_inference_counter.shape[1]) :
        data_frame["kind"].append("inference")
        data_frame["split"].append("bin")
        data_frame["plage_value"].append(bins_centers[j])
        data_frame["count"].append(bin_inference_counter[i, j])
        surv = "D" if i == 1 else "UD"
        data_frame["survey"].append(surv)



for i in range(finetune_counter.shape[0]) :

    for j in range(finetune_counter.shape[1]) : 
   
        data_frame["kind"].append("finetune_base"+str(i+1))
        data_frame["split"].append("uniforme")
        data_frame["plage_value"].append(redshift_distribution_centres[j])
        data_frame["count"].append(finetune_counter[i, j])
        data_frame["survey"].append("UD")

    for j in range(bin_finetune_counter.shape[1]) : 
   
        data_frame["kind"].append("finetune_base"+str(i+1))
        data_frame["split"].append("bin")
        data_frame["plage_value"].append(bins_centers[j])
        data_frame["count"].append(bin_finetune_counter[i, j])
        data_frame["survey"].append("UD")




for i in range(unsup_counter.shape[0]) : 
    for j in range(unsup_counter.shape[1]) :
        data_frame["kind"].append("unsup")
        data_frame["split"].append("uniforme")
        data_frame["plage_value"].append(redshift_distribution_centres[j])
        data_frame["count"].append(unsup_counter[i, j])
        surv = "D" if i == 1 else "UD"
        data_frame["survey"].append(surv)


    for j in range(bin_unsup_counter.shape[1]) :
        data_frame["kind"].append("unsup")
        data_frame["split"].append("bin")
        data_frame["plage_value"].append(bins_centers[j])
        data_frame["count"].append(bin_unsup_counter[i, j])
        surv = "D" if i == 1 else "UD"
        data_frame["survey"].append(surv)


import pandas as pd

df = pd.DataFrame(data_frame)
df.to_csv(base_path+"bases_distributions.csv", index=False)

















