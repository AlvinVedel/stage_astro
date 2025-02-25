import numpy as np
import os 
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import time
from pathlib import Path


# SCRIPT POUR L'ANALYSE DES DONNEES D INFERENCE



dir_path = Path("/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/cleaned_spec/")
extension = "_D.npz"
name= "data_DEEP"

# Utilisation de pathlib pour simplifier et filtrer les fichiers
paths = [file for file in dir_path.rglob(f"*{extension}") if "4" not in str(file)]
print(paths)


bins_edges = np.concatenate([np.linspace(0, 4, 381), np.linspace(4, 6, 21)[1:]], axis=0)
bins_centres = (bins_edges[1:] + bins_edges[:-1])/2
print(bins_edges.shape, bins_centres.shape)


density_bins = np.linspace(0, 6, 501)
density_centres = (density_bins[1:] + density_bins[:-1])/2

def find_bin1(value) :
    bins_edges = np.concatenate([np.linspace(0, 4, 381), np.linspace(4, 6, 21)[1:]], axis=0)
    flag= True
    i= 0
    vec = np.zeros((len(bins_edges)-1))
    while i < len(bins_edges)-1 and flag :
        if value < bins_edges[i+1] and value >= bins_edges[i] :
            flag=False
            vec[i] = 1
            return vec
        i+=1
    vec[i-1] = 1
    return vec

def find_bin2(value) :
    bins_edges = np.linspace(0, 6, 501)
    flag= True
    i= 0
    vec = np.zeros((len(bins_edges)-1))
    while i < len(bins_edges)-1 and flag :
        if value < bins_edges[i+1] and value >= bins_edges[i] :
            flag=False
            vec[i] = 1
            return vec
        i+=1
    vec[i-1] = 1
    return vec


import multiprocessing as mp
from functools import partial


density1 = np.zeros((len(bins_edges)-1))
density2 = np.zeros((len(density_bins)-1))
total_obs = 0

def extract_meta(tup) :
    #                  RA      DEC    EB_V         ZPHOT          EBV
    return np.array([tup[1], tup[2], tup[7], tup[40], tup[35]])

for j, file in enumerate(paths) :


    data = np.load(file, allow_pickle=True)
    meta = data["info"]
    #print(meta)
    meta = np.array([extract_meta(m) for m in meta])
    z_values = meta[:, 3]
    z_values = z_values.astype(np.float32)

    with mp.Pool() as pool:
        vecs1 = pool.map(find_bin1, z_values)
    density1 += np.sum(vecs1, axis=0)

    with mp.Pool() as pool:
        vecs2 = pool.map(find_bin2, z_values)
    density2 += np.sum(vecs2, axis=0)

    total_obs+=len(z_values)
    print("ive seen", total_obs,"obs, file ", j, "/", len(paths))


density2 = density2/np.sum(density2)



plt.bar(bins_centres, density1)
plt.xlabel("Z SPEC")
plt.ylabel("nb obs")
plt.title("Total = "+str(total_obs))
plt.savefig("/lustre/fswork/projects/rech/dnz/ull82ct/astro/plots/distribution_bins_cleaned_base_"+name+".png")
plt.close()

plt.plot(density_centres, density2)
plt.xlabel("Z SPEC")
plt.ylabel("density")
plt.title("Total = "+str(total_obs))
plt.savefig("/lustre/fswork/projects/rech/dnz/ull82ct/astro/plots/distribution_z_cleaned_base_"+name+".png")
plt.close()





    


    



