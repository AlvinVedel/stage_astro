import numpy as np

import os 
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import time


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


bins_edges = np.concatenate([np.linspace(0, 4, 381), np.linspace(4, 6, 21)[1:]], axis=0)
bins_centres = (bins_edges[1:] + bins_edges[:-1])/2
print(bins_edges.shape, bins_centres.shape)


density_bins = np.linspace(0, 10, 1001)
density_centres = (density_bins[1:] + density_bins[:-1])/2

def find_bin(bins_edges, value) :
    flag= True
    i= 0
    vec = np.zeros((len(bins_edges)))
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


density1 = np.zeros((len(bins_edges)))
density2 = np.zeros((len(density_bins)))
total_obs = 0


for file in paths :


    data = np.load(file, allow_pickle=True)
    meta = data["info"]
    z_values = meta[:, 6]
    z_values = z_values.astype(np.float32)
    partial_find_bin = partial(find_bin, bins_edges=bins_edges)
    with mp.Pool() as pool:
        vecs = pool.map(partial_find_bin, z_values)
    density1 += np.sum(vecs)

    partial_find_bin = partial(find_bin, bins_edges=density_bins)
    with mp.Pool() as pool:
        vecs = pool.map(partial_find_bin, z_values)
    density2 += np.sum(vecs)

    total_obs+=len(z_values)


density2 = density2/np.sum(density2)



plt.bar(bins_centres, density1)
plt.xlabel("Z SPEC")
plt.ylabel("nb obs")
plt.title("Total = "+str(total_obs))
plt.savefig("/lustre/fswork/projects/rech/dnz/ull82ct/astro/plots/inferencebase"+name+"_binsplit.png")
plt.close()

plt.bar(density_centres, density2)
plt.xlabel("Z SPEC")
plt.ylabel("density")
plt.title("Total = "+str(total_obs))
plt.savefig("/lustre/fswork/projects/rech/dnz/ull82ct/astro/plots/inferencebase"+name+"_density.png")
plt.close()





    


    



