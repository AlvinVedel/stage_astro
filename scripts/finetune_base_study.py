import numpy as np

import os 
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import time






bins_edges = np.concatenate([np.linspace(0, 4, 381), np.linspace(4, 6, 21)[1:]], axis=0)
bins_centres = (bins_edges[1:] + bins_edges[:-1])/2
print(bins_edges.shape, bins_centres.shape)

for base in ["b1_1", "b1_2", "b2_1", "b2_2", "b3_1", "b3_2"] :


    data = np.load("/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/finetune/"+base+".npz")
    meta = data["info"]
    z_values = meta[:, 6]
    z_values = z_values.astype(np.float32)
    bins_edges = np.concatenate([np.linspace(0, 4, 381), np.linspace(4, 6, 21)[1:]], axis=0)
    bins_centres = (bins_edges[1:] + bins_edges[:-1])/2
    z_bins = np.zeros((400))
    for z in z_values :
        i = 0
        flag = True
        while flag and i < len(bins_edges)-1 :
            if z >= bins_edges[i] and z < bins_edges[i+1] :
                z_bins[i] += 1
                flag = False
            i+=1
        if flag :
            z_bins[i-1] +=1

    z_bins = z_bins.astype(np.float32)
    print("zbins", z_bins.shape, "centres", bins_centres.shape)
    plt.bar(bins_centres, z_bins)
    plt.xlabel("Z SPEC")
    plt.ylabel("nb obs")
    plt.title("Total = "+str(len(z_values)))
    plt.savefig("/lustre/fswork/projects/rech/dnz/ull82ct/astro/plots/bin_split_base="+base+".png")
    plt.close()



    bins_edges = np.linspace(0, 7, 41)
    bins_centres = (bins_edges[1:] + bins_edges[:-1])/2

    z_bins = np.zeros((40))
    for z in z_values :
        i = 0
        flag = True
        while flag and i < len(bins_edges)-1 :
            if z >= bins_edges[i] and z < bins_edges[i+1] :
                z_bins[i] += 1
                flag = False
            i+=1
        if flag :
            z_bins[i-1] +=1


    z_bins = z_bins / np.sum(z_bins)

    plt.plot(bins_centres, z_bins)
    plt.xlabel("Z SPEC")
    plt.ylabel("density")
    plt.title("N samples = "+str(len(z_values)))
    plt.savefig("/lustre/fswork/projects/rech/dnz/ull82ct/astro/plots/z_distr_base="+base+".png")
    plt.close()



