import numpy as np
import os
import random
import gc
from pathlib import Path

### ANCIEN SCRIPT POUR LA CREATION DE BASES DE FINETUNING
# avant filtrage recommandé par astrophysiciens du LAM



dir_path = Path("/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/spec/")
extension = "cos2020_UD.npz"

# Utilisation de pathlib pour simplifier et filtrer les fichiers
files_list = [file for file in dir_path.rglob(f"*{extension}")]
print(len(files_list))
n_files = len(files_list)
n1 = int(5000/n_files)+1
n2 = int(10000/n_files)+1
n3 = int(20000/n_files)+1


def extract_meta(tup) :
    # ORDRE           RA      DEC     EB_V    ZPHOT    EBV     SURVEY    ZSPEC    ZFLAG
    return np.array([tup[1], tup[2], tup[7], tup[29], tup[35], tup[37], tup[40], tup[41]])




b1_1 = []
m1_1 = []
b2_1 = []
m2_1 = []
b3_1 = []
m3_1 = []


import gc

n1_missing = 0
n2_missing = 0
n3_missing = 0

file_count=0
for file in files_list :
        print("j'ouvre un fichier", file_count)
        file_count+=1
        data = np.load(file, allow_pickle=True)
        images = data["cube"]
        meta = data["info"]
        meta_info = np.array([extract_meta(m) for m in meta])
        meta_info[:, 6] = meta_info[:, 6].astype(np.float32)

        inds = np.where((meta_info[:, 6].astype(np.float32)>=0) & (meta_info[:, 6].astype(np.float32)<=6))
        images = images[inds]
        meta_info = meta_info[inds]


        if meta_info.shape[0] > 2*n1 :
            
            taille = images.shape[0]
            indices = np.arange(0, taille)
            random.shuffle(indices)

            if n1_missing > 0 :
                extra_obs = meta_info.shape[0] - 2*n1
                n_sup = min( (extra_obs//2)*2, n1_missing)
            else :
                n_sup=0

            ind_b1 = indices[:n1+n_sup]
            b1_1.append(images[ind_b1])
            m1_1.append(meta_info[ind_b1])


        else :
            taille = images.shape[0]
            indices = np.arange(0, taille)
            random.shuffle(indices)

            ind_b1 = indices[:int(taille//2)]
            b1_1.append(images[ind_b1])
            m1_1.append(meta_info[ind_b1])


            n1_missing += (n1 - int(taille//2))








        if meta_info.shape[0] > 2*n2 :
            
            taille = images.shape[0]
            indices = np.arange(0, taille)
            random.shuffle(indices)

            if n2_missing > 0 :
                extra_obs = meta_info.shape[0] - 2*n2
                n_sup = min( (extra_obs//2)*2, n2_missing)
            else :
                n_sup=0

            ind_b2 = indices[:n2+n_sup]
            b2_1.append(images[ind_b2])
            m2_1.append(meta_info[ind_b2])

            
        else :
            taille = images.shape[0]
            indices = np.arange(0, taille)
            random.shuffle(indices)

            ind_b2 = indices[:int(taille//2)]
            b2_1.append(images[ind_b2])
            m2_1.append(meta_info[ind_b2])

            n2_missing += (n2 - int(taille//2))








        if meta_info.shape[0] > 2*n3 :
            
            taille = images.shape[0]
            indices = np.arange(0, taille)
            random.shuffle(indices)

            if n3_missing > 0 :
                extra_obs = meta_info.shape[0] - 2*n3
                n_sup = min( (extra_obs//2)*2, n3_missing)
            else :
                n_sup=0

            ind_b3 = indices[:n3+n_sup]
            b3_1.append(images[ind_b3])
            m3_1.append(meta_info[ind_b3])

            

        else :
            taille = images.shape[0]
            indices = np.arange(0, taille)
            random.shuffle(indices)

            ind_b3 = indices[:int(taille//2)]
            b3_1.append(images[ind_b3])
            m3_1.append(meta_info[ind_b3])

           

            n3_missing += (n3 - int(taille//2))
        del indices, ind_b3, ind_b2, ind_b1, images, meta_info
        gc.collect()  
        print(len(b1_1), len(b2_1), len(b3_1))


b1_1 = np.concatenate(b1_1, axis=0)
m1_1 = np.concatenate(m1_1, axis=0)

b2_1 = np.concatenate(b2_1, axis=0)
m2_1 = np.concatenate(m2_1, axis=0)

b3_1 = np.concatenate(b3_1, axis=0)
m3_1 = np.concatenate(m3_1, axis=0)


print(len(b1_1), len(b2_1), len(b3_1))
if len(b1_1) > 5000 :
    inds = np.arange(0, len(b1_1))
    random.shuffle(inds)
    inds = inds[:5000]
    b1_1 = b1_1[inds]
    m1_1 = m1_1[inds]



if len(b2_1) > 10000 :
    inds = np.arange(0, len(b2_1))
    random.shuffle(inds)
    inds = inds[:10000]
    b2_1 = b2_1[inds]
    m2_1 = m2_1[inds]


if len(b3_1) > 20000 :
    inds = np.arange(0, len(b3_1))
    random.shuffle(inds)
    inds = inds[:20000]
    b3_1 = b3_1[inds]
    m3_1 = m3_1[inds]






np.savez_compressed("/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/finetune/b1_1_v3.npz", cube=b1_1, info=m1_1)
np.savez_compressed("/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/finetune/b2_1_v3.npz", cube=b2_1, info=m2_1)
np.savez_compressed("/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/finetune/b3_1_v3.npz", cube=b3_1, info=m3_1)
