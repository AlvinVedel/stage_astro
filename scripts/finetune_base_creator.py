import numpy as np
import os
import random
import gc



directory = "/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/spec/"

files_list = []
for root, dirs, files in os.walk(directory) :
    for file in files :
        if file.endswith("UD.npz") :
            filepath = os.path.join(root, file)
            files_list.append(filepath)


n_files = len(files_list)
n1 = int(5000/n_files)+20
n2 = int(10000/n_files)+40
n3 = int(20000/n_files)+80


def extract_meta(tup) :
    # ORDRE RA   DEC   EB_V    ZPHOT    EBV    SURVEY    ZSPEC    ZFLAG
    return np.array([tup[1], tup[2], tup[7], tup[29], tup[35], tup[37], tup[40], tup[41]])


for i in range(2) :

    b1 = []
    m1 = []
    b2 = []
    m2 = []
    b3 = []
    m3 = []

    n1_missing = 5000
    n2_missing = 10000
    n3_missing = 20000

    file_count=0
    for file in files_list :
        print("j'ouvre un fichier", file_count)
        file_count+=1
        data = np.load(file, allow_pickle=True)
        images = data["cube"]
        meta = data["info"]

        taille = images.shape[0]
        indices = np.arange(0, taille)
        random.shuffle(indices)
        ind_b1 = indices[:n1]
        b1.append(images[ind_b1])
        meta_info = np.array([extract_meta(m) for m in meta[ind_b1]])
        m1.append(meta_info)
        n1_missing-= min(n1, taille)



        indices = np.arange(0, taille)
        random.shuffle(indices)
        ind_b2 = indices[:n2]
        b2.append(images[ind_b2])
        meta_info = np.array([extract_meta(m) for m in meta[ind_b2]])
        m2.append(meta_info)
        n2_missing -= min(n2, taille)


        
        indices = np.arange(0, taille)
        random.shuffle(indices)
        ind_b3 = indices[:n3]
        b3.append(images[ind_b3])
        meta_info = np.array([extract_meta(m) for m in meta[ind_b3]])
        m3.append(meta_info)
        n3_missing -= min(n3, taille)

    print("N missing", n1_missing, n2_missing, n3_missing)


    b1 = np.concatenate(b1, axis=0)
    m1 = np.concatenate(m1, axis=0)

    b2 = np.concatenate(b2, axis=0)
    m2 = np.concatenate(m2, axis=0)

    b3 = np.concatenate(b3, axis=0)
    m3 = np.concatenate(m3, axis=0)

    np.savez_compressed("/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/finetune/b1_"+str(i+1)+".npz", cube=b1, info=m1)
    np.savez_compressed("/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/finetune/b2_"+str(i+1)+".npz", cube=b2, info=m2)
    np.savez_compressed("/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/finetune/b3_"+str(i+1)+".npz", cube=b3, info=m3)



