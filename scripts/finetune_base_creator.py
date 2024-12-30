import numpy as np
import os
import random
import gc



directory = "/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/spec/"

files_list = []
for root, dirs, files in os.walk(directory) :
    for file in files :
        if file.endswith("spec_UD.npz") :
            filepath = os.path.join(root, file)
            files_list.append(filepath)


n_files = len(files_list)
n1 = int(5000/n_files)+5
n2 = int(10000/n_files)+10
n3 = int(20000/n_files)+15


def extract_meta(tup) :
    # ORDRE           RA      DEC     EB_V    ZPHOT    EBV     SURVEY    ZSPEC    ZFLAG
    return np.array([tup[1], tup[2], tup[7], tup[29], tup[35], tup[37], tup[40], tup[41]])




b1_1 = []
m1_1 = []
b2_1 = []
m2_1 = []
b3_1 = []
m3_1 = []
b1_2 = []
m1_2 = []
b2_2 = []
m2_2 = []
b3_2 = []
m3_2 = []

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

        inds = np.where((meta_info[:, 6]>=0 & meta_info[:, 6]<=6))
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

            ind_b1 = indices[n1+n_sup:2*(n1+n_sup)]
            b1_2.append(images[ind_b1])
            m1_2.append(meta_info[ind_b1])

        else :
            taille = images.shape[0]
            indices = np.arange(0, taille)
            random.shuffle(indices)

            ind_b1 = indices[:int(taille//2)]
            b1_1.append(images[ind_b1])
            m1_1.append(meta_info[ind_b1])

            ind_b1 = indices[int(taille//2):]
            b1_2.append(images[ind_b1])
            m1_2.append(meta_info[ind_b1])

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

            ind_b2 = indices[n2+n_sup:2*(n2+n_sup)]
            b2_2.append(images[ind_b2])
            m2_2.append(meta_info[ind_b2])

        else :
            taille = images.shape[0]
            indices = np.arange(0, taille)
            random.shuffle(indices)

            ind_b2 = indices[:int(taille//2)]
            b2_1.append(images[ind_b2])
            m2_1.append(meta_info[ind_b2])

            ind_b2 = indices[int(taille//2):]
            b2_2.append(images[ind_b2])
            m2_2.append(meta_info[ind_b2])

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

            ind_b3 = indices[n3+n_sup:2*(n3+n_sup)]
            b3_2.append(images[ind_b3])
            m3_2.append(meta_info[ind_b3])

        else :
            taille = images.shape[0]
            indices = np.arange(0, taille)
            random.shuffle(indices)

            ind_b3 = indices[:int(taille//2)]
            b3_1.append(images[ind_b3])
            m3_1.append(meta_info[ind_b3])

            ind_b3 = indices[int(taille//2):]
            b3_2.append(images[ind_b3])
            m3_2.append(meta_info[ind_b3])

            n3_missing += (n3 - int(taille//2))




if len(b1_1) > 5000 :
    inds = np.arange(0, len(b1_1))
    random.shuffle(inds)
    inds = inds[:5000]
    b1_1 = b1_1[inds]
    m1_1 = m1_1[inds]

if len(b1_2) > 5000 :
    inds = np.arange(0, len(b1_2))
    random.shuffle(inds)
    inds = inds[:5000]
    b1_2 = b1_2[inds]
    m1_2 = m1_2[inds]


if len(b2_1) > 10000 :
    inds = np.arange(0, len(b2_1))
    random.shuffle(inds)
    inds = inds[:10000]
    b2_1 = b2_1[inds]
    m2_1 = m2_1[inds]


if len(b2_2) > 10000 :
    inds = np.arange(0, len(b2_2))
    random.shuffle(inds)
    inds = inds[:10000]
    b2_2 = b2_2[inds]
    m2_2 = m2_2[inds]


if len(b3_1) > 20000 :
    inds = np.arange(0, len(b3_1))
    random.shuffle(inds)
    inds = inds[:20000]
    b3_1 = b3_1[inds]
    m3_1 = m3_1[inds]

if len(b3_2) > 20000 :
    inds = np.arange(0, len(b3_2))
    random.shuffle(inds)
    inds = inds[:20000]
    b3_2 = b3_2[inds]
    m3_2 = m3_2[inds]





b1_1 = np.concatenate(b1_1, axis=0)
m1_1 = np.concatenate(m1_1, axis=0)

b2_1 = np.concatenate(b2_1, axis=0)
m2_1 = np.concatenate(m2_1, axis=0)

b3_1 = np.concatenate(b3_1, axis=0)
m3_1 = np.concatenate(m3_1, axis=0)


b1_2 = np.concatenate(b1_2, axis=0)
m1_2 = np.concatenate(m1_2, axis=0)

b2_2 = np.concatenate(b2_2, axis=0)
m2_2 = np.concatenate(m2_2, axis=0)

b3_2 = np.concatenate(b3_2, axis=0)
m3_2 = np.concatenate(m3_2, axis=0)




np.savez_compressed("/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/finetune/b1_1_v2.npz", cube=b1_1, info=m1_1)
np.savez_compressed("/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/finetune/b2_1_v2.npz", cube=b2_1, info=m2_1)
np.savez_compressed("/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/finetune/b3_1_v2.npz", cube=b3_1, info=m3_1)
np.savez_compressed("/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/finetune/b1_2_v2.npz", cube=b1_2, info=m1_2)
np.savez_compressed("/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/finetune/b2_2_v2.npz", cube=b2_2, info=m2_2)
np.savez_compressed("/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/finetune/b3_2_v2.npz", cube=b3_2, info=m3_2)



