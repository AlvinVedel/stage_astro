import numpy as np
import os
import random
import gc
from pathlib import Path

### SCRIPT POUR LA CREATION DE BASES DE TAILLES FIXES
### UTILISE POUR SEPARER L'ENSEMBLE DES DONNEES EN BLOCS EQUIVALENTS A UNE FRACTION DE LA BASE ET AINSI FINETUNER PROGRESSIVEMENT


dir_path = Path("/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/cleaned_spec/")
extension='UD.npz'

files_list = [file for file in dir_path.rglob(f"*{extension}")]

nb_files = sum([len(np.load(file, allow_pickle=True)["cube"]) for file in files_list])

ten_pct = int(0.1*nb_files)
nb_test = ten_pct*2

nb_per_file = (nb_test // 4) +1



all_test_images = []
all_test_meta = []



leftovers_images = []
leftovers_info = []
train_save_counter = 1

for file in files_list :
    data = np.load(file, allow_pickle=True)
    images = data["cube"]
    meta = data["info"]

    inds = np.arange(images.shape[0])
    random.shuffle(inds)


    test_inds = inds[:nb_per_file]
    test_images = images[test_inds]
    test_info = meta[test_inds]
    all_test_images.append(test_images)
    all_test_meta.append(test_info)


    train_inds = inds[nb_per_file:]

    i = 0 
    j = 0

    if len(train_inds) > ten_pct :
        
        while i < len(train_inds) - ten_pct:
            ten_pct_inds = train_inds[j*ten_pct : (j+1)*ten_pct]
            section_images = images[ten_pct_inds]
            section_info = meta[ten_pct_inds]

            np.savez_compressed("/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/final_spec_split/train_base_UD_"+str(train_save_counter)+".npz", cube=section_images, info=section_info)
            i += ten_pct 
            j += 1
            train_save_counter+=1

    only_section_left = train_inds[j*ten_pct:]
    section_images = images[only_section_left]
    section_info = meta[only_section_left]
    leftovers_images.append(section_images)
    leftovers_info.append(section_info)


leftovers_images = np.concatenate(leftovers_images, axis=0)
leftovers_info = np.concatenate(leftovers_info, axis=0)
leftovers_ind = np.arange(leftovers_images.shape[0])


i=0
j=0
if len(leftovers_ind) > ten_pct :
        
    while i < len(leftovers_ind) - ten_pct:
        ten_pct_inds = leftovers_ind[j*ten_pct : (j+1)*ten_pct]
        section_images = leftovers_images[ten_pct_inds]
        section_info = leftovers_info[ten_pct_inds]

        np.savez_compressed("/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/final_spec_split/train_base_UD_"+str(train_save_counter)+".npz", cube=section_images, info=section_info)
        i += ten_pct 
        j += 1
        train_save_counter+=1


only_section_left = leftovers_ind[j*ten_pct:]
section_images = leftovers_images[only_section_left]
section_info = leftovers_info[only_section_left]
np.savez_compressed("/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/final_spec_split/train_base_UD_"+str(train_save_counter)+".npz", cube=section_images, info=section_info)




all_test_images = np.concatenate(all_test_images, axis=0)
all_test_meta = np.concatenate(all_test_meta, axis=0)

np.savez_compressed("/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/final_spec_split/test_base_20pct.npz", cube=all_test_images, info=all_test_meta)







    


