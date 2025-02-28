import numpy as np
import multiprocessing as mp
from scipy.ndimage import center_of_mass
from scipy import ndimage
import os





import random
import matplotlib.pyplot as plt




def load_data(filepath):
    data = np.load(filepath, allow_pickle=True)
    images = data["cube"]
    meta = data["info"]
    
    return images, meta
    



def process_directory(cube_directories, new_directory, extension='.npz'):
    os.makedirs(new_directory, exist_ok=True)
    n_processed = 0
    all_paths = []
    for cube_directory in cube_directories :
        for root, dirs, files in os.walk(cube_directory):
            for file in files:
                if file.endswith('.npz') and n_processed < 10:
                    filepath = os.path.join(root, file)
                    all_paths.append(filepath)
                #newpath = os.path.join(new_directory, file)
                #load_data(filepath, newpath, n_processed)
                #n_processed+=1

    random.shuffle(all_paths)
    all_paths = np.array(all_paths)
    i = 0
    n_saved=1
    base_name = "/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/unsup_cube_"
    while i < len(all_paths) :
        if i+10 < len(all_paths) :
            selected_paths = all_paths[i:i+10]
            ims = []
            metas = []
            for path in selected_paths :
                im, me = load_data(path)
                ims.append(im)
                metas.append(me)

            ims = np.concatenate(ims, axis=0)
            metas = np.concatenate(metas, axis=0)

            n = len(ims) // 5

            for j in range(5) :
                if j == 4 :
                    ims_to_save = ims[n*j:]
                    metas_to_save = metas[n*j:]
                else :
                    ims_to_save = ims[n*j:n*(j+1)]
                    metas_to_save = metas[n*j:n*(j+1)]


                np.savez_compressed(base_name+str(n_saved)+".npz", cube=ims_to_save, info=metas_to_save)


            

cube_directory = "/lustre/fswork/projects/rech/kof/uve94ap/CUBES_HSC/SPEC/COSMOS/"
new_directory = "/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/spec/"

process_directory(cube_directory, new_directory)

cube_directory = "/lustre/fswork/projects/rech/kof/uve94ap/CUBES_HSC/PHOT/COSMOS/"
new_directory = "/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/phot/"

process_directory(cube_directory, new_directory)
            

