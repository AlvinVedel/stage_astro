import numpy as np
import multiprocessing as mp
from scipy.ndimage import center_of_mass
from scipy import ndimage
import os



def get_mask(image, threshold=0.2, center_window_fraction=0.6) :
    channels_masks = []
    for channel in range(image.shape[-1]) :
        mean_image = image[:, :, channel]
        binary_image = mean_image > threshold * np.max(mean_image)        
        labeled_image, num_labels = ndimage.label(binary_image)        
        h, w = mean_image.shape
        center_h, center_w = h // 2, w // 2
        window_size_h, window_size_w = int(h * center_window_fraction), int(w * center_window_fraction)        
        min_h, max_h = max(0, center_h - window_size_h // 2), min(h, center_h + window_size_h // 2)
        min_w, max_w = max(0, center_w - window_size_w // 2), min(w, center_w + window_size_w // 2)
        best_label = None
        min_distance = float('inf')
        for label in range(1, num_labels + 1):
            mask = labeled_image == label
            obj_center = center_of_mass(mask)
            if (min_h <= obj_center[0] <= max_h) and (min_w <= obj_center[1] <= max_w):
                distance = np.linalg.norm(np.array(obj_center) - np.array([center_h, center_w]))
                if distance < min_distance:
                    min_distance = distance
                    best_label = label
        mask = labeled_image == best_label if best_label is not None else np.zeros_like(mean_image)
        channels_masks.append(mask)
    global_mask = np.clip(np.sum(np.array(channels_masks), axis=0), 0, 1)
    global_mask[image.shape[0]//2-2:image.shape[0]//2+2, image.shape[1]//2-2:image.shape[1]//2+2] = np.ones((4, 4))
    return global_mask


import random
import matplotlib.pyplot as plt






folder_path2 = "/lustre/fswork/projects/rech/kof/uve94ap/CUBES_HSC/SPEC/COSMOS/us" 
new_directory = "/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/cleaned_spec/"



file_paths2 = {'d':[], 'ud':[]}
for file_name in os.listdir(folder_path2):
    if file_name.endswith('_D.npz'):
        file_path = os.path.join(folder_path2, file_name)
        file_paths2['d'].append(file_path)
    elif file_name.endswith('_UD.npz'):
        file_path = os.path.join(folder_path2, file_name)
        file_paths2['ud'].append(file_path)


random.shuffle(file_paths2["ud"])
random.shuffle(file_paths2["d"])


z_key = "zspec"



image_indice = 0
image_bank = np.zeros((80000, 64, 64, 7))
meta_bank = []


ud_files_counter = 1
for i, path in enumerate(file_paths2["ud"]) :
    print("files :",i, "counter :",ud_files_counter)


    info_i = np.load(path, allow_pickle=True)["info"]
    if i == 0 :
        print(info_i.dtype)
    info_i.dtype.names = tuple([x.lower() for x in info_i.dtype.names])
    mask = (info_i["i"] >= 18) & (info_i["i"] <= 25)
    if "flag" in np.load(path).files :
        flags = np.load(path)["flag"][:,[0,  1,  2,  3,  4,  5]]
        mask = mask & (np.sum(flags,axis=1)==0)
    if z_key in info_i.dtype.names:
        mask = mask & (info_i[z_key] >= 0.001) & (info_i[z_key] <= 6)
    mask=mask & (info_i['us']>0)

    images = np.load(path, allow_pickle=True)["cube"][mask][..., :6]
    images = np.sign(images) * (np.sqrt(np.abs(images) + 1) - 1)
    if len(images) > 0 :
       
        print(i, len(images), "images extracted")
        try :
            meta_assoc = np.load(path, allow_pickle=True)["info"][mask]#
            z_assoc = meta_assoc["ZSPEC"]

            ##if no problem with zspec
            n_to_add = images.shape[0]
            with mp.Pool(processes=mp.cpu_count()) as pool:
                masques = pool.map(get_mask, images) 

            nadd = image_bank.shape[0] - image_indice
            if image_indice+n_to_add >= image_bank.shape[0] :
                image_bank[image_indice:] = np.concatenate([images[:nadd], masques[:nadd]], axis=-1)
                meta_bank += [m for m in meta_assoc[:nadd]]
                meta = np.array(meta_bank)
                newpath = new_directory+"cube_"+str(ud_files_counter)+"_UD.npz"
                np.savez_compressed(newpath, cube=image_bank, info=meta)
                ud_files_counter += 1
                meta_bank = []

                nleft = images.shape[0] - nadd
                image_bank[:nleft] = np.concatenate([images[nadd:], masques[nadd:]], axis=-1)
                meta_bank += [m for m in meta_assoc[nadd:]]

                image_indice = nleft


            else :

                image_bank[image_indice:image_indice+n_to_add] = np.concatenate([images, masques], axis=-1)
                image_indice+=n_to_add
                meta_bank += [m for m in meta_assoc]
            
                
        except Exception as e :
            print("problem during loading (probably no zspec)", e)

newpath = new_directory+"cube_"+str(ud_files_counter)+"_UD.npz"
np.savez_compressed(newpath, cube=image_bank[:image_indice], info=np.array(meta_bank))




image_indice = 0
image_bank = np.zeros((80000, 64, 64, 7))
meta_bank = []


ud_files_counter = 1
for i, path in enumerate(file_paths2["d"]) :
    print("files :",i, "counter :",ud_files_counter)


    info_i = np.load(path, allow_pickle=True)["info"]
    if i == 0 :
        print(info_i.dtype)
    info_i.dtype.names = tuple([x.lower() for x in info_i.dtype.names])
    mask = (info_i["i"] >= 18) & (info_i["i"] <= 25)
    if "flag" in np.load(path).files :
        flags = np.load(path)["flag"][:,[0,  1,  2,  3,  4,  5]]
        mask = mask & (np.sum(flags,axis=1)==0)
    if z_key in info_i.dtype.names:
        mask = mask & (info_i[z_key] >= 0.001) & (info_i[z_key] <= 6)
    mask=mask & (info_i['us']>0)

    images = np.load(path, allow_pickle=True)["cube"][mask][..., :6]
    images = np.sign(images) * (np.sqrt(np.abs(images) + 1) - 1)
    if len(images) > 0 :
       
        print(i, len(images), "images extracted")
        try :
            meta_assoc = np.load(path, allow_pickle=True)["info"][mask]#
            z_assoc = meta_assoc["ZSPEC"]

            ##if no problem with zspec
            n_to_add = images.shape[0]
            with mp.Pool(processes=mp.cpu_count()) as pool:
                masques = pool.map(get_mask, images) 

            nadd = image_bank.shape[0] - image_indice
            if image_indice+n_to_add >= image_bank.shape[0] :
                image_bank[image_indice:] = np.concatenate([images[:nadd], masques[:nadd]], axis=-1)
                meta_bank += [m for m in meta_assoc[:nadd]]
                meta = np.array(meta_bank)
                newpath = new_directory+"cube_"+str(ud_files_counter)+"_D.npz"
                np.savez_compressed(newpath, cube=image_bank, info=meta)
                print("I JUSTE SAVE A FILE", ud_files_counter)
                ud_files_counter += 1
                meta_bank = []

                nleft = images.shape[0] - nadd
                image_bank[:nleft] = np.concatenate([images[nadd:], masques[nadd:]], axis=-1)
                meta_bank += [m for m in meta_assoc[nadd:]]

                image_indice = nleft


            else :

                image_bank[image_indice:image_indice+n_to_add] = np.concatenate([images, masques], axis=-1)
                image_indice+=n_to_add
                meta_bank += [m for m in meta_assoc]
            
                
        except Exception as e :
            print("problem during loading (probably no zspec)", e)


newpath = new_directory+"cube_"+str(ud_files_counter)+"_D.npz"
np.savez_compressed(newpath, cube=image_bank[:image_indice], info=np.array(meta_bank))




















folder_path2 = "/lustre/fswork/projects/rech/kof/uve94ap/CUBES_HSC/PHOT/COSMOS" 
new_directory = "/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/cleaned_phot/"


file_paths2 = {'d':[], 'ud':[]}
for file_name in os.listdir(folder_path2):
    if file_name.endswith('_D.npz'):
        file_path = os.path.join(folder_path2, file_name)
        file_paths2['d'].append(file_path)
    elif file_name.endswith('_UD.npz'):
        file_path = os.path.join(folder_path2, file_name)
        file_paths2['ud'].append(file_path)


random.shuffle(file_paths2["ud"])
random.shuffle(file_paths2["d"])



image_indice = 0
image_bank = np.zeros((80000, 64, 64, 6))
meta_bank = []


ud_files_counter = 1
for i, path in enumerate(file_paths2["ud"]) :
    print("files :",i, "counter :",ud_files_counter)


    info_i = np.load(path, allow_pickle=True)["info"]
    if i == 0 :
        print(info_i.dtype)
    info_i.dtype.names = tuple([x.lower() for x in info_i.dtype.names])
    mask = (info_i["i"] >= 18) & (info_i["i"] <= 25)
    if "flag" in np.load(path).files :
        flags = np.load(path)["flag"][:,[0,  1,  2,  3,  4,  5]]
        mask = mask & (np.sum(flags,axis=1)==0)
    if z_key in info_i.dtype.names:
        mask = mask & (info_i[z_key] >= 0.001) & (info_i[z_key] <= 6)
    mask=mask & (info_i['us']>0)

    if 'compact' in info_i.dtype.names:
        mask=mask & (info_i['compact']==0)
    if 'obj_type' in info_i.dtype.names:
        mask=mask & (info_i['obj_type']==0)
    if 'mask' in info_i.dtype.names:
        mask=mask & (info_i['mask']==0)

    images = np.load(path, allow_pickle=True)["cube"][mask][..., :6]
    images = np.sign(images) * (np.sqrt(np.abs(images) + 1) - 1)
    if len(images) > 0 :
       
        print(i, len(images), "images extracted")
        try :
            meta_assoc = np.load(path, allow_pickle=True)["info"][mask]#
            z_assoc = meta_assoc["ZSPEC"]

            ##if no problem with zspec
            n_to_add = images.shape[0]
            with mp.Pool(processes=mp.cpu_count()) as pool:
                masques = pool.map(get_mask, images) 

            nadd = image_bank.shape[0] - image_indice
            if image_indice+n_to_add >= image_bank.shape[0] :
                image_bank[image_indice:] = np.concatenate([images[:nadd], masques[:nadd]], axis=-1)
                meta_bank += [m for m in meta_assoc[:nadd]]
                meta = np.array(meta_bank)
                newpath = new_directory+"cube_"+str(ud_files_counter)+"_UD.npz"
                np.savez_compressed(newpath, cube=image_bank, info=meta)
                print("I JUSTE SAVE A FILE", ud_files_counter)
                ud_files_counter += 1
                meta_bank = []

                nleft = images.shape[0] - nadd
                image_bank[:nleft] = np.concatenate([images[nadd:], masques[nadd:]], axis=-1)
                meta_bank += [m for m in meta_assoc[nadd:]]

                image_indice = nleft


            else :

                image_bank[image_indice:image_indice+n_to_add] = np.concatenate([images, masques], axis=-1)
                image_indice+=n_to_add
                meta_bank += [m for m in meta_assoc]
            
                
        except Exception as e :
            print("problem during loading (probably no zspec)", e)


newpath = new_directory+"cube_"+str(ud_files_counter)+"_UD.npz"
np.savez_compressed(newpath, cube=image_bank[:image_indice], info=np.array(meta_bank))





image_indice = 0
image_bank = np.zeros((80000, 64, 64, 6))
meta_bank = []


ud_files_counter = 1
for i, path in enumerate(file_paths2["d"]) :
    print("files :",i, "counter :",ud_files_counter)

    info_i = np.load(path, allow_pickle=True)["info"]
    if i == 0 :
        print(info_i.dtype)
    info_i.dtype.names = tuple([x.lower() for x in info_i.dtype.names])
    mask = (info_i["i"] >= 18) & (info_i["i"] <= 25)
    if "flag" in np.load(path).files :
        flags = np.load(path)["flag"][:,[0,  1,  2,  3,  4,  5]]
        mask = mask & (np.sum(flags,axis=1)==0)
    if z_key in info_i.dtype.names:
        mask = mask & (info_i[z_key] >= 0.001) & (info_i[z_key] <= 6)
    mask=mask & (info_i['us']>0)

    if 'compact' in info_i.dtype.names:
        mask=mask & (info_i['compact']==0)
    if 'obj_type' in info_i.dtype.names:
        mask=mask & (info_i['obj_type']==0)
    if 'mask' in info_i.dtype.names:
        mask=mask & (info_i['mask']==0)

    images = np.load(path, allow_pickle=True)["cube"][mask][..., :6]
    images = np.sign(images) * (np.sqrt(np.abs(images) + 1) - 1)
    if len(images) > 0 :
       
        print(i, len(images), "images extracted")
        try :
            meta_assoc = np.load(path, allow_pickle=True)["info"][mask]#
            z_assoc = meta_assoc["ZSPEC"]

            ##if no problem with zspec
            n_to_add = images.shape[0]
            with mp.Pool(processes=mp.cpu_count()) as pool:
                masques = pool.map(get_mask, images) 

            nadd = image_bank.shape[0] - image_indice
            if image_indice+n_to_add >= image_bank.shape[0] :
                image_bank[image_indice:] = np.concatenate([images[:nadd], masques[:nadd]], axis=-1)
                meta_bank += [m for m in meta_assoc[:nadd]]
                meta = np.array(meta_bank)
                newpath = new_directory+"cube_"+str(ud_files_counter)+"_D.npz"
                np.savez_compressed(newpath, cube=image_bank, info=meta)
                print("I JUSTE SAVE A FILE", ud_files_counter)
                ud_files_counter += 1
                meta_bank = []

                nleft = images.shape[0] - nadd
                image_bank[:nleft] = np.concatenate([images[nadd:], masques[nadd:]], axis=-1)
                meta_bank += [m for m in meta_assoc[nadd:]]

                image_indice = nleft


            else :

                image_bank[image_indice:image_indice+n_to_add] = np.concatenate([images, masques], axis=-1)
                image_indice+=n_to_add
                meta_bank += [m for m in meta_assoc]
            
                
        except Exception as e :
            print("problem during loading (probably no zspec)", e)


newpath = new_directory+"cube_"+str(ud_files_counter)+"_D.npz"
np.savez_compressed(newpath, cube=image_bank[:image_indice], info=np.array(meta_bank))















