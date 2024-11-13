import numpy as np
import multiprocessing as mp
from scipy.ndimage import center_of_mass
from scipy import ndimage
import os
import gc
import random


def get_mask(image, threshold=0.2, center_window_fraction=0.6) :
            channels_masks = []
            for channel in range(image.shape[-1]) :
                mean_image = image[:, :, channel]
                #print("dans le get mask")
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


dir_path = "/lustre/fswork/projects/rech/kof/uve94ap/CUBES_HSC/PHOT/COSMOS"

file_paths = []

for file_name in os.listdir(dir_path):
    if file_name.endswith("_D.npz"):
        file_path = os.path.join(dir_path, file_name)
        file_paths.append(file_path)

ncubes = len(file_paths)

im_p_cube = int(50000 / ncubes)


images = []
metas = []
i = 0
for file in file_paths :
     print("je lis le fichier", i, "nom :", file)
     data = np.load(file, allow_pickle=True)
     ind = np.arange(0, len(data["info"]))
     random.shuffle(ind)
     selected = ind[:im_p_cube]
     sel_img = data["cube"][selected]
     sel_met = data["info"][selected]

     images.append(sel_img)
     metas.append(sel_met)


     gc.collect()

images = np.concatenate(images, axis=0)
metas = np.concatenate(metas, axis=0)

print("data ready to be save")

np.savez_compressed("my_cube.npz", cube=images, info=metas)


