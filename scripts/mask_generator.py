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

def plot_some(cube, k) :
    random.shuffle(cube)
    sel = cube[:9]
    fig, axes = plt.subplots(ncols=9, nrows=9, figsize=(12, 12))
    for i in range(len(sel)) :
        for j in range(sel.shape[-1]) :
            axes[i, j].imshow(sel[i, :, :, j])
            axes[i, j].axis("off")
            axes[i, j].title("im", i,"c =",j)
    plt.tight_layout()
    plt.savefig("plot_cube_"+str(k)+".png")


def load_data(filepath, newpath):
    data = np.load(filepath, allow_pickle=True)
    images = data["cube"][..., :5]  # Garder les 5 premières bandes
    meta = data["info"]
    images = np.sign(images) * (np.sqrt(np.abs(images) + 1) - 1)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        masques = pool.map(get_mask, images)
    print(f"Masks générés pour : {os.path.basename(filepath)}")

    masques = np.expand_dims(masques, axis=-1)  # N, 64, 64, 1
    images = np.concatenate([images, masques], axis=-1)  # N, 64, 64, 6
    np.savez_compressed(newpath, cube=images, info=meta)

def process_directory(cube_directory, new_directory):
    os.makedirs(new_directory, exist_ok=True)
    n_processed = 0
    for root, dirs, files in os.walk(cube_directory):
        for file in files:
            if file.endswith('.npz') and n_processed < 10:
                filepath = os.path.join(root, file)
                newpath = os.path.join(new_directory, file)
                load_data(filepath, newpath)
                n_processed+=1
            

cube_directory = "/lustre/fswork/projects/rech/kof/uve94ap/CUBES_HSC/SPEC/COSMOS/"
new_directory = "/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/spec/"

process_directory(cube_directory, new_directory)

cube_directory = "/lustre/fswork/projects/rech/kof/uve94ap/CUBES_HSC/PHOT/COSMOS/"
new_directory = "/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/phot/"

process_directory(cube_directory, new_directory)
            

