import numpy as np
import tensorflow as tf
import random
import os
import gc


class DinoGenerator(tf.keras.utils.Sequence):
    def __init__(self, folder_path, batch_size, large_crop_size, small_crop_size, folder_extension='_D.npz', shuffle=True, patch_size=4, limit=8):
        self.folder_path = folder_path
        self.folder_extension = folder_extension
        self.batch_size = batch_size
        self.large_crop_size = large_crop_size
        self.small_crop_size = small_crop_size
        self.shuffle = shuffle
        self.patch_size=patch_size
        self.limit = limit
        self.epoch_count = 0
        self.file_count=0
        self.get_filename()
        self.load_data()
        self.on_epoch_end()


    def get_filename(self) :
        self.file_paths = []
        if isinstance(self.folder_path, str):

            for file_name in os.listdir(self.folder_path):
                if file_name.endswith(self.folder_extension):
                    file_path = os.path.join(self.folder_path, file_name)
                    self.file_paths.append(file_path)
        else : 
            for i, folder in enumerate(self.folder_path) :
                for file_name in os.listdir(folder):
                    if file_name.endswith(self.folder_extension[i]):
                        file_path = os.path.join(folder, file_name)
                        self.file_paths.append(file_path)

    def load_data(self):
        self.images = []  

        for _ in range(self.limit) :
            file_path = self.file_paths[self.file_count]
            self.file_count = (self.file_count+1)%len(self.file_paths)

            data = np.load(file_path, allow_pickle=True)
            images = np.sign(data['cube']) * (np.sqrt(np.abs(data["cube"])+1)-1)
            self.images.append(images)
                
        self.images = np.concatenate(self.images, axis=0)
        



    def _apply_crops(self, image, large_crop_size, small_crop_size, num_large_crops=2, num_small_crops=4, masking_rate=0.4):
        h, w = image.shape[:2]
        ch, cw = large_crop_size
        small_crops = []
        large_crops = []
        #masked_large_crops = []
        masked_indexes = []
        num_patch_per_large = (ch // self.patch_size)**2
        
        mask = np.zeros((self.patch_size, self.patch_size, 9))

        for _ in range(num_large_crops):
            if h > ch and w > cw:
                top = random.randint(0, h - ch)
                left = random.randint(0, w - cw)
                large_crop = image[top:top + ch, left:left + cw]
                large_crops.append(large_crop)

                mask_indices = np.random.choice([False, True], size=(int(num_patch_per_large)), replace=True, p=[1-masking_rate, masking_rate])
    
                masked_indexes.append(mask_indices)


        ch, cw = small_crop_size
        for _ in range(num_small_crops):
            if h > ch and w > cw:
                top = random.randint(0, h - ch)
                left = random.randint(0, w - cw)
                small_crop = image[top:top + ch, left:left + cw]
                small_crops.append(small_crop)

        return np.array(large_crops), np.array(small_crops), np.array(masked_indexes) # np.array(masked_large_crops),
    



    def __len__(self):
        # Nombre de batches par epoch
        return int(np.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        batch_images = self.images[index * self.batch_size:(index + 1) * self.batch_size]
        large_crops = []
        small_crops = []
        #masked_large_crops = []
        masked_patch_index = []

        for img in batch_images:
            large_crop, small_crop, masked_index = self._apply_crops(img, self.large_crop_size, self.small_crop_size)
            large_crops.append(large_crop)
            small_crops.append(small_crop)
            #masked_large_crops.append(masked_crop)
            masked_patch_index.append(masked_index)

        return {
            'large_crop': np.array(large_crops),
            'small_crop': np.array(small_crops),
            #'masked_crop' : np.array(masked_large_crops),
            'masked_patch_index' : np.array(masked_patch_index)

        }

    def on_epoch_end(self):
        self.epoch_count+=1
        if self.epoch_count % 5 == 0 :
            del self.images
            gc.collect()
            self.load_data()
        np.random.shuffle(self.images)










import multiprocessing as mp
from scipy.ndimage import center_of_mass
from scipy import ndimage

def get_mask(image, threshold=0.2, center_window_fraction=0.6) :
            mean_image = np.mean(image, axis=-1)
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
            
            mask[image.shape[0]//2-2:image.shape[0]//2+2, image.shape[1]//2-2:image.shape[1]//2+2] = np.ones((4, 4))
            return mask

class DinoDataAug(tf.keras.utils.Sequence):
    def __init__(self, folder_path, batch_size, large_crop_size, small_crop_size, folder_extension='_D.npz', shuffle=True, patch_size=4, limit=8):
        self.folder_path = folder_path
        self.folder_extension = folder_extension
        self.batch_size = batch_size
        self.large_crop_size = large_crop_size
        self.small_crop_size = small_crop_size
        self.shuffle = shuffle
        self.patch_size=patch_size
        self.limit = limit
        self.epoch_count = 0
        self.file_count=0
        self.get_filename()
        self.load_data()
        self.on_epoch_end()


    def get_filename(self) :
        self.file_paths = []
        if isinstance(self.folder_path, str):

            for file_name in os.listdir(self.folder_path):
                if file_name.endswith(self.folder_extension):
                    file_path = os.path.join(self.folder_path, file_name)
                    self.file_paths.append(file_path)
        else : 
            for i, folder in enumerate(self.folder_path) :
                for file_name in os.listdir(folder):
                    if file_name.endswith(self.folder_extension[i]):
                        file_path = os.path.join(folder, file_name)
                        self.file_paths.append(file_path)

    
    def load_data(self):
        self.images = []  

        for _ in range(self.limit) :
            file_path = self.file_paths[self.file_count]
            self.file_count = (self.file_count+1)%len(self.file_paths)

            data = np.load(file_path, allow_pickle=True)
            images = np.sign(data['cube']) * (np.sqrt(np.abs(data["cube"])+1)-1)
            self.images.append(images)
                
        self.images = np.concatenate(self.images, axis=0)
        with mp.Pool(processes=mp.cpu_count()) as pool:
            masques = pool.map(get_mask, self.images)
        print("MASKS TERMINES")
        masques = np.expand_dims(masques, axis=-1) # shape N, 64, 64, 1
        self.images = np.concatenate([self.images, masques], axis=-1) # N, 64, 64, 10

    
    def apply_augmentation(self, image, mask):
        for i in range(image.shape[-1] - 1):  # On parcourt tous les canaux sauf le dernier (le masque de focus)
            bande = image[:, :, i]
            if random.random() < 0.8:
                noise = tf.random.normal(shape=(image.shape[0], image.shape[1]), mean=0.0, stddev=0.1)
                noise = tf.where(mask, 0.0, noise)  # Applique le bruit seulement aux zones hors du masque
                image = tf.tensor_scatter_nd_add(image, indices=[[..., i]], updates=noise)

            if random.random() < 0.25:
                image = tf.tensor_scatter_nd_update(image, indices=[[..., i]], updates=tf.where(mask, bande, 0.0))
                
        if random.random() < 0.8:
            image = self.apply_mask(image, mask)  # KEEP
        image = self.apply_basic_transform(image)
        return image

    def apply_basic_transform(self, image) :
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.rot90(image, k=random.randint(0, 4))
        return image
    
    def apply_mask(self, image, mask):
        msk = image[:, :, 9]
        image = image[:, :, :9]
        mask_size = 12
        rdm = tf.random.uniform(shape=(2,), minval=0, maxval = image.shape[0] - mask_size, dtype=tf.int32)
        x, y = rdm
        masking = np.zeros((image.shape[0], image.shape[1]))
        masking[x:x+mask_size, y:y+mask_size] = tf.ones((mask_size, mask_size))
        mask_final = tf.logical_and(masking, ~mask)  # on prend les points qui sont dans le masking mais PAS dans le mask
        mask_val = tf.reduce_mean(image[mask_final])*np.random.normal(1, 0.01, size=(1))
        image[mask_final] = mask_val
        image = tf.concat([image, tf.expand_dims(msk, axis=-1)], axis=-1)
        return image



    def _apply_crops(self, image, large_crop_size, small_crop_size, num_large_crops=2, num_small_crops=4, masking_rate=0.4):
        h, w = image.shape[:2]
        ch, cw = large_crop_size
        small_crops = []
        large_crops = []
        #masked_large_crops = []
        masked_indexes = []
        num_patch_per_large = (ch // self.patch_size)**2
        
        #mask = np.zeros((self.patch_size, self.patch_size, 9))

        for _ in range(num_large_crops):
            if h > ch and w > cw:
                top = random.randint(0, h - ch)
                left = random.randint(0, w - cw)
                large_crop = image[top:top + ch, left:left + cw]
                large_crops.append(large_crop)

                mask_indices = np.random.choice([False, True], size=(int(num_patch_per_large)), replace=True, p=[1-masking_rate, masking_rate])
    
                masked_indexes.append(mask_indices)


        ch, cw = small_crop_size
        for _ in range(num_small_crops):
            if h > ch and w > cw:
                top = random.randint(0, h - ch)
                left = random.randint(0, w - cw)
                small_crop = image[top:top + ch, left:left + cw]
                small_crops.append(small_crop)

        return np.array(large_crops), np.array(small_crops), np.array(masked_indexes) # np.array(masked_large_crops),

    def __len__(self):
        # Nombre de batches par epoch
        return int(np.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        batch_images = self.images[index * self.batch_size:(index + 1) * self.batch_size]
        large_crops = []
        small_crops = []
        #masked_large_crops = []
        masked_patch_index = []

        for img in batch_images:
            large_crop, small_crop, masked_index = self._apply_crops(img, self.large_crop_size, self.small_crop_size)
            large_crops.append(large_crop)
            small_crops.append(small_crop)
            #masked_large_crops.append(masked_crop)
            masked_patch_index.append(masked_index)

        return {
            'large_crop': np.array(large_crops),
            'small_crop': np.array(small_crops),
            #'masked_crop' : np.array(masked_large_crops),
            'masked_patch_index' : np.array(masked_patch_index)

        }

    def on_epoch_end(self):
        self.epoch_count+=1
        if self.epoch_count % 5 == 0 :
            del self.images
            gc.collect()
            self.load_data()
        np.random.shuffle(self.images)









def compute_target(x) :
        image = x[..., :9]
        mask = x[..., 9].astype(bool)

        indices = np.where(mask)
        pixels = image[indices]

        colors = np.zeros((8))

        colors[0] = np.mean((pixels[..., 0]-pixels[..., 1])) # u-g
        colors[1] = np.mean((pixels[..., 1] - pixels[..., 2])) # g-r
        colors[2] = np.mean((pixels[..., 3] - pixels[..., 4])) # i-z
        colors[3] = np.mean((pixels[..., 4] - pixels[..., 5])) # z-y
        colors[4] = np.mean((pixels[..., 5] - pixels[..., 6])) # y-j
        colors[5] = np.mean((pixels[..., 6] - pixels[..., 7])) # j-h
        colors[6] = np.mean((pixels[..., 7] - pixels[..., 8])) # h-k
        colors[7] = np.mean((pixels[..., 2] - pixels[..., 3])) # r-i

        return colors

class DinoColor(tf.keras.utils.Sequence):
    def __init__(self, folder_path, batch_size, large_crop_size, small_crop_size, folder_extension='_D.npz', shuffle=True, patch_size=4, limit=8):
        self.folder_path = folder_path
        self.folder_extension = folder_extension
        self.batch_size = batch_size
        self.large_crop_size = large_crop_size
        self.small_crop_size = small_crop_size
        self.shuffle = shuffle
        self.patch_size=patch_size
        self.limit = limit
        self.epoch_count = 0
        self.file_count=0
        self.get_filename()
        self.load_data()
        self.on_epoch_end()


    def get_filename(self) :
        self.file_paths = []
        if isinstance(self.folder_path, str):

            for file_name in os.listdir(self.folder_path):
                if file_name.endswith(self.folder_extension):
                    file_path = os.path.join(self.folder_path, file_name)
                    self.file_paths.append(file_path)
        else : 
            for i, folder in enumerate(self.folder_path) :
                for file_name in os.listdir(folder):
                    if file_name.endswith(self.folder_extension[i]):
                        file_path = os.path.join(folder, file_name)
                        self.file_paths.append(file_path)

    
    def load_data(self):
        self.images = []  

        for _ in range(self.limit) :
            file_path = self.file_paths[self.file_count]
            self.file_count = (self.file_count+1)%len(self.file_paths)

            data = np.load(file_path, allow_pickle=True)
            images = np.sign(data['cube']) * (np.sqrt(np.abs(data["cube"])+1)-1)
            self.images.append(images)
                
        self.images = np.concatenate(self.images, axis=0)
        with mp.Pool(processes=mp.cpu_count()) as pool:
            masques = pool.map(get_mask, self.images)
        print("MASKS TERMINES")
        masques = np.expand_dims(masques, axis=-1) # shape N, 64, 64, 1
        self.images = np.concatenate([self.images, masques], axis=-1) # N, 64, 64, 10

        with mp.Pool(processes=mp.cpu_count()) as pool:
            colors = pool.map(compute_target, self.images)
        self.colors_target = np.array(colors)

    
    def apply_augmentation(self, image, mask):
        for i in range(image.shape[-1] - 1):  # On parcourt tous les canaux sauf le dernier (le masque de focus)
            bande = image[:, :, i]
            if random.random() < 0.8:
                noise = tf.random.normal(shape=(image.shape[0], image.shape[1]), mean=0.0, stddev=0.1)
                noise = tf.where(mask, 0.0, noise)  # Applique le bruit seulement aux zones hors du masque
                image = tf.tensor_scatter_nd_add(image, indices=[[..., i]], updates=noise)

            if random.random() < 0.25:
                image = tf.tensor_scatter_nd_update(image, indices=[[..., i]], updates=tf.where(mask, bande, 0.0))
                
        if random.random() < 0.8:
            image = self.apply_mask(image, mask)  # KEEP
        image = self.apply_basic_transform(image)
        return image

    def apply_basic_transform(self, image) :
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.rot90(image, k=random.randint(0, 4))
        return image
    
    def apply_mask(self, image, mask):
        msk = image[:, :, 9]
        image = image[:, :, :9]
        mask_size = 12
        rdm = tf.random.uniform(shape=(2,), minval=0, maxval = image.shape[0] - mask_size, dtype=tf.int32)
        x, y = rdm
        masking = np.zeros((image.shape[0], image.shape[1]))
        masking[x:x+mask_size, y:y+mask_size] = tf.ones((mask_size, mask_size))
        mask_final = tf.logical_and(masking, ~mask)  # on prend les points qui sont dans le masking mais PAS dans le mask
        mask_val = tf.reduce_mean(image[mask_final])*np.random.normal(1, 0.01, size=(1))
        image[mask_final] = mask_val
        image = tf.concat([image, tf.expand_dims(msk, axis=-1)], axis=-1)
        return image



    def _apply_crops(self, image, large_crop_size, small_crop_size, num_large_crops=2, num_small_crops=4, masking_rate=0.4):
        h, w = image.shape[:2]
        ch, cw = large_crop_size
        small_crops = []
        large_crops = []
        #masked_large_crops = []
        masked_indexes = []
        num_patch_per_large = (ch // self.patch_size)**2
        
        #mask = np.zeros((self.patch_size, self.patch_size, 9))

        for _ in range(num_large_crops):
            if h > ch and w > cw:
                top = random.randint(0, h - ch)
                left = random.randint(0, w - cw)
                large_crop = image[top:top + ch, left:left + cw]
                large_crops.append(large_crop)

                mask_indices = np.random.choice([False, True], size=(int(num_patch_per_large)), replace=True, p=[1-masking_rate, masking_rate])
    
                masked_indexes.append(mask_indices)


        ch, cw = small_crop_size
        for _ in range(num_small_crops):
            if h > ch and w > cw:
                top = random.randint(0, h - ch)
                left = random.randint(0, w - cw)
                small_crop = image[top:top + ch, left:left + cw]
                small_crops.append(small_crop)

        return np.array(large_crops), np.array(small_crops), np.array(masked_indexes) # np.array(masked_large_crops),
    



    def __len__(self):
        # Nombre de batches par epoch
        return int(np.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        batch_images = self.images[index * self.batch_size:(index + 1) * self.batch_size]
        batch_colors = self.colors_target[index * self.batch_size:(index+1)*self.batch_size]
        large_crops = []
        small_crops = []
        #masked_large_crops = []
        masked_patch_index = []

        for img in batch_images:
            large_crop, small_crop, masked_index = self._apply_crops(img, self.large_crop_size, self.small_crop_size)
            large_crops.append(large_crop)
            small_crops.append(small_crop)
            #masked_large_crops.append(masked_crop)
            masked_patch_index.append(masked_index)

        return {
            'large_crop': np.array(large_crops),
            'small_crop': np.array(small_crops),
            #'masked_crop' : np.array(masked_large_crops),
            'masked_patch_index' : np.array(masked_patch_index),
            'colors' : np.array(batch_colors)
        }

    def on_epoch_end(self):
        self.epoch_count+=1
        if self.epoch_count % 5 == 0 :
            del self.images
            gc.collect()
            self.load_data()
        np.random.shuffle(self.images)


