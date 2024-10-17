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








