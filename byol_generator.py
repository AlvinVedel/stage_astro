import numpy as np
import tensorflow as tf
import random




class ByolGenerator(tf.keras.utils.Sequence):
    def __init__(self, folder_path, batch_size, image_size=(64, 64, 9), shuffle=True):
        self.folder_path = folder_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_size = image_size
        self._load_data()
        self.on_epoch_end()
        

    def _load_data(self):
        self.images = []  

        for file_name in os.listdir(self.folder_path):
            if file_name.endswith('.npz'):
                file_path = os.path.join(self.folder_path, file_name)
                data = np.load(file_path, allow_pickle=True)
                images = np.sign(data['cube']) * (np.sqrt(np.abs(data["cube"])+1)-1)
                self.images.append(images)
                
        self.images = np.concatenate(self.images, axis=0)


    def __len__(self):
        # Nombre de batches par epoch
        return int(np.ceil(len(self.images) / self.batch_size))
    

    def preprocess_image(self, image):

        image = self.apply_basic_transform(image)

        if random.random() < 0.5 :
            image = self.apply_mask(image)   # KEEP
        """
        if random.random() < 0.25 :
            image = self.add_noise(image)
        """
        if random.random() < 0.5 :
            image = self.crop_and_resize(image)   # KEEP
        

        return image  #64, 64, 9
    
    def apply_basic_transform(self, image) :
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.rot90(image, k=random.randint(0, 4))
        return image
    
    def apply_white_stripe_h(self, image) :
        
        h0 = random.randint()

    def apply_mask(self, image):
        # ON GARDE LE MASQUE 12 12   => partie aléatoire de l'image masquée
        mask_size = 12
        x = random.randint(0, self.image_size[0] - mask_size)
        y = random.randint(0, self.image_size[1] - mask_size)
        mask = tf.image.pad_to_bounding_box(tf.zeros((mask_size, mask_size, self.image_size[2])), x, y, self.image_size[0], self.image_size[1])
        image = tf.where(mask > 0, mask, image)
        return image
    def add_noise(self, image) :
        gaussian_noise = tf.random.normal(shape=(tf.shape(image)), mean=0, stddev=0.1)
        image = tf.clip_by_value(image+gaussian_noise, 0, 1)
        return image
    def crop_and_resize(self, image) :
        # sélectionne entre 60% et 80% de l'image et resize
        new_size = (self.image_size[0] * random.uniform(0.1, 1), self.image_size[1] * random.uniform(0.1, 1) )
        x = random.randint(0, self.image_size[0] - int(new_size[0]))
        y = random.randint(0, self.image_size[1] - int(new_size[1]))
        crop = image[x:x+int(new_size[0]), y:y+int(new_size[1])]
        image = tf.image.resize(crop, [self.image_size[0], self.image_size[1]], method='bicubic')
        return image
    
 

    

    def __getitem__(self, index):
        # Obtenir un batch d'images
        batch_images = self.images[index * self.batch_size:(index + 1) * self.batch_size]
        augmented_images = np.zeros((self.batch_size*2, self.image_size[0], self.image_size[1], self.image_size[2]))     
        for i, image in enumerate(batch_images) :
            augmented_images[i] = self.preprocess_image(image)
            augmented_images[i+self.batch_size] = self.preprocess_image(image)
            
        #batch_images = [self.preprocess_image(image_path) for image_path in batch_paths]
        
        # pour un batch de 32 retourne 64 images tq 0 = identity0, 1 = transform0, 2 = identity1, 3 = transform1
        return augmented_images.astype(np.float32)

    def on_epoch_end(self):
        # Optionnel : Mélanger les images après chaque époque
        np.random.shuffle(self.images)




import os
class ImageDataGeneratorModified(tf.keras.utils.Sequence):
    def __init__(self, file_path, batch_size, large_crop_size, small_crop_size, shuffle=True, patch_size=4, img_size=(128, 128)):
        self.file_path = file_path
        self.batch_size = batch_size
        self.large_crop_size = large_crop_size
        self.small_crop_size = small_crop_size
        self.shuffle = shuffle
        self.patch_size=patch_size
        self.img_size=img_size
        self._load_data()
        self.on_epoch_end()
        

    def _load_data(self):
        self.images = []
        for root, dirs, files in os.walk(self.file_path):
            for file in files:
                if file.endswith('.jpg'):
                    image_path = os.path.join(root, file)
                    image = tf.io.read_file(image_path)
                    image = tf.image.decode_jpeg(image, channels=3)  
                       
                        
                    image = tf.cast(image, dtype=tf.float32) / 255.0  # Normalisation des pixels entre 0 et 1
                    image = tf.image.resize(image, self.img_size, method='nearest')
                    self.images.append(image.numpy())
        
        self.n = len(self.images)

    def _apply_crops(self, image, large_crop_size, small_crop_size, num_large_crops=2, num_small_crops=4, masking_rate=0.4):
        h, w = image.shape[:2]
        ch, cw = large_crop_size
        small_crops = []
        large_crops = []
        masked_large_crops = []
        masked_indexes = []
        num_patch_per_large = (ch // self.patch_size)**2
        
        mask = np.zeros((self.patch_size, self.patch_size, 3))

        for _ in range(num_large_crops):
            if h > ch and w > cw:
                top = random.randint(0, h - ch)
                left = random.randint(0, w - cw)
                large_crop = image[top:top + ch, left:left + cw]
                large_crops.append(large_crop)

                masked_large_crop = large_crop.copy()
                
                masked_patch_index = np.random.choice(np.arange(0, num_patch_per_large, 1), size=int(num_patch_per_large*masking_rate), replace=False)
                for index in masked_patch_index :
                    row = index // (cw // self.patch_size)  # 
                    col = index % (cw // self.patch_size)

                    row_start = row * self.patch_size
                    col_start = col * self.patch_size

                    masked_large_crop[row_start:row_start+self.patch_size, col_start:col_start+self.patch_size] = mask
                masked_large_crops.append(masked_large_crop)
                masked_indexes.append(masked_patch_index)

        ch, cw = small_crop_size
        for _ in range(num_small_crops):
            if h > ch and w > cw:
                top = random.randint(0, h - ch)
                left = random.randint(0, w - cw)
                small_crop = image[top:top + ch, left:left + cw]
                small_crops.append(small_crop)

        return np.array(large_crops), np.array(small_crops), np.array(masked_large_crops), np.array(masked_indexes)

    def __len__(self):
        # Nombre de batches par epoch
        return int(np.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        batch_images = self.images[index * self.batch_size:(index + 1) * self.batch_size]
        large_crops = []
        small_crops = []
        masked_large_crops = []
        masked_patch_index = []

        for img in batch_images:
            large_crop, small_crop, masked_crop, masked_index = self._apply_crops(img, self.large_crop_size, self.small_crop_size)
            large_crops.append(large_crop)
            small_crops.append(small_crop)
            masked_large_crops.append(masked_crop)
            masked_patch_index.append(masked_index)

        return {
            'large_crop': np.array(large_crops),
            'small_crop': np.array(small_crops),
            'masked_crop' : np.array(masked_large_crops),
            'masked_patch_index' : np.array(masked_patch_index)

        }

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.images)








