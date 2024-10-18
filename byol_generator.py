import numpy as np
import tensorflow as tf
import random
import gc
import os


class ByolGenerator(tf.keras.utils.Sequence):
    def __init__(self, folder_path, folder_extension='.npz', batch_size=256, image_size=(64, 64, 9), shuffle=True, limit=200000):
        self.folder_path = folder_path
        self.folder_extension = folder_extension
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_size = image_size
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
        images = []  

        while len(self.images) < self.limit :

            file_path = self.file_paths[self.file_count]
            self.file_count = (self.file_count+1)%len(self.file_paths)

            data = np.load(file_path, allow_pickle=True)
            images = np.sign(data['cube']) * (np.sqrt(np.abs(data["cube"])+1)-1)
            images.append(images)
            self.images = np.concatenate(images, axis=0)
                
        np.random.shuffle(self.images)
        self.images = self.images[:self.limit]


    def __len__(self):
        # Nombre de batches par epoch
        return int(np.ceil(len(self.images) / self.batch_size))
    
    def preprocess_image(self, image):

        image = self.apply_basic_transform(image)
        if random.random() < 0.5 :
            image = self.apply_mask(image)   # KEEP
        if random.random() < 0.5 :
            image = self.crop_and_resize(image)   # KEEP
        return image  #64, 64, 9
    
    def apply_basic_transform(self, image) :
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.rot90(image, k=random.randint(0, 4))
        return image

    def apply_mask(self, image):
        # ON GARDE LE MASQUE 12 12   => partie aléatoire de l'image masquée
        mask_size = 12
        x = random.randint(0, self.image_size[0] - mask_size)
        y = random.randint(0, self.image_size[1] - mask_size)
        mask = tf.image.pad_to_bounding_box(tf.zeros((mask_size, mask_size, self.image_size[2])), x, y, self.image_size[0], self.image_size[1])
        image = tf.where(mask > 0, mask, image)
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
        self.epoch_count+=1
        if self.epoch_count % 5 == 0 :
            del self.images
            gc.collect()
            self.load_data()
        np.random.shuffle(self.images)




