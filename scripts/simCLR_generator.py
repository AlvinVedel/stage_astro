import tensorflow as tf
import numpy as np
import random
import os
import gc


def rotate_image(inputs):
    image, rotation = inputs
    return tf.image.rot90(image, rotation)

class Gen(tf.keras.utils.Sequence):
    def __init__(self, paths, batch_size, image_size=(64, 64, 5), shuffle=True, extensions=[".npz"]):
        self.paths = []
        self.path_index=0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_size = image_size
        self.max_images = 80000
        self.extensions = extensions
        self.n_epochs = 0
        self._find_paths(paths)
        self._load_data()
        self.on_epoch_end()
        


    def _find_paths(self, dir_paths) :
        for dire in dir_paths :
            for root, dirs, files in os.walk(dire):
                for file in files:
                    if file.endswith(tuple(self.extensions)):
                        filepath = os.path.join(root, file)
                        self.paths.append(filepath)
        random.shuffle(self.paths)
                        

    def _load_data(self):
        random.shuffle(self.paths)
        self.images = []
        gc.collect()
        while np.sum([len(cube) for cube in self.images]) < self.max_images :
            path = self.paths[self.path_index]
            self.path_index = (self.path_index+1)%len(self.paths)
            try :
                data = np.load(path, allow_pickle=True)
                images = data["cube"][..., :5]  # on ne prend que les 5 premières bandes
                masks = np.expand_dims(data["cube"][..., 5], axis=-1)

                #images = np.sign(images)*(np.sqrt(np.abs(images)+1)-1 )   # PAS BESOIN CAR SAUVEGARDEES NORMALISES
                images = np.concatenate([images, masks], axis=-1)  # N, 64, 64, 6
                self.images.append(images)
            except Exception as e :
                print("file couldn't be readen", path)

        self.images = np.concatenate(self.images, axis=0)
        if self.n_epochs == 0 :
            print("je calcule les mads")
            medians = np.median(self.images[..., :5], axis=(0, 1, 2))  # shape (5,) pour chaque channel
            abs_deviation = np.abs(self.images[..., :5] - medians)  # Déviation absolue
            self.mads = np.median(abs_deviation, axis=(0, 1, 2))  # Une MAD par channel
 

       
    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))
    
    def zoom(self, images, masks) :
        batch_size, height, width = tf.shape(images)[0], tf.shape(images)[1], tf.shape(images)[2]
        zoom_values = tf.random.uniform((batch_size,), minval=(height//2)-4, maxval=height // 2, dtype=tf.int32)
        centers_x = height // 2
        centers_y = width // 2

        crop_boxes = tf.stack([
            (centers_x - zoom_values) / height,  # y_min
            (centers_y - zoom_values) / width,   # x_min
            (centers_x + zoom_values) / height,  # y_max
            (centers_y + zoom_values) / width    # x_max
        ], axis=1)
        images = tf.image.crop_and_resize(tf.cast(images, dtype=tf.float32), tf.cast(crop_boxes, dtype=tf.float32), box_indices=tf.range(batch_size), crop_size=[height, width])
        return images
    
    def gaussian_noise(self, images, masks, apply_prob=0.2) :

        masks = tf.tile(tf.expand_dims(masks, axis=-1), (1, 1, 1, tf.shape(images)[-1])) # shape batch, 64, 64, 5
        us = tf.random.uniform((tf.shape(images)[0], tf.shape(images)[-1]), minval=1, maxval=3, dtype=tf.float32)  # shape batch, 5  sample un u par image par channel (1 à 3 fois le bruit médian)
        new_sigmas = tf.multiply(us, tf.expand_dims(tf.cast(self.mads, dtype=tf.float32), axis=0))   #    batch, 5 * 1, 5     batch, 5  le mads représente le noise scale et les u à quel point ils s'expriment 
        # on a un sigma par channel par image

        noises = tf.random.normal(shape=tf.shape(images), mean=0, stddev=1, dtype=tf.float32) # batch, 64, 64, 5
        sampled_noises = tf.multiply(noises, tf.expand_dims(tf.expand_dims(tf.math.sqrt(new_sigmas), axis=1), axis=1))  # on multiplie par la racine du sigma pour avoir un bruit 0, sigma

        # Génère un tenseur de probabilités d'application pour chaque image du batch
        apply_noise = tf.cast(tf.random.uniform((tf.shape(images)[0], 1, 1, 1)) < apply_prob, tf.float32)

        # Applique le bruit uniquement sur les images pour lesquelles apply_noise est 1, et masque les pixels avec `masks`
        return images + apply_noise * sampled_noises * (1 - tf.cast(masks, dtype=tf.float32))
        #return images + sampled_noises * (1 - tf.cast(masks, dtype=tf.float32))
    
    def process_batch(self, images, masks, ebv=None) :
        
        images = self.gaussian_noise(images, masks)
        images = self.zoom(images, masks)

        images = tf.image.random_flip_left_right(images)
        images = tf.image.random_flip_up_down(images)
        rotations = tf.random.uniform((tf.shape(images)[0],), minval=0, maxval=4, dtype=tf.int32)    
        images = tf.map_fn(rotate_image, (images, rotations), dtype=images.dtype)


        return images

    def __getitem__(self, index):
        batch_images = self.images[index * self.batch_size:(index + 1) * self.batch_size]
          

        if tf.shape(batch_images)[0] < self.batch_size:
            # Compléter le batch avec des images dupliquées ou ignorer (selon ta logique)
            pad_size = self.batch_size - batch_images.shape[0]
            batch_images = tf.concat([batch_images, self.images[:pad_size]], axis=0)  # Compléter avec les premières images
          
                    
        batch_masks = batch_images[:, :, :, 5]
        batch_images = batch_images[:, :, :, :5]
        batch_masks = tf.cast(tf.tile(batch_masks, [2, 1, 1]), dtype=bool)
        batch_images = tf.cast(tf.tile(batch_images, [2, 1, 1, 1]), dtype=tf.float32)
        
        
        augmented_images = self.process_batch(batch_images, batch_masks)
        labels = tf.zeros((len(batch_images),), dtype=tf.float32)
        return augmented_images, labels


    def on_epoch_end(self):
        self.n_epochs+=1
        indices = np.arange(0, self.images.shape[0], dtype=np.int32)
        np.random.shuffle(indices)
        self.images = self.images[indices]





class AdversarialGen(tf.keras.utils.Sequence):
    def __init__(self, paths, batch_size, image_size=(64, 64, 5), shuffle=True, extensions=[".npz"]):
        self.paths = []
        self.path_index=0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_size = image_size
        self.max_images = 80000
        self.extensions = extensions
        self.n_epochs = 0
        self._find_paths(paths)
        self._load_data()
        self.on_epoch_end()
        


    def _find_paths(self, dir_paths) :
        for dire in dir_paths :
            for root, dirs, files in os.walk(dire):
                for file in files:
                    if file.endswith(tuple(self.extensions)):
                        filepath = os.path.join(root, file)
                        self.paths.append(filepath)
        random.shuffle(self.paths)
                        

    def _load_data(self):
        self.images = []
        self.surveys = []
        gc.collect()
        random.shuffle(self.paths)
        while np.sum([len(cube) for cube in self.images]) < self.max_images :
            path = self.paths[self.path_index]
            self.path_index = (self.path_index+1)%len(self.paths)

            data = np.load(path, allow_pickle=True)
    
            self.images.append(data["cube"])

            if "_UD.npz" in path :
                self.surveys.append(np.ones((len(data["cube"]))))
            elif "_D.npz" in path :
                self.surveys.append(np.zeros((len(data["cube"]))))

        self.images = np.concatenate(self.images, axis=0)
        if self.n_epochs == 0 :
            print("je calcule les mads")
            medians = np.median(self.images[..., :5], axis=(0, 1, 2))  # shape (5,) pour chaque channel
            abs_deviation = np.abs(self.images[..., :5] - medians)  # Déviation absolue
            self.mads = np.median(abs_deviation, axis=(0, 1, 2))  # Une MAD par channel
        self.surveys = np.concatenate(self.surveys, axis=0)
 

       
    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))
    
    def zoom(self, images, masks) :
        batch_size, height, width = tf.shape(images)[0], tf.shape(images)[1], tf.shape(images)[2]
        zoom_values = tf.random.uniform((batch_size,), minval=(height//2)-4, maxval=height // 2, dtype=tf.int32)
        centers_x = height // 2
        centers_y = width // 2

        crop_boxes = tf.stack([
            (centers_x - zoom_values) / height,  # y_min
            (centers_y - zoom_values) / width,   # x_min
            (centers_x + zoom_values) / height,  # y_max
            (centers_y + zoom_values) / width    # x_max
        ], axis=1)
        images = tf.image.crop_and_resize(tf.cast(images, dtype=tf.float32), tf.cast(crop_boxes, dtype=tf.float32), box_indices=tf.range(batch_size), crop_size=[height, width])
        return images
    
    def gaussian_noise(self, images, masks, apply_prob=0.2) :

        masks = tf.tile(tf.expand_dims(masks, axis=-1), (1, 1, 1, tf.shape(images)[-1])) # shape batch, 64, 64, 5
        us = tf.random.uniform((tf.shape(images)[0], tf.shape(images)[-1]), minval=1, maxval=3, dtype=tf.float32)  # shape batch, 5  sample un u par image par channel (1 à 3 fois le bruit médian)
        new_sigmas = tf.multiply(us, tf.expand_dims(tf.cast(self.mads, dtype=tf.float32), axis=0))   #    batch, 5 * 1, 5     batch, 5  le mads représente le noise scale et les u à quel point ils s'expriment 
        # on a un sigma par channel par image

        noises = tf.random.normal(shape=tf.shape(images), mean=0, stddev=1, dtype=tf.float32) # batch, 64, 64, 5
        sampled_noises = tf.multiply(noises, tf.expand_dims(tf.expand_dims(tf.math.sqrt(new_sigmas), axis=1), axis=1))  # on multiplie par la racine du sigma pour avoir un bruit 0, sigma

        # Génère un tenseur de probabilités d'application pour chaque image du batch
        apply_noise = tf.cast(tf.random.uniform((tf.shape(images)[0], 1, 1, 1)) < apply_prob, tf.float32)

        # Applique le bruit uniquement sur les images pour lesquelles apply_noise est 1, et masque les pixels avec `masks`
        return images + apply_noise * sampled_noises * (1 - tf.cast(masks, dtype=tf.float32))
        #return images + sampled_noises * (1 - tf.cast(masks, dtype=tf.float32))
    
    def process_batch(self, images, masks, ebv=None) :
        
        images = self.gaussian_noise(images, masks)
        images = self.zoom(images, masks)

        images = tf.image.random_flip_left_right(images)
        images = tf.image.random_flip_up_down(images)
        rotations = tf.random.uniform((tf.shape(images)[0],), minval=0, maxval=4, dtype=tf.int32)    
        images = tf.map_fn(rotate_image, (images, rotations), dtype=images.dtype)


        return images

    def __getitem__(self, index):
        batch_images = self.images[index * self.batch_size:(index + 1) * self.batch_size]
        batch_surveys = self.surveys[index*self.batch_size: (index+1)*self.batch_size]
          

        if tf.shape(batch_images)[0] < self.batch_size:
            # Compléter le batch avec des images dupliquées ou ignorer (selon ta logique)
            pad_size = self.batch_size - batch_images.shape[0]
            batch_images = tf.concat([batch_images, self.images[:pad_size]], axis=0)  # Compléter avec les premières images
            batch_surveys = tf.concat([batch_surveys, self.surveys[:pad_size]], axis=0)
          
                    
        batch_masks = batch_images[:, :, :, 5]
        batch_img = batch_images[:, :, :, :5]
        batch_masks = tf.cast(tf.tile(batch_masks, [2, 1, 1]), dtype=bool)
        batch_img = tf.cast(tf.tile(batch_img, [2, 1, 1, 1]), dtype=tf.float32)
        batch_surveys = tf.cast(tf.tile(tf.expand_dims(batch_surveys, axis=1), [2, 1]), dtype=tf.float32)
        
        
        augmented_images = self.process_batch(batch_img, batch_masks)
        #labels = tf.zeros((len(batch_images),), dtype=tf.float32)
        return augmented_images, batch_surveys


    def on_epoch_end(self):
        self.n_epochs+=1
        indices = np.arange(0, self.images.shape[0], dtype=np.int32)
        np.random.shuffle(indices)
        self.images = self.images[indices]
        self.surveys = self.surveys[indices]
       
