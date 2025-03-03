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
       

import multiprocessing as mp
def compute_target(x) :
        image = x[..., :5]
        mask = x[..., 5].astype(bool)

        indices = np.where(mask)
        pixels = image[indices]

        colors = np.zeros((4))

        colors[0] = np.mean((pixels[..., 0]-pixels[..., 1])) # u-g
        colors[1] = np.mean((pixels[..., 1] - pixels[..., 2])) # g-r
        colors[2] = np.mean((pixels[..., 3] - pixels[..., 4])) # i-z
        colors[3] = np.mean((pixels[..., 2] - pixels[..., 3])) # r-i

        return colors


class ColorGen(tf.keras.utils.Sequence):
    def __init__(self, paths, batch_size, image_size=(64, 64, 5), shuffle=True, extensions=[".npz"]):
        self.paths = []
        self.path_index=0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_size = image_size
        self.max_images = 80000
        self.extensions = extensions
        self.n_epochs = 0
        self.mean = None
        self.std = None
        self.file_tracker = {}
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
                        self.file_tracker[filepath] = (0, 0)
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
                self.file_tracker[path] = (self.file_tracker[path][0]+1, images.shape[0])
            except Exception as e :
                print("file couldn't be readen", path)

        self.images = np.concatenate(self.images, axis=0)
        if self.n_epochs == 0 :
            print("je calcule les mads")
            medians = np.median(self.images[..., :5], axis=(0, 1, 2))  # shape (5,) pour chaque channel
            abs_deviation = np.abs(self.images[..., :5] - medians)  # Déviation absolue
            self.mads = np.median(abs_deviation, axis=(0, 1, 2))  # Une MAD par channel
        with mp.Pool() as pool :
            colors = pool.map(compute_target, self.images)
        self.colors = np.array(colors)
        if self.mean is None and self.std is None :
            self.mean = np.mean(self.colors, axis=0)
            self.std = np.std(self.colors, axis=0)

            self.colors = (self.colors - self.mean) / self.std
        else :
            self.colors = (self.colors - self.mean) / self.std

        print("nb files opened :", np.sum([self.file_tracker[path][0] for path in self.paths]), "distinct :", np.sum([1 if self.file_tracker[path][0]>0 else 0 for path in self.paths])) 
        print("nb images loaded :", np.sum([self.file_tracker[path][1]*self.file_tracker[path][0] for path in self.paths]))
        print("nb distinct images loaded :", np.sum([self.file_tracker[path][1] if self.file_tracker[path][0]>0 else 0 for path in self.paths]))
        print("nb files :", len(self.paths))

        
 

       
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
        batch_colors = self.colors[index * self.batch_size:(index + 1) * self.batch_size]

        if tf.shape(batch_images)[0] < self.batch_size:
            # Compléter le batch avec des images dupliquées ou ignorer (selon ta logique)
            pad_size = self.batch_size - batch_images.shape[0]
            batch_images = tf.concat([batch_images, self.images[:pad_size]], axis=0)  # Compléter avec les premières images
            batch_colors = tf.concat([batch_colors, self.colors[:pad_size]], axis=0)

        batch_masks = batch_images[:, :, :, 5]
        batch_images = batch_images[:, :, :, :5]
        batch_masks = tf.cast(tf.tile(batch_masks, [2, 1, 1]), dtype=bool)
        batch_images = tf.cast(tf.tile(batch_images, [2, 1, 1, 1]), dtype=tf.float32)
        batch_colors = tf.cast(tf.tile(batch_colors, [2, 1]), dtype=tf.float32)
        
        
        augmented_images = self.process_batch(batch_images, batch_masks)
        #labels = tf.zeros((len(batch_images),), dtype=tf.float32)
        #labels = tf.cast(tf.tile(self.colors[index*self.batch_size:(index+1)*self.batch_size], [2, 1]), dtype=tf.float32)  # batch*2, 4
        return augmented_images, batch_colors


    def on_epoch_end(self):
        self.n_epochs+=1
        indices = np.arange(0, self.images.shape[0], dtype=np.int32)
        np.random.shuffle(indices)
        self.images = self.images[indices]
        self.colors = self.colors[indices]




class MultiGen(tf.keras.utils.Sequence):
    def __init__(self, paths, batch_size, do_color=True, do_seg=True, do_mask_band=True, do_mask_patch=False, image_size=(64, 64, 5), shuffle=True, extensions=[".npz"]):
        self.paths = []
        self.path_index=0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_size = image_size
        self.max_images = 80000
        self.extensions = extensions
        self.n_epochs = 0
        self.do_color=do_color
        self.do_seg=do_seg
        self.do_mask_band = do_mask_band
        self.do_mask_patch = do_mask_patch
        self.mean = None
        self.std = None
        self.file_tracker = {}
        test = self.drop_band(np.random.random((32, 64, 64, 5)))
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
                        self.file_tracker[filepath] = (0, 0)
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
                self.file_tracker[path] = (self.file_tracker[path][0]+1, images.shape[0])
            except Exception as e :
                print("file couldn't be readen", path)

        self.images = np.concatenate(self.images, axis=0)
        if self.n_epochs == 0 :
            print("je calcule les mads")
            medians = np.median(self.images[..., :5], axis=(0, 1, 2))  # shape (5,) pour chaque channel
            abs_deviation = np.abs(self.images[..., :5] - medians)  # Déviation absolue
            self.mads = np.median(abs_deviation, axis=(0, 1, 2))  # Une MAD par channel
        if self.do_color : 
            with mp.Pool() as pool :
                colors = pool.map(compute_target, self.images)
            self.colors = np.array(colors)
            if self.mean is None and self.std is None :
                self.mean = np.mean(self.colors, axis=0)
                self.std = np.std(self.colors, axis=0)

                self.colors = (self.colors - self.mean) / self.std
            else :
                self.colors = (self.colors - self.mean) / self.std

        print("nb files opened :", np.sum([self.file_tracker[path][0] for path in self.paths]), "distinct :", np.sum([1 if self.file_tracker[path][0]>0 else 0 for path in self.paths])) 
        print("nb images loaded :", np.sum([self.file_tracker[path][1]*self.file_tracker[path][0] for path in self.paths]))
        print("nb distinct images loaded :", np.sum([self.file_tracker[path][1] if self.file_tracker[path][0]>0 else 0 for path in self.paths]))
        print("nb files :", len(self.paths))

        
 

       
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
    

    def drop_band(self, images) :
        b, h, w, c = tf.shape(images)
        probas = tf.random.uniform((b, c), 0, 1)
        band_to_drop = tf.argmax(probas, axis=1) # shape batch, 
        prob_to_drop = tf.random.uniform((b,), 0, 1)
        apply_drop = tf.cast(tf.less(prob_to_drop, 0.25), dtype=tf.float32)  # renvoie 0 ou 1 si inférieur à 0.4 ?  donc shape batch, 

        band_to_drop = tf.one_hot(band_to_drop, depth=tf.shape(images)[-1]) # batch, 5
        band_to_drop = tf.expand_dims(tf.expand_dims(band_to_drop, axis=1), axis=1)  # batch, 1, 1, 1
        band_to_drop = 1 - tf.tile(band_to_drop, [1, h, w, 1])*tf.expand_dims(tf.expand_dims(tf.expand_dims(apply_drop, axis=-1), axis=-1), axis=-1)  # batch 64 64 5   avec que des 0 si apply drop vaut 0 donc que des 1

        dropped_band = images * band_to_drop
        return dropped_band


    def drop_patch(self, images, prob=0.4):
        """Masque un patch de taille 8x8 pour chaque image dans le batch avec une probabilité."""
        b, h, w, c = tf.shape(images)  # Récupère la forme des images
        mask_size = 8

        # Génère des coordonnées de départ aléatoires pour chaque image (batch, 2)
        starts = tf.cast(tf.random.uniform((b, 2), 0, h - mask_size, dtype=tf.int32), dtype=tf.int32)  # Début de chaque patch (x, y)

        # Crée les indices pour chaque pixel du patch 8x8
        y_indices = tf.reshape(tf.range(mask_size), (1, -1)) + starts[:, 0:1]  # (batch_size, 8)
        x_indices = tf.reshape(tf.range(mask_size), (-1, 1)) + starts[:, 1:2]  # (8, batch_size)

        # Combine les indices x et y pour obtenir un tableau de taille (batch_size, 8, 8, 2)
        patch_indices = tf.concat([y_indices, x_indices], axis=-1)  # (batch_size, 8, 8, 2)

        # Aplatis les images pour faciliter l'indexation
        images_flat = tf.reshape(images, [b, -1, c])  # (batch_size, h * w, c)

        # Aplatis également les indices du patch pour un accès rapide
        patch_indices_flat = tf.reshape(patch_indices, [-1, 2])  # (batch_size * 8 * 8, 2)

        # Crée un masque binaire avec une probabilité de 40 % pour chaque image
        apply_mask_prob = tf.random.uniform((b, 1), minval=0, maxval=1) < prob  # (batch_size, 1)
        
        # Si apply_mask_prob est True, on applique le masque, sinon on garde l'image intacte
        apply_mask = tf.reshape(apply_mask_prob, [-1])  # (batch_size,)

        # Crée un tableau de valeurs "0" (valeur à appliquer pour le masquage)
        patch_values = tf.zeros([patch_indices_flat.shape[0], c], dtype=images_flat.dtype)  # (batch_size * 8 * 8, c)

        # Applique le masque uniquement si apply_mask est True
        def mask_image(_):
            # Masque les pixels
            return tf.tensor_scatter_nd_update(images_flat, patch_indices_flat, patch_values)

        # Applique le masquage conditionnel sur chaque image du batch
        images_flat = tf.cond(apply_mask[0], lambda: mask_image(images_flat), lambda: images_flat)

        # Ramène les images à leur forme d'origine
        images = tf.reshape(images_flat, [b, h, w, c])  # (batch_size, h, w, c)

        return images


    def __getitem__(self, index):
        batch_images = self.images[index * self.batch_size:(index + 1) * self.batch_size]

        labels_dict = {}

        if self.do_color :
            batch_colors = self.colors[index * self.batch_size:(index + 1) * self.batch_size]
            
        if tf.shape(batch_images)[0] < self.batch_size:
            # Compléter le batch avec des images dupliquées ou ignorer (selon ta logique)
            pad_size = self.batch_size - batch_images.shape[0]
            batch_images = tf.concat([batch_images, self.images[:pad_size]], axis=0)  # Compléter avec les premières images
            if self.do_color :
                batch_colors = tf.concat([batch_colors, self.colors[:pad_size]], axis=0)
                
        if self.do_color :
            batch_colors = tf.cast(tf.tile(batch_colors, [2, 1]), dtype=tf.float32)
            labels_dict["color"] = batch_colors

        batch_masks = batch_images[:, :, :, 5]
        batch_images = batch_images[:, :, :, :5]
        batch_masks = tf.cast(tf.tile(batch_masks, [2, 1, 1]), dtype=bool)
        batch_images = tf.cast(tf.tile(batch_images, [2, 1, 1, 1]), dtype=tf.float32)

        augmented_images = self.process_batch(batch_images, batch_masks)
        

        if self.do_seg :
            labels_dict["seg_mask"] = tf.expand_dims(batch_masks, axis=-1)
        
        if self.do_mask_band :
            augmented_images = self.drop_band(augmented_images)

        if self.do_mask_patch :
            augmented_images = self.drop_patch(augmented_images)
        
        
        
        #labels = tf.zeros((len(batch_images),), dtype=tf.float32)
        #labels = tf.cast(tf.tile(self.colors[index*self.batch_size:(index+1)*self.batch_size], [2, 1]), dtype=tf.float32)  # batch*2, 4
        return augmented_images, labels_dict


    def on_epoch_end(self):
        self.n_epochs+=1
        indices = np.arange(0, self.images.shape[0], dtype=np.int32)
        np.random.shuffle(indices)
        self.images = self.images[indices]
        self.colors = self.colors[indices]
