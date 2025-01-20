import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import random
import os
import gc


def rotate_image(inputs):
    image, rotation = inputs
    return tf.image.rot90(image, rotation)


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





class MultiGen(tf.keras.utils.Sequence):
    def __init__(self, paths, batch_size, do_color=True, do_seg=True, do_mask_band=True, do_adversarial=False, image_size=(64, 64, 5), shuffle=True, extensions=[".npz"]):
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
        self.do_adversarial = do_adversarial
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
        self.surveys = []

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

                if "_UD.npz" in path :
                    self.surveys.append(np.ones((len(data["cube"]))))
                elif "_D.npz" in path :
                    self.surveys.append(np.zeros((len(data["cube"]))))

            

            except Exception as e :
                print("file couldn't be readen", path)
        self.surveys = np.concatenate(self.surveys, axis=0)

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
        apply_noise = tf.cast(tf.random.uniform((tf.shape(images)[0], 1, 1, 1)) < apply_prob, tf.float32)
        return images + apply_noise * sampled_noises * (1 - tf.cast(masks, dtype=tf.float32))
    
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


    


    def __getitem__(self, index):
        batch_images = self.images[index * self.batch_size:(index + 1) * self.batch_size]

        labels_dict = {}

        if self.do_color :
            batch_colors = self.colors[index * self.batch_size:(index + 1) * self.batch_size]

        if self.do_adversarial : 
            batch_survey = self.surveys[index * self.batch_size: (index+1)*self.batch_size]
            
        if tf.shape(batch_images)[0] < self.batch_size:
            # Compléter le batch avec des images dupliquées ou ignorer (selon ta logique)
            pad_size = self.batch_size - batch_images.shape[0]
            batch_images = tf.concat([batch_images, self.images[:pad_size]], axis=0)  # Compléter avec les premières images
            if self.do_color :
                batch_colors = tf.concat([batch_colors, self.colors[:pad_size]], axis=0)
            if self.do_adversarial :
                batch_survey = tf.concat([batch_survey, self.surveys[:pad_size]], axis=0)
                
        if self.do_color :
            batch_colors = tf.cast(tf.tile(batch_colors, [2, 1]), dtype=tf.float32)
            labels_dict["color"] = batch_colors

        if self.do_adversarial :
            batch_surveys = tf.cast(tf.tile(tf.expand_dims(batch_surveys, axis=1), [2, 1]), dtype=tf.float32)
            labels_dict["survey"] = batch_survey

        

        batch_masks = batch_images[:, :, :, 5]
        batch_images = batch_images[:, :, :, :5]
        batch_masks = tf.cast(tf.tile(batch_masks, [2, 1, 1]), dtype=bool)
        batch_images = tf.cast(tf.tile(batch_images, [2, 1, 1, 1]), dtype=tf.float32)

        augmented_images = self.process_batch(batch_images, batch_masks)
        

        if self.do_seg :
            labels_dict["seg_mask"] = tf.expand_dims(batch_masks, axis=-1)
        
        if self.do_mask_band :
            augmented_images = self.drop_band(augmented_images)

        
        
        
        
        #labels = tf.zeros((len(batch_images),), dtype=tf.float32)
        #labels = tf.cast(tf.tile(self.colors[index*self.batch_size:(index+1)*self.batch_size], [2, 1]), dtype=tf.float32)  # batch*2, 4
        return augmented_images, labels_dict


    def on_epoch_end(self):
        self.n_epochs+=1
        indices = np.arange(0, self.images.shape[0], dtype=np.int32)
        np.random.shuffle(indices)
        self.images = self.images[indices]
        self.colors = self.colors[indices]









class SupervisedGenerator(keras.utils.Sequence) :
    def __init__(self, data_path, batch_size, nbins=150, adversarial=False, adv_extensions=["_D.npz"], adversarial_dir=None) :
        super(SupervisedGenerator, self).__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        self.nbins=nbins
        self.adversarial = adversarial
        self.adversarial_dir = adversarial_dir
        self.adversarial_paths = []
        self.extensions=adv_extensions
        self.load_data()
        self.on_epoch_end()
        if self.adversarial :
            self._find_paths(self.adversarial_dir)

    def _find_paths(self, dir_paths) :
        for dire in dir_paths :
            for root, dirs, files in os.walk(dire):
                for file in files:
                    if file.endswith(tuple(self.extensions)):
                        filepath = os.path.join(root, file)
                        self.adversarial_paths.append(filepath)
        random.shuffle(self.adversarial_paths)

    def load_data(self) :
        data = np.load(self.data_path, allow_pickle=True)
        images = data["cube"][..., :5]  # on ne prend que les 5 premières bandes
        masks = np.expand_dims(data["cube"][..., 5], axis=-1)

        #images = np.sign(images)*(np.sqrt(np.abs(images)+1)-1 )   # PAS BESOIN CAR SAUVEGARDEES NORMALISES
        self.images = np.concatenate([images, masks], axis=-1).astype(np.float32)  # N, 64, 64, 6

        meta = data["info"]
        self.z_values = meta[:, 6]
        self.z_values = self.z_values.astype("float32")
        print("Z VALS", self.z_values)
        #bins_edges = np.linspace(0, 6, 300)
        bins_edges = np.concatenate([np.linspace(0, 4, 381), np.linspace(4, 6, 21)[1:]], axis=0)
        self.z_bins = np.zeros((len(self.z_values)))
        for j, z in enumerate(self.z_values) :
            i = 0
            flag = True
            while flag and i < len(bins_edges)-1 :
                if z >= bins_edges[i] and z < bins_edges[i+1] :
                    self.z_bins[j] = i
                    flag = False
                i+=1
            if flag : 
                self.z_bins[j] = i-1
        print(np.max(self.z_bins), np.min(self.z_bins))
        print("NAN IMGS :",np.any(np.isnan(self.images)))
        print("NAN Z :", np.any(np.isnan(self.z_values)), np.any(np.isnan(self.z_bins)))
        self.z_bins = self.z_bins.astype(np.int32)
        print(self.z_bins)

        if self.adversarial :

            adv_imgs = []
            for p in self.adversarial_paths :
                data = np.load(p, allow_pickle=True)
                images = data["cube"][..., :5]
                masks = np.expand_dims(data["cube"][..., 5], axis=-1)
                adv_imgs.append(np.concatenate([images, masks], axis=-1).astype(np.float32) )
            self.adversarial_images = np.concatenate(adv_imgs, axis=0)



    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))

    def process_batch(self, images, masks, ebv=None) :

        images = tf.image.random_flip_left_right(images)
        images = tf.image.random_flip_up_down(images)
        rotations = tf.random.uniform((tf.shape(images)[0],), minval=0, maxval=4, dtype=tf.int32)
        images = tf.map_fn(rotate_image, (images, rotations), dtype=images.dtype)


        return images

    def __getitem__(self, index):
        batch_images = self.images[index*self.batch_size : (index+1)*self.batch_size]
        batch_z = self.z_bins[index*self.batch_size : (index+1)*self.batch_size]
        batch_z2 = self.z_values[index*self.batch_size : (index+1)*self.batch_size]


        if tf.shape(batch_images)[0] < self.batch_size:
            # Compléter le batch avec des images dupliquées ou ignorer (selon ta logique)
            pad_size = self.batch_size - batch_images.shape[0]
            batch_images = tf.concat([batch_images, self.images[:pad_size]], axis=0)  # Compléter avec les premières images
            batch_z = tf.concat([batch_z, self.z_bins[:pad_size]], axis=0)
            batch_z2 = tf.concat([batch_z2, self.z_values[:pad_size]], axis=0)


        batch_masks = batch_images[:, :, :, 5]
        batch_images = batch_images[:, :, :, :5]

        augmented_images = self.process_batch(batch_images, batch_masks)
        if self.adversarial :
            batch_images = self.adversarial_images[index*self.batch_size:(index+1)*self.batch_size]
            if tf.shape(batch_images)[0] < self.batch_size:
                # Compléter le batch avec des images dupliquées ou ignorer (selon ta logique)
                pad_size = self.batch_size - batch_images.shape[0]
                batch_images = tf.concat([batch_images, self.adversarial_images[:pad_size]], axis=0)

            adversarial_images = self.process_batch(batch_images[..., :5], batch_images[..., 5])

            return (augmented_images, adversarial_images), {"pdf":batch_z, "reg":batch_z2}

        return augmented_images, {"pdf":batch_z, "reg":batch_z2}

    def on_epoch_end(self):
        indices = np.arange(0, self.images.shape[0], dtype=np.int32)
        np.random.shuffle(indices)
        self.images = self.images[indices]
        self.z_values = self.z_values[indices]
        self.z_bins = self.z_bins[indices]
