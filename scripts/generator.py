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
        image = x[..., :6]
        mask = x[..., 6].astype(bool)

        indices = np.where(mask)
        pixels = image[indices]

        colors = np.zeros((5))

        colors[0] = np.mean((pixels[..., 0]-pixels[..., 1])) # u-g
        colors[1] = np.mean((pixels[..., 1] - pixels[..., 2])) # g-r
        colors[2] = np.mean((pixels[..., 3] - pixels[..., 4])) # i-z
        colors[3] = np.mean((pixels[..., 2] - pixels[..., 3])) # r-i
        colors[4] = np.mean((pixels[..., 4] - pixels[..., 5]))

        return colors





class MultiGen(tf.keras.utils.Sequence):
    def __init__(self, paths, batch_size, do_color=True, do_seg=True, do_mask_band=True, do_adversarial=False, image_size=(64, 64, 6), shuffle=True, extensions=[".npz"], same_samples=True, n_samples=40000):
        self.paths = []
        self.survey_paths = {"UD":[], "D":[]}
        self.path_index=0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_size = image_size
        self.max_images = 70000
        self.n_samples=n_samples
        self.extensions = extensions
        self.n_epochs = 0
        self.do_color=do_color
        self.do_seg=do_seg
        self.do_mask_band = do_mask_band
        self.do_adversarial = do_adversarial
        self.mean = None
        self.std = None
        self.same_samples = same_samples
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
                        if "_UD" in file :
                            self.survey_paths["UD"].append(filepath)
                        else :
                            self.survey_paths["D"].append(filepath)
                        self.paths.append(filepath)
                        self.file_tracker[filepath] = (0, 0)
        random.shuffle(self.paths)
        random.shuffle(self.survey_paths["UD"])
        random.shuffle(self.survey_paths["D"])

    def _load_data(self):
        random.shuffle(self.paths)
        self.images = []
        self.surveys = []
        self.colors = []

        gc.collect()
        if self.same_samples :
            ud_images = np.zeros((self.n_samples, 64, 64, 6))
            ud_colors = np.zeros((self.n_samples, 5))

            random.shuffle(self.survey_paths["UD"])
            random.shuffle(self.survey_paths["D"])

            ud_index = 0
            path_iter = 0
            while ud_index < self.n_samples :
                path = self.survey_paths["UD"][path_iter]
                data = np.load(path, allow_pickle=True)
                images = data["cube"][..., :6]  # on ne prend que les 5 premières bande
                meta = data["info"]
                self.file_tracker[path] = (self.file_tracker[path][0]+1, images.shape[0])
                indices = np.arange(len(meta))
                random.shuffle(indices)
                images = images[indices]
                meta = meta[indices]
                #print(meta[0], meta[0].dtype)
                colors = np.array([np.array([m["u"] - m["g"], m["g"] - m["r"], m['r'] - m["i"], m["i"] - m["z"], m["z"] - m["y"]]) for m in meta])
                #self.colors.append(colors)
                #masks = np.expand_dims(data["cube"][..., 6], axis=-1)
                    #images = np.sign(images)*(np.sqrt(np.abs(images)+1)-1 )   # PAS BESOIN CAR SAUVEGARDEES NORMALISES
                #images = np.concatenate([images, masks], axis=-1)
                if len(images) + ud_index > ud_images.shape[0] :
                    ud_images[ud_index:] = images[:(ud_images.shape[0] - ud_index)]
                    ud_colors[ud_index:] = colors[:(ud_images.shape[0] - ud_index)]
                else :
                    ud_images[ud_index:ud_index+len(images)] = images
                    ud_colors[ud_index:ud_index+len(images)] = colors

                ud_index += len(images)
            self.images.append(ud_images)
            self.colors.append(ud_colors)
            self.surveys.append(np.ones((self.n_samples)))


            ud_images = np.zeros((self.n_samples, 64, 64, 6))
            ud_colors = np.zeros((self.n_samples, 5))
            ud_index = 0
            path_iter = 0
            while ud_index < self.n_samples :
                path = self.survey_paths["D"][path_iter]
                data = np.load(path, allow_pickle=True)
                meta = data["info"]
                images = data["cube"][..., :6]  # on ne prend que les 5 premières bandes
                self.file_tracker[path] = (self.file_tracker[path][0]+1, images.shape[0])
                indices = np.arange(len(meta))
                random.shuffle(indices)
                images = images[indices]
                meta = meta[indices]
                colors = np.array([np.array([m["u"] - m["g"], m["g"] - m["r"], m['r'] - m["i"], m["i"] - m["z"], m["z"] - m["y"]]) for m in meta])
                #self.colors.append(colors)
                #masks = np.expand_dims(data["cube"][..., 6], axis=-1)
                    #images = np.sign(images)*(np.sqrt(np.abs(images)+1)-1 )   # PAS BESOIN CAR SAUVEGARDEES NORMALISES
                #images = np.concatenate([images, masks], axis=-1)
                if len(images) + ud_index > ud_images.shape[0] :
                    ud_images[ud_index:] = images[:(ud_images.shape[0] - ud_index)]
                    ud_colors[ud_index:] = colors[:(ud_images.shape[0] - ud_index)]
                else :
                    ud_images[ud_index:ud_index+len(images)] = images
                    ud_colors[ud_index:ud_index+len(images)] = colors

                ud_index += len(images)
            self.images.append(ud_images)
            self.colors.append(ud_colors)
            self.surveys.append(np.zeros((self.n_samples)))

            del ud_images
            gc.collect()
            


        else :
            while np.sum([len(cube) for cube in self.images]) < self.max_images :
                path = self.paths[self.path_index]
                self.path_index = (self.path_index+1)%len(self.paths)
                try :
                    data = np.load(path, allow_pickle=True)
                    images = data["cube"][..., :6]  # on ne prend que les 5 premières bandes
                    #masks = np.expand_dims(data["cube"][..., 6], axis=-1)

                    #images = np.sign(images)*(np.sqrt(np.abs(images)+1)-1 )   # PAS BESOIN CAR SAUVEGARDEES NORMALISES
                    #images = np.concatenate([images, masks], axis=-1)  # N, 64, 64, 6
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
            medians = np.median(self.images[..., :6], axis=(0, 1, 2))  # shape (5,) pour chaque channel
            abs_deviation = np.abs(self.images[..., :6] - medians)  # Déviation absolue
            self.mads = np.median(abs_deviation, axis=(0, 1, 2))  # Une MAD par channel
        if self.do_color : 
        #    with mp.Pool() as pool :
        #        colors = pool.map(compute_target, self.images)
        #    self.colors = np.array(colors)
            self.colors = np.concatenate(self.colors, axis=0)
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
        #print("I have load datas :", self.images.shape, self.colors.shape)

        
 

       
    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))
    
    def zoom(self, images) :
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
    
    def center_jitter(self, images):
        batch_size, img_h, img_w, channels = images.shape  # Assumes NHWC format
        
        nx = tf.random.uniform(shape=(batch_size,), minval=-5, maxval=5, dtype=tf.int32)
        ny = tf.random.uniform(shape=(batch_size,), minval=-5, maxval=5, dtype=tf.int32)

        max_pixels_x = tf.minimum(img_h // 2 - nx, img_h // 2 + nx)
        max_pixels_y = tf.minimum(img_w // 2 - ny, img_w // 2 + ny)

        x1 = tf.maximum(0, img_h // 2 - max_pixels_x)
        x2 = tf.minimum(img_h, img_h // 2 + max_pixels_x)
        y1 = tf.maximum(0, img_w // 2 - max_pixels_y)
        y2 = tf.minimum(img_w, img_w // 2 + max_pixels_y)

        cropped_images = tf.image.crop_to_bounding_box(images, x1[0], y1[0], x2[0] - x1[0], y2[0] - y1[0])
        resized_images = tf.image.resize(cropped_images, (64, 64), method='nearest')

        return resized_images


    def gaussian_noise(self, images, apply_prob=0.2) :

        us = tf.random.uniform((tf.shape(images)[0], tf.shape(images)[-1]), minval=1, maxval=3, dtype=tf.float32)  # shape batch, 5  sample un u par image par channel (1 à 3 fois le bruit médian)
        new_sigmas = tf.multiply(us, tf.expand_dims(tf.cast(self.mads, dtype=tf.float32), axis=0))   #    batch, 5 * 1, 5     batch, 5  le mads représente le noise scale et les u à quel point ils s'expriment 
        # on a un sigma par channel par image
        noises = tf.random.normal(shape=tf.shape(images), mean=0, stddev=1, dtype=tf.float32) # batch, 64, 64, 5
        sampled_noises = tf.multiply(noises, tf.expand_dims(tf.expand_dims(tf.math.sqrt(new_sigmas), axis=1), axis=1))  # on multiplie par la racine du sigma pour avoir un bruit 0, sigma
        apply_noise = tf.cast(tf.random.uniform((tf.shape(images)[0], 1, 1, 1)) < apply_prob, tf.float32)
        return images + apply_noise * sampled_noises 
    
    def process_batch(self, images, ebv=None) :
        
        images = self.gaussian_noise(images)
        images = self.zoom(images)
        images = self.center_jitter(images)

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
            #print(batch_colors.shape)
            labels_dict["color"] = batch_colors

        if self.do_adversarial :
            batch_survey = tf.cast(tf.tile(tf.expand_dims(batch_survey, axis=1), [2, 1]), dtype=tf.float32)
            labels_dict["survey"] = batch_survey

        

        #batch_masks = batch_images[:, :, :, 6]
        batch_images = batch_images[:, :, :, :6]
        #batch_masks = tf.cast(tf.tile(batch_masks, [2, 1, 1]), dtype=bool)
        batch_images = tf.cast(tf.tile(batch_images, [2, 1, 1, 1]), dtype=tf.float32)

        augmented_images = self.process_batch(batch_images)
        

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
        if self.do_color :
            self.colors = self.colors[indices]









class SupervisedGenerator(keras.utils.Sequence) :
    def __init__(self, data_path, batch_size, nbins=400, adversarial=False, adv_extensions=["_D.npz"], adversarial_dir=None, contrast=False, apply_log=False) :
        super(SupervisedGenerator, self).__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        self.nbins=nbins
        self.adversarial = adversarial
        self.adversarial_dir = adversarial_dir
        self.adversarial_paths = []
        self.extensions=adv_extensions
        self.contrast = contrast
        self.apply_log = apply_log
        #self.load_data()
        #self.on_epoch_end()
        if self.adversarial :
            self._find_paths(self.adversarial_dir)
            print(self.extensions, self.adversarial_dir)
            print("nb :", len(self.adversarial_paths), self.adversarial_paths)
        self.load_data()
        self.on_epoch_end()


    def _find_paths(self, dir_path) :
        #for dire in dir_paths :
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith(tuple(self.extensions)):
                    filepath = os.path.join(root, file)
                    self.adversarial_paths.append(filepath)
                    print("+1 file")
        random.shuffle(self.adversarial_paths)

    def load_data(self) :
        if isinstance(self.data_path, str) :
            data = np.load(self.data_path, allow_pickle=True)
            images = data["cube"][..., :6]
            meta = data["info"]
        elif isinstance(self.data_path, list) :
            images = []
            meta = []
            for path in self.data_path :
                data = np.load(path, allow_pickle=True)
                images.append(data["cube"][..., :6])
                meta.append(data["info"])
            images = np.concatenate(images, axis=0)
            meta = np.concatenate(meta, axis=0)
          # on ne prend que les 5 premières bandes

        #images = np.sign(images)*(np.sqrt(np.abs(images)+1)-1 )   # PAS BESOIN CAR SAUVEGARDEES NORMALISES
        self.images = images.astype(np.float32)  # N, 64, 64, 6

        
        print(meta[0])
        print(meta[0].dtype)
        print(meta.dtype)
        self.z_values = np.array([m["ZSPEC"] for m in meta])
        self.z_values = self.z_values.astype("float32")
        if self.apply_log :
            self.z_values = np.log(1+self.z_values)
        print("Z VALS", self.z_values)
        #bins_edges = np.linspace(0, 6, 300)
        if self.apply_log :
            bins_edges = np.linspace(np.log(0+1), np.log(1+6), 401)

        else :
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
                images = data["cube"][..., :6]
                masks = np.expand_dims(data["cube"][..., 6], axis=-1)
                adv_imgs.append(np.concatenate([images, masks], axis=-1).astype(np.float32))
            self.adversarial_images = np.concatenate(adv_imgs, axis=0)



    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))

    def process_batch(self, images, ebv=None) :

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


        #batch_masks = batch_images[:, :, :, 6]
        batch_images = batch_images[:, :, :, :6]
        if self.contrast :
            batch_images = tf.tile(batch_images, [2, 1, 1, 1])
            batch_z = tf.tile(batch_z, [2])
            batch_z2 = tf.tile(batch_z2, [2])

        augmented_images = self.process_batch(batch_images)
        if self.adversarial :
            batch_images = self.adversarial_images[index*self.batch_size:(index+1)*self.batch_size]
            if tf.shape(batch_images)[0] < self.batch_size:
                # Compléter le batch avec des images dupliquées ou ignorer (selon ta logique)
                pad_size = self.batch_size - batch_images.shape[0]
                batch_images = tf.concat([batch_images, self.adversarial_images[:pad_size]], axis=0)

            adversarial_images = self.process_batch(batch_images[..., :6], batch_images[..., 6])

            return (augmented_images, adversarial_images), {"pdf":batch_z, "reg":batch_z2}

        return augmented_images, {"pdf":batch_z, "reg":batch_z2}

    def on_epoch_end(self):
        indices = np.arange(0, self.images.shape[0], dtype=np.int32)
        np.random.shuffle(indices)
        self.images = self.images[indices]
        self.z_values = self.z_values[indices]
        self.z_bins = self.z_bins[indices]









class COINGenerator(keras.utils.Sequence) :
    def __init__(self, data_path, batch_size, nbins=400, contrast=True, apply_log=False) :
        super(COINGenerator, self).__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        self.nbins=nbins
        self.contrast=contrast
        self.apply_log = apply_log
        #self.load_data()
        #self.on_epoch_end()
        self.load_data()
        self.on_epoch_end()


    def _find_paths(self, dir_path) :
        #for dire in dir_paths :
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith(tuple(self.extensions)):
                    filepath = os.path.join(root, file)
                    self.adversarial_paths.append(filepath)
                    print("+1 file")
        random.shuffle(self.adversarial_paths)

    def load_data(self) :
        if isinstance(self.data_path, str) :
            data = np.load(self.data_path, allow_pickle=True)
            images = data["cube"][..., :6]
            meta = data["info"]
        elif isinstance(self.data_path, list) :
            images = []
            meta = []
            for path in self.data_path :
                data = np.load(path, allow_pickle=True)
                images.append(data["cube"][..., :6])
                meta.append(data["info"])
            images = np.concatenate(images, axis=0)
            meta = np.concatenate(meta, axis=0)
          # on ne prend que les 5 premières bandes

        #images = np.sign(images)*(np.sqrt(np.abs(images)+1)-1 )   # PAS BESOIN CAR SAUVEGARDEES NORMALISES
        self.images = images.astype(np.float32)  # N, 64, 64, 6

        
        print(meta[0])
        print(meta[0].dtype)
        print(meta.dtype)
        self.z_values = np.array([m["ZSPEC"] for m in meta])
        self.z_values = self.z_values.astype("float32")
        if self.apply_log :
            self.z_values = np.log(1+self.z_values)
        print("Z VALS", self.z_values)
        #bins_edges = np.linspace(0, 6, 300)
        if self.apply_log :
            bins_edges = np.linspace(np.log(0+1), np.log(1+6), 401)

        else :
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

        

        ## considère méthode basique avec intervale arbitraire

        


    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))

    def process_batch(self, images, ebv=None) :

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
            


        #batch_masks = batch_images[:, :, :, 6]
        batch_images = batch_images[:, :, :, :6]
        
        if self.contrast :
            batch_images = tf.tile(batch_images, [2, 1, 1, 1])
            batch_z = tf.tile(batch_z, [2])
            batch_z2 = tf.tile(batch_z2, [2])
        

        augmented_images = self.process_batch(batch_images)
        

        return augmented_images, {"pdf":batch_z, "reg":batch_z2}

    def on_epoch_end(self):
        indices = np.arange(0, self.images.shape[0], dtype=np.int32)
        np.random.shuffle(indices)
        self.images = self.images[indices]
        self.z_values = self.z_values[indices]
        self.z_bins = self.z_bins[indices]







class CoreGenerator(keras.utils.Sequence) :
    def __init__(self, data_path, batch_size, nbins=400, contrast=True, apply_log=False) :
        super(COINGenerator, self).__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        self.nbins=nbins
        self.contrast=contrast
        self.apply_log = apply_log
        #self.load_data()
        #self.on_epoch_end()
        self.load_data()
        self.on_epoch_end()


    def _find_paths(self, dir_path) :
        #for dire in dir_paths :
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith(tuple(self.extensions)):
                    filepath = os.path.join(root, file)
                    self.adversarial_paths.append(filepath)
                    print("+1 file")
        random.shuffle(self.adversarial_paths)

    def load_data(self) :
        if isinstance(self.data_path, str) :
            data = np.load(self.data_path, allow_pickle=True)
            images = data["cube"][..., :6]
            meta = data["info"]
        elif isinstance(self.data_path, list) :
            images = []
            meta = []
            for path in self.data_path :
                data = np.load(path, allow_pickle=True)
                images.append(data["cube"][..., :6])
                meta.append(data["info"])
            images = np.concatenate(images, axis=0)
            meta = np.concatenate(meta, axis=0)
          # on ne prend que les 5 premières bandes

        #images = np.sign(images)*(np.sqrt(np.abs(images)+1)-1 )   # PAS BESOIN CAR SAUVEGARDEES NORMALISES
        self.images = images.astype(np.float32)  # N, 64, 64, 6

        
        print(meta[0])
        print(meta[0].dtype)
        print(meta.dtype)
        self.z_values = np.array([m["ZSPEC"] for m in meta])
        self.z_values = self.z_values.astype("float32")
        if self.apply_log :
            self.z_values = np.log(1+self.z_values)
        print("Z VALS", self.z_values)
        #bins_edges = np.linspace(0, 6, 300)
        if self.apply_log :
            bins_edges = np.linspace(np.log(0+1), np.log(1+6), 401)

        else :
            bins_edges = np.concatenate([np.linspace(0, 4, 381), np.linspace(4, 6, 21)[1:]], axis=0)
        self.z_bins = np.zeros((len(self.z_values)))
        self.core_z_bins = np.zeros((len(self.z_values), len(bins_edges)-1))
        for j, z in enumerate(self.z_values) :
            i = 0
            flag = True
            while flag and i < len(bins_edges)-1 :
                if z >= bins_edges[i] and z < bins_edges[i+1] :
                    self.z_bins[j] = i
                    flag = False
                    self.core_z_bins[j, i] = 1
                    if i > 0 :
                        self.core_z_bins[j, i-1] = 1
                    if i <len(bins_edges)-2 :
                        self.core_z_bins[j, i+1] = 1
                i+=1
            if flag : 
                self.z_bins[j] = i-1
                self.core_z_bins[j, i-1] = 1
                self.core_z_bins[j, i-2] = 1
        print(np.max(self.z_bins), np.min(self.z_bins))
        print("NAN IMGS :",np.any(np.isnan(self.images)))
        print("NAN Z :", np.any(np.isnan(self.z_values)), np.any(np.isnan(self.z_bins)))
        self.z_bins = self.z_bins.astype(np.int32)
        print(self.z_bins)

        

        ## considère méthode basique avec intervale arbitraire

        


    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))

    def process_batch(self, images, ebv=None) :

        images = tf.image.random_flip_left_right(images)
        images = tf.image.random_flip_up_down(images)
        rotations = tf.random.uniform((tf.shape(images)[0],), minval=0, maxval=4, dtype=tf.int32)
        images = tf.map_fn(rotate_image, (images, rotations), dtype=images.dtype)


        return images

    def __getitem__(self, index):
        batch_images = self.images[index*self.batch_size : (index+1)*self.batch_size]
        batch_z = self.z_bins[index*self.batch_size : (index+1)*self.batch_size]
        batch_z2 = self.z_values[index*self.batch_size : (index+1)*self.batch_size]
        batch_core = self.core_z_bins[index*self.batch_size : (index+1)*self.batch_size]
        


        if tf.shape(batch_images)[0] < self.batch_size:
            # Compléter le batch avec des images dupliquées ou ignorer (selon ta logique)
            pad_size = self.batch_size - batch_images.shape[0]
            batch_images = tf.concat([batch_images, self.images[:pad_size]], axis=0)  # Compléter avec les premières images
            batch_z = tf.concat([batch_z, self.z_bins[:pad_size]], axis=0)
            batch_z2 = tf.concat([batch_z2, self.z_values[:pad_size]], axis=0)
            batch_core = tf.concat([batch_core, self.core_z_bins[:pad_size]], axis=0)
            


        #batch_masks = batch_images[:, :, :, 6]
        batch_images = batch_images[:, :, :, :6]
        
        if self.contrast :
            batch_images = tf.tile(batch_images, [2, 1, 1, 1])
            batch_z = tf.tile(batch_z, [2])
            batch_z2 = tf.tile(batch_z2, [2])
            batch_core = tf.tile(batch_core, [2, 1])
        

        augmented_images = self.process_batch(batch_images)
        

        return augmented_images, {"pdf":batch_z, "reg":batch_z2, "core":batch_core}

    def on_epoch_end(self):
        indices = np.arange(0, self.images.shape[0], dtype=np.int32)
        np.random.shuffle(indices)
        self.images = self.images[indices]
        self.z_values = self.z_values[indices]
        self.z_bins = self.z_bins[indices]
        self.core_z_bins = self.core_z_bins[indices]

