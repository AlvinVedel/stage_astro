import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'

def inception_block(input):
    c1 = layers.Conv2D(101, activation='relu', kernel_size=1, strides=1, padding='same')(input)
    c2 = layers.Conv2D(101, activation='relu', kernel_size=1, strides=1, padding='same')(input)
    c3 = layers.Conv2D(101, activation='relu', kernel_size=1, strides=1, padding='same')(input)

    c4 = layers.Conv2D(156, activation='relu', kernel_size=3, strides=1, padding='same')(c1)
    c5 = layers.Conv2D(156, activation='relu', kernel_size=5, strides=1, padding='same')(c2)
    c6 = layers.AveragePooling2D((2, 2), strides=1, padding='same')(c3)

    c7 = layers.Conv2D(109, activation='relu', kernel_size=1, strides=1, padding='same')(input)

    conc = layers.Concatenate(axis=-1)([c4, c5, c6, c7])   # 156 + 156 + 101 + 109 = 522
    return conc

def create_model(with_ebv=False) :

    input_img = keras.Input((64, 64, 9))
    #input_ebv = keras.Input((1,))
    conv1 = layers.Conv2D(96, kernel_size=5, activation='relu', strides=1, padding='same', name='c1')(input_img)
    conv2 = layers.Conv2D(96, kernel_size=3, activation='tanh', strides=1, padding='same', name='c2')(conv1)
    avg_pool = layers.AveragePooling2D(pool_size=(2, 2), strides=2, name='p1')(conv2)  # batch, 32, 32, 96

    incep1 = inception_block(avg_pool)
    incep2 = inception_block(incep1)
    incep3 = inception_block(incep2)

    avg_pool2 = layers.AveragePooling2D((2, 2), strides=2, name='p2')(incep3) # 16, 16, 522

    incep4 = inception_block(avg_pool2)
    incep5 = inception_block(incep4)

    avg_pool3 = layers.AveragePooling2D((2, 2), strides=2, name='p3')(incep5)  # 8, 8, 522

    incep6 = inception_block(avg_pool3) 

    conv3 = layers.Conv2D(96, kernel_size=3, activation='relu', padding='valid', strides=1, name='c3')(incep6)  # 6, 6, 96
    conv4 = layers.Conv2D(96, kernel_size=3, activation='relu', padding='valid', strides=1, name='c4')(conv3)  # 4, 4, 96
    conv5 = layers.Conv2D(96, kernel_size=3, activation='relu', padding='valid', strides=1, name='c5')(conv4)   # 2, 2, 96
    avg_pool4 = layers.AveragePooling2D((2, 2), strides=1, name='p4')(conv5)   # 1, 1, 96
    resh = layers.Reshape(target_shape=(96,))(avg_pool4) # batch, 96
    if with_ebv : 
        ebv_input = keras.Input((1))
        resh = layers.Concatenate(axis=-1)([resh, ebv_input])
    #conc = layers.Concatenate()([resh, input_ebv])
    l1 = layers.Dense(1024, activation='relu', name='l1')(resh)

    l2 = layers.Dense(1024, activation='relu', name='l2')(l1)
    l3 = layers.Dense(512, activation='tanh', name='l3')(l1)

    pdf = layers.Dense(300, activation='softmax', name='pdf')(l2)
    regression = layers.Dense(1, activation='relu', name='reg')(l3)
    if with_ebv :
        model = keras.Model(inputs=[input_img, ebv_input], outputs=[pdf, regression])
    else :
        model = keras.Model(inputs=[input_img], outputs=[pdf, regression])
    return model

def rotate_image(inputs):
    image, rotation = inputs
    return tf.image.rot90(image, rotation)

class DataGen(keras.utils.Sequence) :
    def __init__(self, data_path, batch_size, nbins=150) :
        super(DataGen, self).__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        self.nbins=nbins
        self.load_data()
        self.on_epoch_end()

    def load_data(self) :
        data = np.load(self.data_path, allow_pickle=True)
        images = data["cube"][..., :5]  # on ne prend que les 5 premières bandes
        masks = np.expand_dims(data["cube"][..., 5], axis=-1)

        images = np.sign(images)*(np.sqrt(np.abs(images)+1)-1 )   # PAS BESOIN CAR SAUVEGARDEES NORMALISES
        self.images = np.concatenate([images, masks], axis=-1).astype(np.float32)  # N, 64, 64, 6

        meta = data["info"]
        self.z_values = meta[:, 6]
        self.z_values = self.z_values.astype("float32")
        print("Z VALS", self.z_values)
        
        bins_edges = np.concatenate([np.linspace(0, 4, 380), np.linspace(4, 6, 21)[1:]], axis=0)
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
                self.z_bins[j] = i

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
        return augmented_images, (batch_z, batch_z2)

    def on_epoch_end(self):
        indices = np.arange(0, self.images.shape[0], dtype=np.int32)
        np.random.shuffle(indices)
        self.images = self.images[indices]
        self.z_values = self.z_values[indices]
        self.z_bins = self.z_bins[indices]
    

class LearningRateScheduler(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if epoch==35 or epoch==45 :
            old_lr = self.model.optimizer.lr.numpy()  # On récupère le LR actuel
            new_lr = old_lr / 10  # Diviser le LR par 10
            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
            print(f"\nEpoch {epoch+1}: Learning rate is reduced to {new_lr}")

base_path = "/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/finetune/"
base_names = ["b1_1", "b1_2", "b2_1", "b2_2", "b3_1", "b3_2"]

for base in base_names :

    model = create_model()
    gen = DataGen(base_path+base_names+".npz", batch_size=32)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=[tf.keras.losses.SparseCategoricalCrossentropy(), tf.keras.losses.MeanSquaredError()])
    model.fit(gen, epochs=50, callbacks=[LearningRateScheduler()])

    model.save_weights("treyer_supervised_"+base+".weights.h5")







#model.backbone.save_weights("sdss_backbone.weights.h5")
#model.classifier.save_weights("sdss_classifier.weights.h5")





