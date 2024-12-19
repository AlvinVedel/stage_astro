import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from contrastiv_model import simCLR, ContrastivLoss
import os 
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import time


def backbone(bn=True) :
    inp = keras.Input((64, 64, 5))
    c1 = layers.Conv2D(96, padding='same', strides=1, kernel_size=3)(inp) # 64
    c1 = layers.PReLU()(c1) 
    c2 = layers.Conv2D(96, padding='same', kernel_size=3, strides=1, activation='tanh')(c1)  #64
    p1 = layers.AveragePooling2D((2, 2))(c2)  # 32
    c3 = layers.Conv2D(128, padding='same', strides=1, kernel_size=3)(p1)
    c3 = layers.PReLU()(c3)
    c4 = layers.Conv2D(128, padding='same', kernel_size=3, strides=1)(c3)  #32
    c4 = layers.PReLU(name='c4')(c4) 
    p2 = layers.AveragePooling2D((2, 2))(c4)  # 16
    c5 = layers.Conv2D(256, padding='same', strides=1, kernel_size=3)(p2) #16
    c5 = layers.PReLU()(c5)
    c6 = layers.Conv2D(256, padding='same', kernel_size=3, strides=1)(c5)  #16
    c6 = layers.PReLU()(c6)
    p3 = layers.AveragePooling2D((2, 2))(c6) # 8
    c7 = layers.Conv2D(256, kernel_size=3, strides=1, padding='valid')(p3) # 6
    c7 = layers.PReLU()(c7)
    c8 = layers.Conv2D(256, kernel_size=3, strides=1, padding='valid')(c7) # 4
    c8 = layers.PReLU()(c8)
    c9 = layers.Conv2D(256, padding='valid', kernel_size=3, strides=1)(c8) # 2, 2, 256
    c9 = layers.PReLU()(c9)
    
    flat = layers.Flatten(name='flatten')(c9) # 2, 2, 256 = 1024 

    l1 = layers.Dense(1024)(flat) 
    l1 = layers.PReLU()(l1)
    if bn :
        l1 = layers.BatchNormalization()(l1)

    return keras.Model(inputs=inp, outputs=l1)

def mlp(input_shape=100):
    latent_input = keras.Input((input_shape))
    x = layers.Dense(512, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(5e-7), bias_regularizer=tf.keras.regularizers.l2(5e-7))(latent_input)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)
    x = layers.Dense(256, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(5e-7), bias_regularizer=tf.keras.regularizers.l2(5e-7))(x)
    return keras.Model(latent_input, x)

def mlp_adversarial(input_shape=1024) :
    latent_inp = keras.Input((input_shape))
    x = layers.BatchNormalization()(latent_inp)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(2, activation='softmax')(x)
    return keras.Model(latent_inp, x)

def regression_head(input_shape=1024) :
    inp = keras.Input((input_shape))
    l1 = layers.Dense(1024)(inp)
    l1 = layers.PReLU()(l1)
    l2 = layers.Dense(1024)(l1)
    l2 = layers.PReLU()(l2)
    reg = layers.Dense(1, activation='linear')(l2)
    return keras.Model(inp, reg)



class FineTuneModel(keras.Model) :
    def __init__(self, back, head, train_back=False) :
        super(FineTuneModel, self).__init__()
        self.backbone = back
        self.head = head
        self.train_back = train_back
    
    def call(self, inputs, training=True) :
        latent = self.backbone(inputs, training=self.train_back)
        pred = self.head(latent, training=training)
        return pred
    
def rotate_image(inputs):
    image, rotation = inputs
    return tf.image.rot90(image, rotation)

class DataGen(keras.Sequence) :
    def __init__(self, data_path, batch_size) :
        super(DataGen, self).__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        self.load_data()
        self.on_epoch_end()
    
    def load_data(self) :
        data = np.load(self.data_path, allow_pickle=True)
        images = data["cube"][..., :5]  # on ne prend que les 5 premières bandes
        masks = np.expand_dims(data["cube"][..., 5], axis=-1)

        images = np.sign(images)*(np.sqrt(np.abs(images)+1)-1 )   # PAS BESOIN CAR SAUVEGARDEES NORMALISES
        self.images = np.concatenate([images, masks], axis=-1)  # N, 64, 64, 6

        meta = data["meta"]
        self.z_values = meta[:, 6]

    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))
    
    def process_batch(self, images, masks, ebv=None) :
        
        images = tf.image.random_flip_left_right(images)
        images = tf.image.random_flip_up_down(images)
        rotations = tf.random.uniform((tf.shape(images)[0],), minval=0, maxval=4, dtype=tf.int32)    
        images = tf.map_fn(rotate_image, (images, rotations), dtype=images.dtype)

        return images

    def __getitem__(self, index):
        batch_images = self.images[index * self.batch_size:(index + 1) * self.batch_size]
        batch_z = self.z_values[index * self.batch_size : (index+1)*self.batch_size]
          
        if tf.shape(batch_images)[0] < self.batch_size:
            # Compléter le batch avec des images dupliquées ou ignorer (selon ta logique)
            pad_size = self.batch_size - batch_images.shape[0]
            batch_images = tf.concat([batch_images, self.images[:pad_size]], axis=0)  # Compléter avec les premières images
            batch_z = tf.concat([batch_z, self.z_values[:pad_size]], axis=0)
          
                    
        batch_masks = batch_images[:, :, :, 5]
        batch_images = batch_images[:, :, :, :5]
        
        
        augmented_images = self.process_batch(batch_images, batch_masks)
        return augmented_images, batch_z

    def on_epoch_end(self):
        indices = np.arange(0, self.images.shape[0], dtype=np.int32)
        np.random.shuffle(indices)
        self.images = self.images[indices]
        self.z_values = self.z_values[indices]

class LearningRateDecay(tf.keras.callbacks.Callback):
    def __init__(self, decay_factor=0.1):
        super(LearningRateDecay, self).__init__()
        self.decay_factor = decay_factor

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 35 or epoch == 45 :
            old_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            new_lr = old_lr * self.decay_factor
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)


bn=True

weights_path = "/lustre/fswork/projects/rech/dnz/ull82ct/astro/model_save/checkpoints_simCLR_UD/simCLR_cosmos_bnTrue_100.weights.h5"
name = "UD"


for base in ["b1_1", "b1_2", "b2_1", "b2_2", "b3_1.npz", "b3_2.npz"] :

    data_gen = DataGen("/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/finetune/"+base+".npz", batch_size=32)

    # PARTIE 1
    model = simCLR(backbone(bn), mlp(1024))
    model(np.random.random((32, 64, 64, 5)))
    model.load_weights(weights_path)

    extracteur = model.backbone
    predictor = regression_head(1024)

    model1 = FineTuneModel(extracteur, predictor, train_back=False)
    model1.compile(optimizer=keras.optimizers.Adam(1e-4), loss="mse")
    history = model1.fit(data_gen, epochs=50, callbacks=[LearningRateDecay()])
    model1.save_weights("simCLR_finetune_HeadOnly_base="+base+"_model="+name+".weights.h5")

    plt.plot(np.arange(1, 51), history["loss"])
    plt.xlabel("epochs")
    plt.ylabel("loss (mse)")
    plt.title("finetuning loss")
    plt.savefig("/lustre/fswork/projects/rech/dnz/ull82ct/astro/model_save/checkpoints_simCLR_finetune/simCLR_finetune_HeadOnly_base="+base+"_model="+name+".png")


    # PARTIE 2
    model = simCLR(backbone(bn), mlp(1024))
    model(np.random.random((32, 64, 64, 5)))
    model.load_weights(weights_path)

    extracteur = model.backbone
    predictor = regression_head(1024)

    model1 = FineTuneModel(extracteur, predictor, train_back=False)
    model1.compile(optimizer=keras.optimizers.Adam(1e-4), loss="mse")
    history = model1.fit(data_gen, epochs=50, callbacks=[LearningRateDecay()])
    model1.save_weights("simCLR_finetune_ALL_base="+base+"_model="+name+".weights.h5")

    plt.plot(np.arange(1, 51), history["loss"])
    plt.xlabel("epochs")
    plt.ylabel("loss (mse)")
    plt.title("finetuning loss")
    plt.savefig("/lustre/fswork/projects/rech/dnz/ull82ct/astro/model_save/checkpoints_simCLR_finetune/simCLR_finetune_ALL_base="+base+"_model="+name+".png")


