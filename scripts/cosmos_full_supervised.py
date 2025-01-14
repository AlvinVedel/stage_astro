import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from astro_metrics import Bias, SigmaMAD, OutlierFraction
import matplotlib.pyplot as plt
from vit_layers import ViT_backbone


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

    input_img = keras.Input((64, 64, 5))
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

    pdf = layers.Dense(400, activation='softmax', name='pdf')(l2)
    regression = layers.Dense(1, activation='relu', name='reg')(l3)
    if with_ebv :
        model = keras.Model(inputs=[input_img, ebv_input], outputs=[pdf, regression])
    else :
        model = keras.Model(inputs=[input_img], outputs=[pdf, regression])
    return model




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

    l2 = layers.Dense(1024)(l1)
    l2 = layers.PReLU()(l2)
    l3 = layers.Dense(1024)(l2)
    l3 = layers.PReLU()(l3)
    pdf = layers.Dense(400, activation='softmax', name='pdf')(l3)
    l3b = layers.Dense(512, activation='tanh')(l2)
    reg = layers.Dense(1, activation='linear', name='reg')(l3b)

    return keras.Model(inputs=inp, outputs=[pdf, reg])



def ViT_model() :
    back = ViT_backbone(256, 4, 8, 'average')
    inp = keras.Input((64, 64, 5))
    x = back(inp)
    x = layers.Dense(1024)(x)
    x1 = layers.PReLU()(x)
    x2 = layers.Dense(1024)(x1)
    x2 = layers.PReLU()(x2)
    pdf = layers.Dense(400, activation='softmax', name='pdf')(x2)
    x3 = layers.Dense(512, activation='tanh')(x1)
    reg = layers.Dense(1, activation='linear', name='reg')(x3)
    return keras.Model(inp, [pdf, reg])



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
base_names = ["b1_1", "b2_1", "b3_1"]

for base in base_names :

    model = backbone()
    gen = DataGen(base_path+base+"_v2.npz", batch_size=32)
    n_epochs = 50
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss={"pdf" : tf.keras.losses.SparseCategoricalCrossentropy(), "reg":tf.keras.losses.MeanAbsoluteError()}, 
                  metrics= {"pdf":["accuracy"], "reg" :[Bias(name='global_bias'), SigmaMAD(name='global_smad'), OutlierFraction(name='global_outl'),Bias(inf=0, sup=0.4, name='bias1'), Bias(inf=0.4, sup=2, name='bias2'), Bias(inf=2, sup=4, name='bias3'), Bias(inf=4, sup=6, name='bias4'), 
                  SigmaMAD(inf=0, sup=0.4, name='smad1'), SigmaMAD(inf=0.4, sup=2, name='smad2'), SigmaMAD(inf=2, sup=4, name='smad3'), SigmaMAD(inf=4, sup=6, name='smad4'), OutlierFraction(inf=0, sup=0.4, name='outl1'), OutlierFraction(inf=0.4, sup=2, name='outl2'), OutlierFraction(inf=2, sup=4, name='outl3'), OutlierFraction(inf=4, sup=6, name='outl4')]})
    history = model.fit(gen, epochs=n_epochs, callbacks=[LearningRateScheduler()])

    model.save_weights("/lustre/fswork/projects/rech/dnz/ull82ct/astro/model_save/checkpoints_supervised/vit_backbone_supervised_v2_"+base+".weights.h5")

    print("history keys :", history.history.keys())

    plt.plot(np.arange(1, n_epochs+1), history.history["reg_loss"])
    plt.xlabel("epochs")
    plt.ylabel("loss (mae)")
    plt.title("supervised loss")
    plt.savefig("/lustre/fswork/projects/rech/dnz/ull82ct/astro/plots/simCLR/simCLR_finetune/loss_vit_base="+base+".png")
    plt.close()


    plt.plot(np.arange(1, n_epochs+1), history.history["reg_global_bias"], label='biais moyen')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_bias1"], label='[0, 0.4[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_bias2"], label='[0.4, 2[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_bias3"], label='[2, 4[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_bias4"], label='[4, 6[')
    plt.xlabel("epochs")
    plt.ylabel("Bias")
    plt.legend()
    plt.title("supervised bias")
    plt.savefig("/lustre/fswork/projects/rech/dnz/ull82ct/astro/plots/simCLR/simCLR_finetune/bias_vit_base="+base+".png")
    plt.close()

    plt.plot(np.arange(1, n_epochs+1), history.history["reg_global_smad"], label='smad moyen')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_smad1"], label='[0, 0.4[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_smad2"], label='[0.4, 2[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_smad3"], label='[2, 4[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_smad4"], label='[4, 6[')
    plt.xlabel("epochs")
    plt.ylabel("Sigma MAD")
    plt.legend()
    plt.title("supervised smad")
    plt.savefig("/lustre/fswork/projects/rech/dnz/ull82ct/astro/plots/simCLR/simCLR_finetune/smad_vit_base="+base+".png")
    plt.close()


    plt.plot(np.arange(1, n_epochs+1), history.history["reg_global_outl"], label='outl moyen')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_outl1"], label='[0, 0.4[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_outl2"], label='[0.4, 2[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_outl3"], label='[2, 4[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_outl4"], label='[4, 6[')
    plt.xlabel("epochs")
    plt.ylabel("Outlier Fraction")
    plt.legend()
    plt.title("supervised outl")
    plt.savefig("/lustre/fswork/projects/rech/dnz/ull82ct/astro/plots/simCLR/simCLR_finetune/outl_vit_base="+base+".png")
    plt.close()







#model.backbone.save_weights("sdss_backbone.weights.h5")
#model.classifier.save_weights("sdss_classifier.weights.h5")





