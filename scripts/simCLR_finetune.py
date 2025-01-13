
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from contrastiv_model import simCLR, ContrastivLoss, simCLR_adversarial, simCLRcolor1, simCLRcolor1, simCLRcolor2, simCLRmultitask
import os 
from astro_metrics import Bias, SigmaMAD, OutlierFraction
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import time


def backbone(bn=True, adv=False, return_all=False) :
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
    if adv :
        return keras.Model(inputs=inp, outputs=[l1, flat])
    elif return_all :
        return keras.Model(inputs=inp, outputs=[l1, flat, c4])
    else :
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

def output_head(input_shape=1024) :
    inp = keras.Input((input_shape))
    l1 = layers.Dense(1024)(inp)
    l1 = layers.PReLU()(l1)
    l2 = layers.Dense(1024)(l1)
    l2 = layers.PReLU()(l2)
    pdf = layers.Dense(400, activation='softmax', name='pdf')(l2)
    l3 = layers.Dense(512, activation='tanh')(l1)
    reg = layers.Dense(1, activation='linear', name='reg')(l3)
    return keras.Model(inp, [pdf, reg])


def color_mlp():
    inp = keras.Input((1024))
    x = layers.Dense(256)(inp)
    x = layers.PReLU()(x)
    x = layers.Dense(256)(x)
    x = layers.PReLU()(x)
    out = layers.Dense(4, activation='linear')(x)
    return keras.Model(inp, out)


def segmentor(input_shape1=1024, input_shape2=(32, 32, 128)) :

    inp = keras.Input(input_shape1)
    deflat = layers.Reshape((8, 8, 16))(inp)
    c1 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(deflat)  # 8 8 256
    c1 = layers.PReLU()(c1)
    c1_r = layers.Reshape((16, 16, 64))(c1)  # 16 16 64
    c2 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(c1_r)
    c2 = layers.PReLU()(c2)
    c2_r = layers.Reshape((32, 32, 64))(c2)  #32 32 64
    c3 = layers.Conv2D(128, strides=1, kernel_size=3, padding='same')(c2_r)
    c3 = layers.PReLU()(c3)
    inp2 = keras.Input(input_shape2)
    conc = layers.Concatenate()([c3, inp2])
    c4 = layers.Conv2D(256, kernel_size=3, padding='same', strides=1)(conc)  # 32 32 256
    c4 = layers.PReLU()(c4)
    c4_r = layers.Reshape((64, 64, 64))(c4)
    segmentation = layers.Conv2D(1, kernel_size=3, padding='same', strides=1, activation='sigmoid')(c4_r)
    return keras.Model([inp, inp2], segmentation)



def deconvolutor(input_shape1=1024) :

    inp = keras.Input(input_shape1)
    l1 = layers.Dense(1024)(inp)
    l1 = layers.PReLU()(l1)

    deflat = layers.Reshape((8, 8, 16))(inp)
    c1 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(deflat)  # 8 8 256
    c1 = layers.PReLU()(c1)
    c1 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(c1)  # 8 8 256
    c1 = layers.PReLU()(c1)
    c1_r = layers.Reshape((16, 16, 64))(c1)  # 16 16 64
    c2 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(c1_r)
    c2 = layers.PReLU()(c2)
    c2 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(c2)
    c2 = layers.PReLU()(c2)
    c2_r = layers.Reshape((32, 32, 64))(c2)  #32 32 64
    c3 = layers.Conv2D(256, strides=1, kernel_size=3, padding='same')(c2_r)
    c3 = layers.PReLU()(c3)
    c4 = layers.Conv2D(256, kernel_size=3, padding='same', strides=1)(c3)  # 32 32 256
    c4 = layers.PReLU()(c4)
    c4_r = layers.Reshape((64, 64, 64))(c4)
    c5 = layers.Conv2D(256, strides=1, kernel_size=3, padding='same')(c4_r)
    c5 = layers.PReLU()(c5)
    reconstruction = layers.Conv2D(5, kernel_size=3, padding='same', strides=1, activation='linear')(c5)
    return keras.Model(inp, reconstruction)


class FineTuneModel(keras.Model) :
    def __init__(self, back, head, train_back=False, adv=False, multi=False) :
        super(FineTuneModel, self).__init__()
        self.backbone = back
        self.head = head
        self.adv = adv
        self.multi = multi
        self.train_back = train_back
    
    def call(self, inputs, training=True) :
        if self.adv :
            latent, flat = self.backbone(inputs, training=self.train_back)
        elif self.multi :
            latent, flat, c4 = self.backbone(inputs, training=self.train_back)
        else :
            latent = self.backbone(inputs, training=self.train_back)
        probs, reg = self.head(latent, training=training)
        return {"pdf":probs, "reg":reg}


    
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
        return augmented_images, {"pdf":batch_z, "reg":batch_z2}

    def on_epoch_end(self):
        indices = np.arange(0, self.images.shape[0], dtype=np.int32)
        np.random.shuffle(indices)
        self.images = self.images[indices]
        self.z_values = self.z_values[indices]
        self.z_bins = self.z_bins[indices]
    



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

weights_path = "/lustre/fswork/projects/rech/dnz/ull82ct/astro/model_save/checkpoints_simCLR_UD_D/simCLR_cosmos_bnTrue_200_Multi_TTTF_.weights.h5"
name = "UD200_Multitask"


for base in ["b1_1", "b2_1", "b3_1"] :

    data_gen = DataGen("/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/finetune/"+base+"_v2.npz", batch_size=32)

    # PARTIE 1
    #model = simCLR_adversarial(backbone(bn, adv=True), mlp(1024), mlp_adversarial(1024))
    #model = simCLRcolor2(backbone(bn), mlp(1024))
    #model = simCLR(backbone(bn), mlp(1024))
    model = simCLRmultitask(backbone(bn=bn, return_all=True), mlp(1024), do_color=True, color_head=color_mlp(), do_seg=True, segmentor=segmentor(),
                        do_reco=True, deconvolutor=deconvolutor())
    #model = simCLRcolor1(backbone(bn), mlp(1024), color_mlp())
    model(np.random.random((32, 64, 64, 5)))
    model.load_weights(weights_path)

    extracteur = model.backbone
    predictor = output_head(1024)

    model1 = FineTuneModel(extracteur, predictor, train_back=False, adv=False, multi=True)
    model1.compile(optimizer=keras.optimizers.Adam(1e-4), loss={"pdf" : tf.keras.losses.SparseCategoricalCrossentropy(), "reg":tf.keras.losses.MeanAbsoluteError()}, metrics= {"pdf":["accuracy"], "reg" :[Bias(name='global_bias'), SigmaMAD(name='global_smad'), OutlierFraction(name='global_outl'),Bias(inf=0, sup=0.4, name='bias1'), Bias(inf=0.4, sup=2, name='bias2'), Bias(inf=2, sup=4, name='bias3'), Bias(inf=4, sup=6, name='bias4'), 
                  SigmaMAD(inf=0, sup=0.4, name='smad1'), SigmaMAD(inf=0.4, sup=2, name='smad2'), SigmaMAD(inf=2, sup=4, name='smad3'), SigmaMAD(inf=4, sup=6, name='smad4'), OutlierFraction(inf=0, sup=0.4, name='outl1'), OutlierFraction(inf=0.4, sup=2, name='outl2'), OutlierFraction(inf=2, sup=4, name='outl3'), OutlierFraction(inf=4, sup=6, name='outl4')]})
    n_epochs = 50
    history = model1.fit(data_gen, epochs=n_epochs, callbacks=[LearningRateDecay()])
    model1.save_weights("/lustre/fswork/projects/rech/dnz/ull82ct/astro/model_save/checkpoints_simCLR_finetune/simCLR_finetune_HeadOnly_base="+base+"_model="+name+".weights.h5")
    print("TRAIN ENDED FOR ", "simCLR_finetune_HeadOnly_base="+base+"_model="+name+".weights.h5")
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_loss"])
    plt.xlabel("epochs")
    plt.ylabel("loss (mae)")
    plt.title("finetuning loss")
    plt.savefig("/lustre/fswork/projects/rech/dnz/ull82ct/astro/plots/simCLR/simCLR_finetune/loss_HeadOnly_base="+base+"_model="+name+".png")
    plt.close()

    plt.plot(np.arange(1, n_epochs+1), history.history["reg_global_bias"], label='biais moyen')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_bias1"], label='[0, 0.4[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_bias2"], label='[0.4, 2[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_bias3"], label='[2, 4[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_bias4"], label='[4, 6[')
    plt.xlabel("epochs")
    plt.ylabel("Bias")
    plt.legend()
    plt.title("finetuning bias")
    plt.savefig("/lustre/fswork/projects/rech/dnz/ull82ct/astro/plots/simCLR/simCLR_finetune/bias_HeadOnly_base="+base+"_model="+name+".png")
    plt.close()

    plt.plot(np.arange(1, n_epochs+1), history.history["reg_global_smad"], label='smad moyen')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_smad1"], label='[0, 0.4[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_smad2"], label='[0.4, 2[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_smad3"], label='[2, 4[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_smad4"], label='[4, 6[')
    plt.xlabel("epochs")
    plt.ylabel("Sigma MAD")
    plt.legend()
    plt.title("finetuning smad")
    plt.savefig("/lustre/fswork/projects/rech/dnz/ull82ct/astro/plots/simCLR/simCLR_finetune/smad_HeadOnly_base="+base+"_model="+name+".png")
    plt.close()


    plt.plot(np.arange(1, n_epochs+1), history.history["reg_global_outl"], label='outl moyen')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_outl1"], label='[0, 0.4[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_outl2"], label='[0.4, 2[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_outl3"], label='[2, 4[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_outl4"], label='[4, 6[')
    plt.xlabel("epochs")
    plt.ylabel("Outlier Fraction")
    plt.legend()
    plt.title("finetuning outl")
    plt.savefig("/lustre/fswork/projects/rech/dnz/ull82ct/astro/plots/simCLR/simCLR_finetune/outl_HeadOnly_base="+base+"_model="+name+".png")
    plt.close()

    # PARTIE 2
    #model = simCLR_adversarial(backbone(bn, adv=True), mlp(1024), mlp_adversarial(1024))
    #model = simCLRcolor2(backbone(bn), mlp(1024))
    #model = simCLRcolor1(backbone(bn), mlp(1024), color_mlp())
    model = simCLRmultitask(backbone(bn=bn,return_all=True), mlp(1024), do_color=True, color_head=color_mlp(), do_seg=True, segmentor=segmentor(),
                        do_reco=True, deconvolutor=deconvolutor())
    #model = simCLR(backbone(bn), mlp(1024))
    model(np.random.random((32, 64, 64, 5)))
    model.load_weights(weights_path)

    extracteur = model.backbone
    predictor = output_head(1024)

    model1 = FineTuneModel(extracteur, predictor, train_back=True, adv=False, multi=True)
    model1.compile(optimizer=keras.optimizers.Adam(1e-4), loss={"pdf" : tf.keras.losses.SparseCategoricalCrossentropy(), "reg":tf.keras.losses.MeanAbsoluteError()}, metrics= {"pdf":["accuracy"], "reg" :[Bias(name='global_bias'), SigmaMAD(name='global_smad'), OutlierFraction(name='global_outl'),Bias(inf=0, sup=0.4, name='bias1'), Bias(inf=0.4, sup=2, name='bias2'), Bias(inf=2, sup=4, name='bias3'), Bias(inf=4, sup=6, name='bias4'), 
                  SigmaMAD(inf=0, sup=0.4, name='smad1'), SigmaMAD(inf=0.4, sup=2, name='smad2'), SigmaMAD(inf=2, sup=4, name='smad3'), SigmaMAD(inf=4, sup=6, name='smad4'), OutlierFraction(inf=0, sup=0.4, name='outl1'), OutlierFraction(inf=0.4, sup=2, name='outl2'), OutlierFraction(inf=2, sup=4, name='outl3'), OutlierFraction(inf=4, sup=6, name='outl4')]})
    
    history = model1.fit(data_gen, epochs=n_epochs, callbacks=[LearningRateDecay()])
    model1.save_weights("/lustre/fswork/projects/rech/dnz/ull82ct/astro/model_save/checkpoints_simCLR_finetune/simCLR_finetune_ALL_base="+base+"_model="+name+".weights.h5")
    print("TRAIN ENDED FOR ", "simCLR_finetune_ALL_base="+base+"_model="+name+".weights.h5")
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_loss"])
    plt.xlabel("epochs")
    plt.ylabel("loss (mae)")
    plt.title("finetuning loss")
    plt.savefig("/lustre/fswork/projects/rech/dnz/ull82ct/astro/plots/simCLR/simCLR_finetune/loss_ALL_base="+base+"_model="+name+".png")
    plt.close()



    plt.plot(np.arange(1, n_epochs+1), history.history["reg_global_bias"], label='biais moyen')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_bias1"], label='[0, 0.4[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_bias2"], label='[0.4, 2[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_bias3"], label='[2, 4[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_bias4"], label='[4, 6[')
    plt.xlabel("epochs")
    plt.ylabel("Bias")
    plt.legend()
    plt.title("finetuning bias")
    plt.savefig("/lustre/fswork/projects/rech/dnz/ull82ct/astro/plots/simCLR/simCLR_finetune/bias_ALL_base="+base+"_model="+name+".png")
    plt.close()

    plt.plot(np.arange(1, n_epochs+1), history.history["reg_global_smad"], label='smad moyen')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_smad1"], label='[0, 0.4[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_smad2"], label='[0.4, 2[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_smad3"], label='[2, 4[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_smad4"], label='[4, 6[')
    plt.xlabel("epochs")
    plt.ylabel("Sigma MAD")
    plt.legend()
    plt.title("finetuning smad")
    plt.savefig("/lustre/fswork/projects/rech/dnz/ull82ct/astro/plots/simCLR/simCLR_finetune/smad_ALL_base="+base+"_model="+name+".png")
    plt.close()


    plt.plot(np.arange(1, n_epochs+1), history.history["reg_global_outl"], label='outl moyen')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_outl1"], label='[0, 0.4[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_outl2"], label='[0.4, 2[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_outl3"], label='[2, 4[')
    plt.plot(np.arange(1, n_epochs+1), history.history["reg_outl4"], label='[4, 6[')
    plt.xlabel("epochs")
    plt.ylabel("Outlier Fraction")
    plt.legend()
    plt.title("finetuning outl")
    plt.savefig("/lustre/fswork/projects/rech/dnz/ull82ct/astro/plots/simCLR/simCLR_finetune/outl_ALL_base="+base+"_model="+name+".png")
    plt.close()

