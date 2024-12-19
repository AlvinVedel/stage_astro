import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from contrastiv_model import simCLR

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


base_path = "/lustre/fswork/projects/rech/dnz/ull82ct/astro/"


#model = FineTuneModel(backbone(True), head=regression_head(1024))
model = create_model()
treyer = True
model.load_weights(base_path+"model_save/checkpoints_supervised/treyer_supervised_b1_1.weights.h5")
model_name='treyer_b1_1'

directory = base_path+"data/spec/"

npz_files = [f for f in os.listdir(directory) if f.endswith('spec_UD.npz')]



true_z = []
pred_z = []


def extract_z(tup) :
    return tup[40]

for file in npz_files :
    data = np.load(file, allow_pickles=True)
    images = data["cube"]
    meta = data["info"]
    z = np.array([extract_z(m) for m in meta])
    true_z.append(z)
    if treyer :
        probas, reg = model.predict(images)

    else :
        reg = model.predict(images)

    pred_z.append(reg)


true_z = np.concatenate(true_z, axis=0)
pred_z = np.concatenate(pred_z, axis=0)

#### MATRICE DE CHALEUR
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

xy = np.vstack([true_z, pred_z])
density = gaussian_kde(xy)(xy)

plt.scatter(true_z, pred_z, c=density, cmap='hot', s=5)
plt.colorbar(label='density')
plt.xlabel("true Z")
plt.ylabel("pred Z")
plt.xlim((-1, 6))
plt.ylim((-1, 6))
plt.title("prediction density heatmap")
plt.savefig("density_heatmap_"+model_name+".png")
plt.close()


def delta_z(z_pred, z_spec) :
    return (z_pred - z_spec) / (1 + z_spec)

#### CALCUL DES METRIQUES ASTRO 
bigbins_edges = np.linspace(0, 6, 24)
megabins_edges = np.array([0, 0.6, 2, 4, 6])

deltas_z = (pred_z - true_z) / (1 + true_z) 

### LES PLOTS

bias = np.zeros((len(bigbins_edges)-1))
smad = np.zeros((len(bigbins_edges)-1))
outl = np.zeros((len(bigbins_edges)-1))



for i in range(len(bigbins_edges)-1) :
    inds = np.where((true_z>=bigbins_edges[i]) & (true_z<bigbins_edges[i+1]))

    selected_deltas = deltas_z[inds]

    bias[i] = np.mean(selected_deltas)

    median_delta_z_norm = np.median(selected_deltas)
    mad = np.median(np.abs(selected_deltas - median_delta_z_norm))
    sigma_mad = 1.4826 * mad
    smad[i] = sigma_mad

    outliers = np.abs(selected_deltas) > 0.05
    fraction_outliers = np.sum(outliers) / (len(selected_deltas)+1e-6)
    outl[i] = fraction_outliers
    


bins_centres = (bigbins_edges[1:] + bigbins_edges[:-1])/2

plt.plot(bins_centres, bias)
plt.xlabel("Z")
plt.ylabel("prediction bias")
plt.title("prediction bias for "+model_name)
plt.savefig(base_path+"plots/simCLR/bias_"+model_name+".png")

plt.plot(bins_centres, smad)
plt.xlabel("Z")
plt.ylabel("sMAD")
plt.title("Sigma MAD for "+model_name)
plt.savefig(base_path+"plots/simCLR/smad_"+model_name+".png")

plt.plot(bins_centres, outl)
plt.xlabel("Z")
plt.ylabel("outlier fraction")
plt.title("outlier fraction for "+model_name)
plt.savefig(base_path+"plots/simCLR/outl_"+model_name+".png")



bias = np.zeros((len(megabins_edges)-1))
smad = np.zeros((len(megabins_edges)-1))
outl = np.zeros((len(megabins_edges)-1))
#### LES VALEURS DE TABLEAU :
for i in range(len(megabins_edges)-1) :
    inds = np.where((true_z>=megabins_edges[i]) & (true_z<megabins_edges[i+1]))

    selected_deltas = deltas_z[inds]

    bias[i] = np.mean(selected_deltas)

    median_delta_z_norm = np.median(selected_deltas)
    mad = np.median(np.abs(selected_deltas - median_delta_z_norm))
    sigma_mad = 1.4826 * mad
    smad[i] = sigma_mad

    outliers = np.abs(selected_deltas) > 0.05
    fraction_outliers = np.sum(outliers) / (len(selected_deltas)+1e-6)
    outl[i] = fraction_outliers


print("RESULTS ON MEGABINS EDGES :")
print("PLAGES : [0, 0.6]     [0.6, 2]    [2, 4]     [4, 6]")
print("BIAS :", bias)
print("SMAD :", smad)
print("OUTL :", outl)
#model.backbone.save_weights("sdss_backbone.weights.h5")
#model.classifier.save_weights("sdss_classifier.weights.h5")





