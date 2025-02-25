import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import layers
import random
from utils.schedulers import TreyerScheduler
from generators.generator import DataGenerator
import matplotlib.pyplot as plt
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""
SCRIPT POUR L ENTRAINEMENT SUPERVISE DE LA PREDICTION DU REDSHIFT
EN IMITANT LES TRAVAUX DE TREYER ET AL : CNN photometric redshifts in the SDSS at  r≤20  https://arxiv.org/abs/2310.02173


"""




data = np.load("/home/barrage/HSC_READY_CUBES/XMM_SHALLOW_SPECTRO.npz", allow_pickle=True)

images = np.sign(data["cube"]) * np.sqrt(np.abs(data["cube"])+1)-1
#images = data["cube"]


meta = data["info"].item()
nb_bins = 150
zspec = meta["ZSPEC"]


index = np.where(zspec<0.4)

images = images[index]
zspec = zspec[index]

zmax = np.max(zspec)
bin_width = zmax/nb_bins
hist, bin_edges = np.histogram(zspec, bins=nb_bins)
linear_bin_edges = np.linspace(0, 0.4, 151)


discretized_data = np.digitize(zspec, linear_bin_edges) - 1

plt.hist(zspec, bins=150)
plt.savefig("distribution_hist.png")


ebv = meta["EBV"]
ebv = ebv[index]

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

    pdf = layers.Dense(150, activation='softmax', name='pdf')(l2)
    regression = layers.Dense(1, activation='relu', name='reg')(l3)
    if with_ebv :
        model = keras.Model(inputs=[input_img, ebv_input], outputs=[pdf, regression])
    else :
        model = keras.Model(inputs=[input_img], outputs=[pdf, regression])
    return model




data_base = np.concatenate([np.expand_dims(np.arange(0, zspec.shape[0], 1), axis=1), np.expand_dims(ebv, axis=1), np.expand_dims(zspec, axis=1), np.expand_dims(discretized_data, axis=1)], axis=1)

length = data_base.shape[0]
random.shuffle(data_base)
max_ind = int(length*0.7)
train = data_base[:max_ind]


test = data_base[max_ind:]

count_train = []
count_test = []
for i in range(np.max(train[:, 3]).astype(np.int32)) :
    train_i = train[np.where(train[:, 3]==i)]
    test_i = test[np.where(test[:, 3]==i)]
    count_train.append(len(train_i))
    count_test.append(len(test_i))


# MODEL
model = create_model(with_ebv=True)  
optimizer = tf.keras.optimizers.Adam(lr=1e-3)
model.compile(optimizer=optimizer, loss={'pdf':'sparse_categorical_crossentropy', 'reg':'mse'}, metrics={'pdf':'accuracy', 'reg':['mse','mae']})
train_indices = train[:, 0].astype(np.int32)
train_image = images[train_indices]


data_gen = DataGenerator(image_data=train_image, ebv=train[:, 1], label=train[:, 3], zspec=train[:, 2],batch_size=32)

history = model.fit(data_gen, epochs=50, callbacks=[TreyerScheduler()])


model.save("./cnn_with_ebv_data_aug.h5")
np.save("test_data_ebv.npy", test)


























