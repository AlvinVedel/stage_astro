import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import layers
from byol_generator import ByolGenerator
from byol_model import BYOL, ByolLoss

#from tensorflow.keras.applications import ResNet50
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def mlp(input_shape=2048):
    latent_input = keras.Input((input_shape))
    x = layers.Dense(4096, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(5e-7))(latent_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(256, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(5e-7))(x)
    return keras.Model(latent_input, x)

def ResNet50():
    inp = tf.keras.Input((64, 64, 9))
    c1 = layers.Conv2D(64, kernel_size=7, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-7))(inp)  # 32, 32
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.ReLU()(c1)
    c1 = layers.MaxPooling2D((2, 2))(c1)  # 16, 16

    r1 = bottleneck_block(c1, 64, downsample=True)
    r1 = bottleneck_block(r1, 64)
    r1 = bottleneck_block(r1, 64)

    r2 = bottleneck_block(r1, 128, 2, True)   #8, 8
    r2 = bottleneck_block(r2, 128)
    r2 = bottleneck_block(r2, 128)
    r2 = bottleneck_block(r2, 128)

    r3 = bottleneck_block(r2, 256, 2, True) # 4, 4
    r3 = bottleneck_block(r3, 256)
    r3 = bottleneck_block(r3, 256)
    r3 = bottleneck_block(r3, 256)
    r3 = bottleneck_block(r3, 256)
    r3 = bottleneck_block(r3, 256)

    r4 = bottleneck_block(r3, 512, 2, True)  #2, 2
    r4 = bottleneck_block(r4, 512)
    r4 = bottleneck_block(r4, 512)

    x = layers.GlobalAveragePooling2D()(r4)  # 512
    return tf.keras.Model(inp, x)

def bottleneck_block(x, filters, stride=1, downsample=False) :
    identity = x
    if downsample :
        identity = layers.Conv2D(filters*4, (1, 1), strides=stride, padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(5e-7))(identity)
        identity = layers.BatchNormalization()(identity)

    x = conv_block(x, filters, kernel_size=1, stride=stride)
    x = conv_block(x, filters)

    x = layers.Conv2D(filters*4, (1, 1), use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(5e-7))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, identity])
    x = layers.ReLU()(x)
    return x

def conv_block(x, filters, kernel_size=3, stride=1, padding='same'):
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding=padding, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(5e-7))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


#data_gen = ByolGenerator("/home/barrage/HSC_READY_CUBES/XMM_SHALLOW_UNSUP_200K_OBJECTS.npz", batch_size=256)

data_gen = ByolGenerator("/lustre/fswork/projects/rech/kof/uve94ap/CUBES_HSC/PHOT/COSMOS", batch_size=256)



optimizer = tf.keras.optimizers.Adam(lr=0.3)

model = BYOL(ResNet50(), ResNet50(), mlp(2048), mlp(2048), mlp(256))
model(np.random.random((32, 64, 64, 9)))
#model.load_weights("byol.weights.h5")

model.compile(optimizer=optimizer, loss=ByolLoss())


def cosine_annealing(epoch):
    # Calcul du taux d'apprentissage selon une fonction cosinus
    return 0 + 0.5 * (0.3/2 - 0) * (1 + np.cos(np.pi * epoch / 600))
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(cosine_annealing)

checkpoint_dir = 'checkpoints/'
os.makedirs(checkpoint_dir, exist_ok=True)

# Callback pour sauvegarder les poids toutes les 50 époques
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'byol_epoch_{epoch:02d}.h5'),
    save_weights_only=True,  # Sauvegarde uniquement les poids
    save_freq=50 * len(data_gen),  # Sauvegarde toutes les 50 époques
    save_best_only=False  # Sauvegarde toutes les 50 époques, pas seulement le meilleur modèle
)

model.fit(data_gen, epochs=600, callbacks=[lr_scheduler])

model.save_weights("byol.weights.h5")

















"""
online = keras.Sequential()
#online.add(ResNet50(input_shape=(64, 64, 9), include_top=False, weights=None, pooling='avg'))
online.add(ResNet50())
online.add(mlp(2048))
online.add(mlp(256))

target = keras.Sequential()
#target.add(ResNet50(input_shape=(64, 64, 9), include_top=False, weights=None, pooling='avg'))
target.add(ResNet50())
target.add(mlp(2048))


def compute_loss(x, y) :
    x = tf.math.l2_normalize(x, axis=-1)
    y = tf.math.l2_normalize(y, axis=-1)
    loss = 2 - 2 * tf.reduce_sum(x*y, axis=-1)
    return tf.reduce_mean(loss)

def update_target_weights(target_weights, online_weights, tau=0.99):
    for target_w, online_w in zip(target_weights, online_weights):
        target_w.assign(tau * target_w + (1 - tau) * online_w)

optimizer = tf.keras.optimizers.Adam(lr=1e-3)

for epoch in range(300) :
    total_loss = 0
    for batch in data_gen :
        target_res0 = target(batch[:, 0], training=False)
        target_res1 = target(batch[:, 1], training=False)
        with tf.GradientTape() as tape :
            online_res0 = online(batch[:, 0], training=True)
            online_res1 = online(batch[:, 1], training=True)

            loss = compute_loss(target_res0, online_res1) + compute_loss(target_res1, online_res0)
        gradients = tape.gradient(loss, online.trainable_variables)        
        optimizer.apply_gradients(zip(gradients, online.trainable_variables))
        total_loss += loss.numpy()
        print("batch loss :", loss)
        update_target_weights(target.variables, online.variables)

    print(loss/len(data_gen))
        
"""         




