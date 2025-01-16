import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from vit_layers import ViT_backbone



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


def basic_backbone() :
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
   
    return keras.Model(inputs=inp, outputs=l1)


def treyer_backbone(bn=False) :

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
    
    #conc = layers.Concatenate()([resh, input_ebv])
    l1 = layers.Dense(1024, activation='relu', name='l1')(resh)
    if bn :
        l1 = layers.BatchNormalization()(l1)

    
    model = keras.Model(inputs=[input_img], outputs=[l1])
    return model


def projection_mlp(input_shape=1024, bn=False):
    latent_input = keras.Input((input_shape))
    x = layers.Dense(512, activation='linear')(latent_input)
    if bn :
        x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)
    x = layers.Dense(256, activation='linear', activity_regularizer=tf.keras.regularizer.L1L2(l1=1e-3, l2=1e-2))(x)
    return keras.Model(latent_input, x)


def color_mlp(input_shape=1024) :
    latent_input = keras.Input((input_shape))
    x = layers.Dense(256)(latent_input)
    x = layers.PReLU()(x)
    x = layers.Dense(256)(x)
    x = layers.PReLU()(x)
    output = layers.Dense(4, activation='linear')(x)
    return keras.Model(latent_input, output)

def classif_mlp(input_shape=1024) :
    latent_input = keras.Input((input_shape))
    x = layers.Dense(256, activation='relu')(latent_input)
    x = layers.Dense(256, activation='relu')(x)
    output = layers.Dense(2, activation='softmax')(x)
    return keras.Model(latent_input, output)



def astro_head(input_shape=1024, nbins=400) :
    inp = keras.Input((input_shape))
    l1 = layers.Dense(1024)(inp)
    l1 = layers.PReLU()(l1)

    l2 = layers.Dense(1024)(l1)
    l2 = layers.PReLU()(l2)
    pdf = layers.Dense(nbins, activation='softmax', name='pdf')(l2)

    l2b = layers.Dense(512, activation='tanh')(l1)
    reg = layers.Dense(1, activation='linear', name='reg')(l2b)
    return keras.Model(inp, [pdf, reg])


def astro_model(back, head) :
    inp = keras.Input((64, 64, 5))
    x = back(inp)
    output = head(x)
    return keras.Model(inp, output)



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