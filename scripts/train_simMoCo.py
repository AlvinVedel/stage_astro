import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from contrastiv_model import simCLR, NTXent as ContrastivLoss, simCLRcolor1
from generator import MultiGen
from regularizers import VarRegularizer, TripletCosineRegularizer, CosineDistRegularizer
from deep_models import basic_backbone, projection_mlp, color_mlp, treyer_backbone, segmentor, deconvolutor, classif_mlp, noregu_projection_mlp
from vit_layers import Block, ViT_backbone
from keras.applications import ResNet50
from schedulers import CosineDecay, LinearDecay

import os 
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
import time


model_save = 'checkpoints_new_simCLR/simCLR_UD_D_norm'
iter_suffixe="_ColorHead_NotRegularized_fullBN"
allowed_extensions = ["UD.npz", "_D.npz"]
batch_size=256
lr = 1e-4
optimizer = keras.optimizers.Adam(lr)
callbacks = [LinearDecay(0, 2, 40)]

#### PARAMS  générateur
do_color = True
do_seg = False
do_drop_band = False
do_adversarial = False

load_model = False
iter = 0

momentum = 0.999
bank_size = 5000
loss = ContrastivLoss(normalize=True)

def update_network_with_momentum(network_1, network_2, momentum=0.999):
    
    for layer_1, layer_2 in zip(network_1.layers, network_2.layers):
        if hasattr(layer_1, 'kernel') and hasattr(layer_2, 'kernel'):
            new_kernel = momentum * layer_2.kernel + (1 - momentum) * layer_1.kernel
            layer_2.kernel.assign(new_kernel)
        
        if hasattr(layer_1, 'bias') and hasattr(layer_2, 'bias'):
            new_bias = momentum * layer_2.bias + (1 - momentum) * layer_1.bias
            layer_2.bias.assign(new_bias)
    
    return network_2


backbone = ResNet50(include_top=False, weights=None, input_shape=(64, 64, 6), pooling='avg')
head = noregu_projection_mlp(1024, bn=True)
color_head = color_mlp(1024)

late_backbone = ResNet50(include_top=False, weights=None, input_shape=(64, 64, 6), pooling='avg')
late_head = noregu_projection_mlp(1024, bn=True)

if load_model :
    backbone.load_weights("../model_save/"+model_save+str(iter*10)+iter_suffixe+".weights.h5")
    head.load_weights("../model_save/"+model_save+str(iter*10)+iter_suffixe+".weights.h5")
    color_head.load_weights("../model_save/"+model_save+str(iter*10)+iter_suffixe+".weights.h5")


late_backbone.set_weights(backbone.get_weights())
late_head.set_weights(head.get_weights())


data_gen = MultiGen(["/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/cleaned_spec/", "/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/cleaned_phot/"], 
               batch_size=batch_size, extensions=allowed_extensions, do_color=do_color, do_seg=do_seg, do_mask_band=do_drop_band)



def compute_loss(batch_representation, bank_representation, temperature=0.1) :
    ## BANK DE SHAPE BANK_SIZE, PROJ_DIM

    large_num = 1e8
    batch_representation = tf.math.l2_normalize(batch_representation, axis=1)
    bank_representation = tf.math.l2_normalize(bank_representation, axis=1)

    hidden1, hidden2 = tf.split(batch_representation, 2, 0)   # shape BATCH, PROJ_DIM
    batch_size = tf.shape(hidden1)[0]

    hidden1_large = hidden1
    hidden2_large = hidden2


    labels = tf.one_hot(tf.range(batch_size), batch_size*2 + bank_representation.shape[0])    # matrice des des labels,    batch_size x 2*batch_size
    masks = tf.one_hot(tf.range(batch_size), batch_size)       # mask de shape     batch x batch

    logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / temperature       ### si normalisé cela aurait été cosine sim => shape batch, batch    distance x x
    logits_aa = logits_aa - masks * large_num    ### on rempli la diagonale de très petite valeur car forcément cosine sim entre vecteurs identique = 1

    

    logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature     ### sim x x'
    

    loss_a = tf.nn.softmax_cross_entropy_with_logits(              ### matrice labels contient info de où sont les paires positives
        labels, tf.concat([logits_ab, logits_aa, logits_abank], 1))              ### en concaténant ab et aa on obtient similarité de a vers toutes les autres images (en ayant mis sa propre correspondance à 0) 
    
    
    ## 2EME PARTIE
    del logits_aa, logits_ab, logits_abank


    logits_aa = tf.matmul(hidden2, hidden2_large, transpose_b=True) / temperature
    logits_aa = logits_aa - masks * large_num    ###  idem ici ==> donc là on fait distances entre x' x'

    logits_ab = tf.matmul(hidden2, hidden1_large, transpose_b=True) / temperature     ### sim x' x 

    logits_abank = tf.matmul(hidden2, bank_representation, transpose_b=True) / temperature

    loss_b = tf.nn.softmax_cross_entropy_with_logits(              ### idem de b vers toutes les images
        labels, tf.concat([logits_ab, logits_aa, logits_abank], 1))


    loss = tf.reduce_mean(loss_a + loss_b)     ### moyenne des 2 et loss




while True :   ## on attend fin du job

    for epoch in range(10) :

        bank = data_gen.images[:bank_size]   # pas grave si pas d'augmentations sur elles ?
        train_images = data_gen.images[bank_size:]
        train_colors = data_gen.colors[bank_size:]

        nstep = train_images.shape[0] // batch_size

        for step in range(nstep) :
            batch_images = train_images[step*batch_size:(step+1)*batch_size]
            batch_colors = train_colors[step*batch_size:(step+1)*batch_size]

            bank_augmented = data_gen.process_batch(bank)

            bank_representation = late_backbone.predict(bank_augmented)
            bank_representation = late_head.predict(bank_representation)

            batch_images = data_gen.process_batch(batch_images)
            
            with tf.GradientTape(persistent=True) as tape :
                x = backbone(batch_images, training=True)
                c = color_head(x, training=True)
                z = head(x, training=True)



                





















while iter <= 1000 :
    iter+=1
    model.fit(data_gen, epochs=10, callbacks=callbacks)  # normalement 4mn max par epoch = 400mn 
    data_gen._load_data()

    if iter % 5 == 0 :
        filename = "../model_save/"+model_save+str(iter*10)+iter_suffixe+".weights.h5"
        model.save_weights(filename)  # 6000 minutes   ==> 15 fois 100 épochs
