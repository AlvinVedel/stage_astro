
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
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import time

bank_size=10000
model_save = 'checkpoints_new_simCLR/simCLR_UD_D_norm'
iter_suffixe="_resnet50_bank"+str(bank_size)
allowed_extensions = ["UD.npz", "_D.npz"]
batch_size=256
lr = 1e-4
optimizer = tf.keras.optimizers.Adam(lr)
callbacks = [LinearDecay(0, 2, 40)]
loss_tracker = tf.keras.metrics.Mean(name="loss")

#### PARAMS  générateur
do_color = True
do_seg = False
do_drop_band = False
do_adversarial = False

load_model = False
iter = 0

momentum = 0.999


def update_network_with_momentum(network_1, network_2, momentum=0.999):
    
    for layer_1, layer_2 in zip(network_1.layers, network_2.layers):
        if hasattr(layer_1, 'kernel') and hasattr(layer_2, 'kernel'):
            new_kernel = momentum * layer_2.kernel + (1 - momentum) * layer_1.kernel
            layer_2.kernel.assign(new_kernel)
        
        if hasattr(layer_1, 'bias') and hasattr(layer_2, 'bias'):
            new_bias = momentum * layer_2.bias + (1 - momentum) * layer_1.bias
            layer_2.bias.assign(new_bias)
    
    return network_2

#backbone = basic_backbone(full_bn=True, all_bn=False)
backbone = ResNet50(include_top=False, weights=None, input_shape=(64, 64, 6), pooling='avg')
head = noregu_projection_mlp(2048, bn=True, out_dim=128)
color_head = color_mlp(2048)

late_backbone = ResNet50(include_top=False, weights=None, input_shape=(64, 64, 6), pooling='avg')
late_head = noregu_projection_mlp(2048, bn=True, out_dim=128)

if load_model :
    backbone.load_weights("../model_save/"+model_save+str(iter*10)+iter_suffixe+".weights.h5")
    head.load_weights("../model_save/"+model_save+str(iter*10)+iter_suffixe+".weights.h5")
    color_head.load_weights("../model_save/"+model_save+str(iter*10)+iter_suffixe+".weights.h5")


late_backbone.set_weights(backbone.get_weights())
late_head.set_weights(head.get_weights())


data_gen = MultiGen(["/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/cleaned_spec/", "/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/cleaned_phot/"], 
               batch_size=batch_size, extensions=allowed_extensions, do_color=do_color, do_seg=do_seg, do_mask_band=do_drop_band, same_samples=True, n_samples=30000)

print("data_gen ready")

#checkpoints_new_simCLR
bank = np.zeros((bank_size, 128))
bank_index = 0
bank_counter = 0
loss_function = ContrastivLoss(0.1)


def compute_loss_cpu(batch_representation, bank_representation, temperature=0.1) :
    ## BANK DE SHAPE BANK_SIZE, PROJ_DIM

    large_num = 1e8
    batch_representation = batch_representation / np.sqrt(np.sum(batch_representation**2, axis=1, keepdims=True))
    bank_representation = bank_representation / np.sqrt(np.sum(bank_representation**2, axis=1, keepdims=True))

    
    hidden1, hidden2 = np.split(batch_representation, 2, axis=0)  # shape BATCH, PROJ_DIM
    batch_size = np.shape(hidden1)[0]

    hidden1_large = hidden1
    hidden2_large = hidden2

    labels = np.concatenate([np.eye(batch_size)[np.arange(batch_size)], np.ones((batch_size, batch_size+bank_representation.shape[0]))], axis=1)  ## batch size, 2*batch_size+banksize
    #labels = np.one_hot(np.arange(batch_size), batch_size*2 + bank_representation.shape[0])    # matrice des des labels,    batch_size x 2*batch_size
    #masks = np.one_hot(np.arange(batch_size), batch_size)       # mask de shape     batch x batch
    masks = np.eye(batch_size)[np.arange(batch_size)]

    logits_aa = np.matmul(hidden1, hidden1_large.T) / temperature       ### si normalisé cela aurait été cosine sim => shape batch, batch    distance x x
    logits_aa = logits_aa - masks * large_num    ### on rempli la diagonale de très petite valeur car forcément cosine sim entre vecteurs identique = 1

    logits_abank = np.matmul(hidden2, bank_representation.T) / temperature

    logits_ab = np.matmul(hidden1, hidden2_large.T) / temperature     ### sim x x'
    
    logits = np.concatenate([logits_ab, logits_aa, logits_abank], axis=1)   # batch, batch*2+bank_size
    probs = logits / np.expand_dims(np.sum(logits, axis=1), axis=1)  # même shape en 1 vecteur de probas pour chaque élément du batch

    loss_a = - np.sum(labels * np.log(probs))

    #loss_a = tf.nn.softmax_cross_entropy_with_logits(              ### matrice labels contient info de où sont les paires positives
    #    labels, tf.concat([logits_ab, logits_aa, logits_abank], 1))              ### en concaténant ab et aa on obtient similarité de a vers toutes les autres images (en ayant mis sa propre correspondance à 0) 
    
    
    ## 2EME PARTIE
    del logits_aa, logits_ab, logits_abank


    logits_aa = np.matmul(hidden2, hidden2_large.T) / temperature
    logits_aa = logits_aa - masks * large_num    ###  idem ici ==> donc là on fait distances entre x' x'

    logits_ab = np.matmul(hidden2, hidden1_large.T) / temperature     ### sim x' x 

    logits_abank = np.matmul(hidden2, bank_representation.T) / temperature


    logits = np.concatenate([logits_ab, logits_aa, logits_abank], axis=1)   # batch, batch*2+bank_size
    probs = logits / np.expand_dims(np.sum(logits, axis=1), axis=1)  # même shape en 1 vecteur de probas pour chaque élément du batch

    loss_b = - np.sum(labels * np.log(probs+1e-10))

    #loss_b = tf.nn.softmax_cross_entropy_with_logits(              ### idem de b vers toutes les images
    #    labels, tf.concat([logits_ab, logits_aa, logits_abank], 1))


    loss = (loss_a + loss_b)  / (batch_size*2)    ### moyenne des 2 et loss
    return loss

def compute_loss_tf(batch_representation, bank_representation, temperature=0.1):
    large_num = 1e8
    batch_size = tf.shape(batch_representation)[0] // 2

    # Normalisation L2 (cosine similarity)
    batch_representation = tf.math.l2_normalize(batch_representation, axis=1)
    bank_representation = tf.math.l2_normalize(tf.cast(bank_representation, dtype=tf.float32), axis=1)

    hidden1, hidden2 = tf.split(batch_representation, 2, axis=0)

    # Masque pour éviter self-similarity
    masks = tf.eye(batch_size)

    # Calcul des similarités
    logits_aa = tf.matmul(hidden1, hidden1, transpose_b=True) / temperature
    logits_aa -= masks * large_num  # Évite la diagonale

    logits_abank = tf.matmul(hidden1, bank_representation, transpose_b=True) / temperature
    logits_ab = tf.matmul(hidden1, hidden2, transpose_b=True) / temperature

    logits = tf.concat([logits_ab, logits_aa, logits_abank], axis=1)
    labels = tf.concat([tf.eye(batch_size), tf.zeros((batch_size, batch_size + tf.shape(bank_representation)[0]))], axis=1)

    loss_a = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
    
    ### PARTIE B ###
    logits_aa = tf.matmul(hidden2, hidden2, transpose_b=True) / temperature
    logits_aa -= masks * large_num

    logits_ab = tf.matmul(hidden2, hidden1, transpose_b=True) / temperature
    logits_abank = tf.matmul(hidden2, bank_representation, transpose_b=True) / temperature

    logits = tf.concat([logits_ab, logits_aa, logits_abank], axis=1)
    loss_b = tf.nn.softmax_cross_entropy_with_logits(labels, logits)

    # Moyenne des deux pertes
    loss = tf.reduce_mean(loss_a + loss_b)
    return tf.minimum(loss, 100.0)




iter_ = 0
while True :   ## on attend fin du job
    iter_+=1
    for epoch in range(10) :
        print("ITER ", iter_, "epoch :", epoch)
        #bank = data_gen.images[:bank_size]   # pas grave si pas d'augmentations sur elles ?
        #train_images = data_gen.images[bank_size:]
        #train_colors = data_gen.colors[bank_size:]

        #nstep = train_images.shape[0] // batch_size

        for b, batch in enumerate(data_gen) :
            batch, labels_dict = batch
            color_labels = labels_dict["color"]
            #bank_representation = late_head(late_backbone(batch, training=False), training=False).numpy()
            ## stockage dans mémoire
            #for i in range(bank_representation.shape[0]) :
                #bank[bank_index] = bank_representation[i]
                #bank_index = (bank_index+1)%bank.shape[0]
                #bank_counter = min(bank.shape[0], bank_counter+1)

            with tf.GradientTape() as tape :
                x = backbone(batch, training=True)
                current_representation = head(x, training=True)
                #current_representation = head(backbone(batch, training=True), training=True)
                if bank_counter > 0 :
                    selected_el = bank[:bank_counter]
                    #print(selected_el.shape)
                    #loss = loss_function.call(
                    loss = compute_loss_tf(current_representation,selected_el, 0.1)
                else :
                    loss = loss_function.call(current_representation, 0.1)
                color_loss = tf.reduce_mean(tf.reduce_sum( (color_head(x, training=True) - color_labels)**2, axis=-1))
                total_loss = loss + color_loss
            gradients = tape.gradient(total_loss, head.trainable_variables+backbone.trainable_variables+color_head.trainable_variables)
            optimizer.apply_gradients(zip(gradients, head.trainable_variables+backbone.trainable_variables+color_head.trainable_variables))

            #if b % 100 == 0 :
            #    print("LOSS : ", loss, color_loss)

            loss_tracker.update_state(loss)  # Met à jour la moyenne de la loss

            if b % 50 == 0:  # Affichage toutes les 100 itérations
                print(f"Epoch {epoch}, Step {b}, Loss: {loss_tracker.result().numpy():.4f}")

            bank_representation = late_head(late_backbone(batch, training=False), training=False).numpy()
            ## stockage dans mémoire
            #print("representation : ",bank_representation.shape)
            for i in range(bank_representation.shape[0]) :
                bank[bank_index] = bank_representation[i]
                bank_index = (bank_index+1)%bank.shape[0]
                bank_counter = min(bank.shape[0], bank_counter+1)
                
            late_head = update_network_with_momentum(head, late_head, momentum=momentum)
            late_backbone = update_network_with_momentum(backbone, late_backbone, momentum=momentum)  


    data_gen._load_data()
    loss_tracker.reset_states()

    backbone.save_weights("/lustre/fswork/projects/rech/dnz/ull82ct/astro/model_save/backbone_simMoco.weights.h5")
    head.save_weights("/lustre/fswork/projects/rech/dnz/ull82ct/astro/model_save/head_simMoco.weights.h5")

            
                
            


                

















