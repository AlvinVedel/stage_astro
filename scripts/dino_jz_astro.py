import numpy as np
import tensorflow as tf
from vit_layers import BackboneAstro, Head
from dino_generator import DinoGenerator, DinoDataAug
import os


os.environ["CUDA_VISIBLE_DEVICES"] ='0, 1'

 

teacher_head = Head(576, 150)
student_head = Head(576, 150)
teacher_backbone = BackboneAstro()
teacher_ibot_head = Head(576, 150)
student_ibot_head = Head(576, 150)
student_backbone = BackboneAstro()

batch_size=32
masking_rate = 0.3
data_generator = DinoDataAug("/lustre/fswork/projects/rech/kof/uve94ap/CUBES_HSC/PHOT/COSMOS", batch_size, (48, 48), (16, 16))

momentum = 0.95

dino_momentum = 0.9
dino_centre = tf.zeros(shape=(150), dtype=tf.float32)

ibot_momentum=0.9
ibot_centre = tf.zeros(shape=(150), dtype=tf.float32)

def compute_cross_entropy(student_output, teacher_output, centre) :
    teacher_output = tf.nn.softmax((teacher_output-centre)/0.8)   
    student_output = tf.nn.log_softmax(student_output/0.7)

    loss = -tf.reduce_sum(teacher_output * student_output, axis=-1)  
    return tf.reduce_mean(loss)

def compute_koleo_loss(student_outputs) :
        # forme batch, 576
        # Normalisation L2 :
        student_outputs = student_outputs / (tf.math.sqrt(tf.reduce_sum((student_outputs)**2, axis=1, keepdims=True)) + 1e-8) # batch, 576 / batch, 1 => batch, 576
        # calcul des distance  : batch, 1, 576 * 1, batch, 576 => batch, batch, 576 => batch, batch  distance entre chaque élément du batch
        dist_mat = tf.reduce_sum((tf.expand_dims(student_outputs, axis=0) - tf.expand_dims(student_outputs, axis=1))**2, axis=2)  # shape batch, batch
        max_val = tf.reduce_max(dist_mat)+1
        eye = tf.eye(dist_mat.shape[0], dtype=tf.float32)*max_val
        dist_mat = dist_mat + eye # on rempli diagonale de valeur plus grande que les autres
        min_pairwise_dist = tf.math.sqrt(tf.reduce_min(dist_mat, axis=1)+1e-8)  # on a un matrice symétrique normalement alors axis pas important, on prend les distances min
        # distances min entre le batch de la colonne i et n'importe quel batch de la ligne j (j différent de i car diagonale très grande)
        loss = - tf.reduce_mean(tf.math.log(min_pairwise_dist+1e-8))
        return loss

def update_teacher_model(student_model, teacher_model, momentum):
        student_weights = student_model.get_weights()
        teacher_weights = teacher_model.get_weights()
        new_teacher_weights = []
        for student_w, teacher_w in zip(student_weights, teacher_weights):
            new_teacher_w = momentum * teacher_w + (1 - momentum) * student_w
            new_teacher_weights.append(new_teacher_w)
        teacher_model.set_weights(new_teacher_weights)
        
        

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, decay=0.01)
total_epoch = 100
initial_weight_decay = 0.04
final_weight_decay = 0.4



for epoch in range(total_epoch) :
        for batch in  data_generator: 

            
                # shape 32, 2, 48, 48, 9
            large_crops = batch["large_crop"]
            n_global = tf.shape(large_crops)[1]
                # shape 32, 4, 16, 16
            small_crops = batch["small_crop"]
            n_local = tf.shape(small_crops)[1]
                # same shape as large
            #masked_large = batch["masked_crop"]
                # liste d'index
            masked_index = batch["masked_patch_index"]

            

            b, n, w, h, c = large_crops.shape[0], large_crops.shape[1], large_crops.shape[2], large_crops.shape[3], large_crops.shape[4]
                

            with tf.GradientTape() as tape :
                    # 1 LARGE CROPS TOKEN  BAKCBONE
                    large_crops = tf.concat([large_crops[:, 0], large_crops[:, 1]], axis=0)  # 32, 48, 48, 9 & 32, 48, 48, 9  ==>  64, 48, 48, 9
                    
                    teacher_output_dict = teacher_backbone(large_crops, training=False)  # DINO LOCAL, DINO GLOBAL 
                    student_output_dict = student_backbone(large_crops, training=True)  # DINO GLOBAL, KOLEO

                    # 2  CLS TOKENS PASSAGE DANS HEAD  :   
                    teacher_cls_token = teacher_output_dict["cls_token"]  # batch*2, 576


                    teacher_cls_tokens_after_head = teacher_head(teacher_cls_token, training=False) 


                    teacher_cls_tokens_after_head = tf.reshape(teacher_cls_tokens_after_head, (b, n, tf.shape(teacher_cls_tokens_after_head)[1])) # batch, 2, 150
                    teacher_cls_tokens_after_head = tf.concat([tf.expand_dims(teacher_cls_tokens_after_head[:, 1], axis=1), tf.expand_dims(teacher_cls_tokens_after_head[:, 0], axis=1)], axis=1)                
                    

                    student_cls_tokens_after_head = student_head(student_output_dict["cls_token"], training=True)


                    student_cls_tokens_after_head = tf.reshape(student_cls_tokens_after_head, (b, n, tf.shape(student_cls_tokens_after_head)[1])) # batch, 2, 150
                    ##### DINO GLOBAL
                    dino_global_loss =0
                    for i in range(n_global) :
                        # concat permet de comparer global A teacher avec global B student 
                        dino_global_loss += compute_cross_entropy(student_cls_tokens_after_head[:, i], teacher_cls_tokens_after_head[:, i], dino_centre)
                    dino_global_loss /= float(n_global)
                    

                    # 3 CLS TOKENS DES VUES REDUITES (STUDENT)
                    small_crops = tf.reshape(small_crops, (tf.shape(small_crops)[0]*tf.shape(small_crops)[1], tf.shape(small_crops)[2], tf.shape(small_crops)[3], tf.shape(small_crops)[4]))# batch*n_local, 16, 16, 9
                
                    student_local_output_dict = student_backbone(small_crops, training=True)  # DINO LOCAL    cls tokens shape batch*n_local, 576
                    student_local_cls_tokens_after_head = student_head(student_local_output_dict["cls_token"], training=True) # batch*n_local, 150

                    student_local_cls_tokens_after_head = tf.reshape(student_local_cls_tokens_after_head, (b, n_local,tf.shape(student_local_cls_tokens_after_head)[-1]))
                    # shape batch, n_local, 150
                    
                    #### DINO LOCAL 
                    dino_local_loss = 0
                    for i in range(n_global) :
                        for j in range(n_local) : 
                            dino_local_loss += compute_cross_entropy(student_local_cls_tokens_after_head[:, j], teacher_cls_tokens_after_head[:, i], dino_centre)
                    dino_local_loss /= float(n_global * n_local) 


                    # 4 
                    ####  KOLEO LOSS :   student_global_cls_token
                    student_cls_token = student_output_dict["cls_token"]  # shape batch*2, 576
                    student_cls_token = tf.reshape(student_cls_token, (b, n, tf.shape(student_cls_token)[-1]))  # shape batch, 2, 576
                    koleo_loss = 0
                    #student_cls_token = tf.cast(student_cls_token, dtype=tf.float32)
                    for i in range(n_global) :
                        koleo_loss += compute_koleo_loss(student_cls_token[:, i])
                        #print(koleo_loss.dtype)
                    koleo_loss /= float(n_global)
                    

                    #print("c'est dans le ibot que je bug")

                    # 5  RECUPERATION PATCH TOKENS MASQUES TEACHER
                    teacher_patch_token = teacher_output_dict["patch_token"]  # shape batch*2, num_patch, 576    
                    teacher_patch_token = tf.reshape(teacher_patch_token, (b, n,tf.shape(teacher_patch_token)[1], tf.shape(teacher_patch_token)[2]))
                    masked_teacher_patch_tokens = tf.boolean_mask(teacher_patch_token, masked_index)

                    #masked_teacher_patch_tokens = tf.gather(teacher_patch_token, masked_index, batch_dims=2, axis=2) # masked index de shape 32, 2, 57, 576
                    #masked_large = tf.reshape(masked_large, (b*n, w, h, c))  # batch*2, 48, 48, 9 avec parties masquées (0)
                    masked_index_flat = tf.reshape(masked_index, (-1, tf.shape(masked_index)[2]))
                    
                    masked_student_output_dict = student_backbone(large_crops, masks=masked_index_flat, training=True)
                    masked_student_patch_token = masked_student_output_dict["patch_token"] # shape batch*2, num_patch, 576
                    masked_student_patch_token = tf.reshape(masked_student_patch_token, (b, n, tf.shape(masked_student_patch_token)[1], tf.shape(masked_student_patch_token)[2])) # batch, 2, num_patch, 576
                    masked_student_patch_token = tf.boolean_mask(masked_student_patch_token, masked_index)
                    #masked_student_patch_token = tf.gather(masked_student_patch_token, masked_index, batch_dims=2, axis=2)  # batch, 2, 57, 576
                    # input de shape batch * 2 * num_patch, 576
                    
                    teacher_patch_tokens_after_head = teacher_ibot_head(tf.reshape(masked_teacher_patch_tokens, (-1, tf.shape(masked_teacher_patch_tokens)[-1])), training=False)
                    student_patch_tokens_after_head = student_ibot_head(tf.reshape(masked_student_patch_token, (-1, tf.shape(masked_student_patch_token)[-1])), training=True)
                    # les 2 sont de shape batch * 2 * num_patch, 150
                
                    ##### IBOT LOSS : cross entropy entre les patch tokens (version masquée vs pas masquée)
                    ibot_loss = compute_cross_entropy(student_patch_tokens_after_head, teacher_patch_tokens_after_head, ibot_centre)
                    total_loss = dino_global_loss + dino_local_loss + ibot_loss + 0.1*koleo_loss
                
                    
        
        
            gradients = tape.gradient(total_loss, student_backbone.trainable_variables + student_head.trainable_variables)
            optimizer.apply_gradients(zip(gradients, student_backbone.trainable_variables + student_head.trainable_variables))



            current_weight_decay = initial_weight_decay + (final_weight_decay - initial_weight_decay) * (epoch / total_epoch)
            optimizer.weight_decay = current_weight_decay
            lr = np.clip(0.001 * np.cos(np.pi * epoch / total_epoch), 1e-6, 1)  # Exemple de decay cosin
            optimizer.learning_rate = lr

            ibot_centre = tf.reduce_mean(teacher_patch_tokens_after_head, axis=0)*(1-ibot_momentum) + ibot_momentum * ibot_centre  # shape 150
            dino_centre = dino_momentum * dino_centre + (1 - dino_momentum)*tf.reduce_mean(tf.reshape(teacher_cls_tokens_after_head, (b*n, -1)), axis=0)   # shape 150

            
            update_teacher_model(student_backbone, teacher_backbone, momentum)
            update_teacher_model(student_head, teacher_head, momentum)

        if epoch % 1 == 0 :
            print("weights saved")
            teacher_backbone.save_weights("./checkpoints_dino_astro/teacher_backbone.weights.h5")
            student_backbone.save_weights("./checkpoints_dino_astro/student_backbone.weights.h5")
            teacher_head.save_weights("checkpoints_dino_astro/teacher_head.weights.h5")
            student_head.save_weights("checkpoints_dino_astro/student_head.weights.h5")
            teacher_ibot_head.save_weights("checkpoints_dino_astro/teacher_ibot_head.weights.h5")
            student_ibot_head.save_weights("checkpoints_dino_astro/student_ibot_head.weights.h5")



