
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from contrastiv_model import simCLR, simCLRcolor1, simCLRcolor1_adversarial, simCLR1, NTXent, CoreTuning
from deep_models import basic_backbone, projection_mlp, color_mlp, segmentor, deconvolutor, astro_head, astro_model, classif_mlp, AstroFinetune, noregu_projection_mlp
import os 
from vit_layers import ViT_backbone
from generator import SupervisedGenerator, COINGenerator, MultiGen
from schedulers import TreyerScheduler, AlternateTreyerScheduler
from astro_metrics import Bias, SigmaMAD, OutlierFraction
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import time
from tensorflow.keras.applications import ResNet50

base_path = "/lustre/fswork/projects/rech/dnz/ull82ct/astro/"




model = simCLRcolor1(basic_backbone(full_bn=False), projection_mlp(1024, False), color_mlp(1024))
#model = simCLRcolor1(ResNet50(include_top=False, weights=None, input_shape=(64, 64, 6), pooling='avg'), noregu_projection_mlp(2048, True), color_mlp(2048))
#model = simCLRcolor1(ViT_backbone(embed_dim=1024, num_blocks=4, num_heads=8, patch_size=8, gp='none', mlp_ratio=2.0), noregu_projection_mlp(1024), color_mlp(1024))
#model(np.random.random((32, 64, 64, 6)))
in_dim = 1024
train_temp=0.1

#load_w_path = "model_save/checkpoints_new_simCLR/simCLR_UD_D_norm150ViT_petit_model_v2.weights.h5"
load_w_path = "model_save/checkpoints_new_simCLR/v2__300_ColorHead_RegularizedTrue_noBN.weights.h5"
save_w_path = "model_save/simCLR_finetune_comparaison/basic_color_t01_reg"

for condition in ["finecon"] : #, "finecon","coretuning"] :
    for base in ["base1", "base2", "base3"] :

        #data_gen = SupervisedGenerator("/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/finetune/"+base+".npz", batch_size=32, nbins=400)
        data_gen = COINGenerator("/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/finetune/"+base+".npz", batch_size=256, nbins=400, apply_log=True)
        n_epochs = 100

        if condition == "sup" :

            data_gen.contrast=False
            back = basic_backbone(full_bn=False, all_bn=False)
            #back = ViT_backbone(embed_dim=1024, num_blocks=4, num_heads=8, patch_size=8, gp='none', mlp_ratio=2.0)
            #back = ResNet50(include_top=False, weights=None, input_shape=(64, 64, 6), pooling='avg')
            classifier = astro_head(in_dim, 400)
            optim = keras.optimizers.Adam(1e-4)

            for ep in range(n_epochs) :
                if ep == 70 or ep == 90 :
                    optim.learning_rate.assign(optim.learning_rate * 0.1)
                for batch in data_gen :
                    images, labels_dict = batch
                    with tf.GradientTape() as tape :
                        x = back(images, training=True)
                        pdf, reg = classifier(x, training=True)

                        pdf_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels_dict["pdf"], pdf))
                        reg_loss = tf.reduce_mean(tf.math.sqrt(tf.reduce_sum( (labels_dict["reg"] - reg)**2, axis=1) ), axis=0)

                        total_loss = pdf_loss+reg_loss
                    gradients = tape.gradient(total_loss, back.trainable_variables+classifier.trainable_variables)
                    optim.apply_gradients(zip(gradients, back.trainable_variables+classifier.trainable_variables))

            model_to_save = AstroFinetune(back, classifier)
            model_to_save.save_weights(base_path+save_w_path+condition+"_"+base+".weights.h5")

            print("supervisé sur", base, "terminé")




        if condition == "fine" :

            data_gen.contrast=False
            model(np.random.random((32, 64, 64, 6)))
            model.load_weights(base_path+load_w_path)
            back = model.backbone
            classifier = astro_head(in_dim, 400)
            optim = keras.optimizers.Adam(1e-4)

            for ep in range(n_epochs) :
                if ep == 70 or ep == 90 :
                    optim.learning_rate.assign(optim.learning_rate * 0.1)
                for batch in data_gen :
                    images, labels_dict = batch
                    with tf.GradientTape() as tape :
                        x = back(images, training=True)
                        pdf, reg = classifier(x, training=True)

                        pdf_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels_dict["pdf"], pdf))
                        reg_loss = tf.reduce_mean(tf.math.sqrt(tf.reduce_sum( (labels_dict["reg"] - reg)**2, axis=1) ), axis=0)

                        total_loss = pdf_loss+reg_loss
                    gradients = tape.gradient(total_loss, back.trainable_variables+classifier.trainable_variables)
                    optim.apply_gradients(zip(gradients, back.trainable_variables+classifier.trainable_variables))
                print("finetune epoch", ep, total_loss, pdf_loss, reg_loss)

            model_to_save = AstroFinetune(back, classifier)
            model_to_save.save_weights(base_path+save_w_path+condition+"_"+base+".weights.h5")

            print("finetune sur", base, "terminé")
                




        if condition == "finecon" :
            
            data_gen.contrast=True
            contrastiv_loss = NTXent(normalize=True)
            model(np.random.random((32, 64, 64, 6)))
            model.load_weights(base_path+load_w_path)
            back = model.backbone
            classifier = astro_head(in_dim, 400)
            proj = model.head
            optim = keras.optimizers.Adam(1e-4)

            for ep in range(n_epochs) :
                if ep == 70 or ep == 90 :
                    optim.learning_rate.assign(optim.learning_rate * 0.1)
                for batch in data_gen :
                    images, labels_dict = batch
                    with tf.GradientTape() as tape :
                        x = back(images, training=True)
                        pdf, reg = classifier(x, training=True)
                        z = proj(x, training=True)

                        pdf_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels_dict["pdf"], pdf))
                        reg_loss = tf.reduce_mean(tf.math.sqrt(tf.reduce_sum( (labels_dict["reg"] - reg)**2, axis=1) ), axis=0)
                        con_loss = contrastiv_loss.call(z, train_temp) * 0.1

                        total_loss = pdf_loss+reg_loss+con_loss
                    gradients = tape.gradient(total_loss, back.trainable_variables+classifier.trainable_variables+proj.trainable_variables)
                    optim.apply_gradients(zip(gradients, back.trainable_variables+classifier.trainable_variables+proj.trainable_variables))
                print("finetunecon epoch", ep, total_loss, pdf_loss, reg_loss, con_loss)

            model_to_save = AstroFinetune(back, classifier)
            model_to_save.save_weights(base_path+save_w_path+condition+"_"+base+".weights.h5")

            print("finetune contrastif sur", base, "terminé")






        if condition == "coretuning" :
            
            data_gen.contrast=False
            #contrastiv_loss = NTXent(normalize=True)
            coretuning_loss = CoreTuning()
            model(np.random.random((32, 64, 64, 6)))
            model.load_weights(base_path+load_w_path)
            back = model.backbone
            classifier = astro_head(in_dim, 400)
            proj = model.head
            optim = keras.optimizers.Adam(1e-4)

            for ep in range(n_epochs) :
                if ep == 70 or ep == 90 :
                    optim.learning_rate.assign(optim.learning_rate * 0.1)
                for i, batch in enumerate(data_gen) :
                    images, labels_dict = batch
                    #print(labels_dict["reg"].shape)
                    with tf.GradientTape() as tape :
                        x = back(images, training=True)
                        pdf, reg = classifier(x, training=True)
                        z = proj(x, training=True)
                        

                        pdf_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels_dict["pdf"], pdf))
                        reg_loss = tf.reduce_mean(tf.math.sqrt(tf.reduce_sum( (labels_dict["reg"] - reg)**2, axis=1) ), axis=0)
                        core_loss = coretuning_loss.call(z, labels_dict["reg"]) * 0.1


                        total_loss = pdf_loss+reg_loss+core_loss
                    gradients = tape.gradient(total_loss, back.trainable_variables+classifier.trainable_variables+proj.trainable_variables)
                    optim.apply_gradients(zip(gradients, back.trainable_variables+classifier.trainable_variables+proj.trainable_variables))
                print("coretune epoch", ep, total_loss, pdf_loss, reg_loss, core_loss)

            model_to_save = AstroFinetune(back, classifier)
            model_to_save.save_weights(base_path+save_w_path+condition+"_"+base+".weights.h5")

            print("finetune contrastif externe sur", base, "terminé")

            



        
        
