import numpy as np
import tensorflow as tf
from generator import SupervisedGenerator, COINGenerator, MultiGen
import tensorflow.keras as keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers
from deep_models import AstroFinetune, noregu_projection_mlp, astro_head, color_mlp
from contrastiv_model import simCLRcolor1, simCLR1
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from schedulers import AlternateTreyerScheduler


base_path = "/lustre/fswork/projects/rech/dnz/ull82ct/astro/"
finetune_base = "base1.npz"
ssl_weights = "checkpoints_new_simCLR/simCLR_UD_D_norm300_ColorHead_NotRegularized_resnet50"
save_name = "resnet5k"


tsne_data = np.load(base_path+"data/cleaned_spec/cube_1_UD.npz", allow_pickle=True)
tsne_images = tsne_data["cube"]
tsne_z = np.array([m["ZSPEC"] for m in tsne_data["info"]])


all_coords = np.zeros((10, 50000, 2))  # random, supervisé, ssl , ssl finetuné, ssl finetuné + contrastive,  ssl coin


def compute_contrastive(batch, temperature, normalize=True) :
    # sépare les images x des images x'
    large_num = 1e8
    hidden1, hidden2 = tf.split(batch, 2, 0)
    batch_size = tf.shape(hidden1)[0]
    if normalize :
        hidden1 = tf.nn.l2_normalize(hidden1, axis=1)
        hidden2 = tf.nn.l2_normalize(hidden2, axis=1)
    hidden1_large = hidden1
    hidden2_large = hidden2
    labels = tf.one_hot(tf.range(batch_size), batch_size*2)    # matrice des des labels,    batch_size x 2*batch_size
    masks = tf.one_hot(tf.range(batch_size), batch_size)       # mask de shape     batch x batch

    logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / temperature       ### si normalisé cela aurait été cosine sim => shape batch, batch    distance x x
    logits_aa = logits_aa - masks * large_num    ### on rempli la diagonale de très petite valeur car forcément cosine sim entre vecteurs identique = 1
    logits_bb = tf.matmul(hidden2, hidden2_large, transpose_b=True) / temperature
    logits_bb = logits_bb - masks * large_num    ###  idem ici ==> donc là on fait distances entre x' x'
    logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature     ### sim x x'
    logits_ba = tf.matmul(hidden2, hidden1_large, transpose_b=True) / temperature     ### sim x' x 

    loss_a = tf.nn.softmax_cross_entropy_with_logits(              ### matrice labels contient info de où sont les paires positives
        labels, tf.concat([logits_ab, logits_aa], 1))              ### en concaténant ab et aa on obtient similarité de a vers toutes les autres images (en ayant mis sa propre correspondance à 0) 
    loss_b = tf.nn.softmax_cross_entropy_with_logits(              ### idem de b vers toutes les images
        labels, tf.concat([logits_ba, logits_bb], 1))
    loss = tf.reduce_mean(loss_a + loss_b)     ### moyenne des 2 et loss

    return loss
    
def compute_coin(batch, z, temperature=0.1, normalize=True) :

    large_num = 1e8
    hidden1, hidden2 = tf.split(batch, 2, 0)
    batch_size = tf.shape(hidden1)[0]
    if normalize :
        hidden1 = tf.nn.l2_normalize(hidden1, axis=1)
        hidden2 = tf.nn.l2_normalize(hidden2, axis=1)
    hidden1_large = hidden1
    hidden2_large = hidden2
    labels = tf.cast(tf.abs(tf.expand_dims(z, axis=1) - tf.expand_dims(z, axis=0))<0.2, dtype=tf.float32)  # matrice batch, batch  
    masks = tf.one_hot(tf.range(batch_size), batch_size)       # mask de shape     batch x batch

    logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / temperature       ### si normalisé cela aurait été cosine sim => shape batch, batch    distance x x
    logits_aa = logits_aa - masks * large_num    ### on rempli la diagonale de très petite valeur car forcément cosine sim entre vecteurs identique = 1
    logits_bb = tf.matmul(hidden2, hidden2_large, transpose_b=True) / temperature
    logits_bb = logits_bb - masks * large_num    ###  idem ici ==> donc là on fait distances entre x' x'
    logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature     ### sim x x'
    logits_ba = tf.matmul(hidden2, hidden1_large, transpose_b=True) / temperature     ### sim x' x 

    loss_a = tf.nn.softmax_cross_entropy_with_logits(              ### matrice labels contient info de où sont les paires positives
        labels, tf.concat([logits_ab, logits_aa], 1))              ### en concaténant ab et aa on obtient similarité de a vers toutes les autres images (en ayant mis sa propre correspondance à 0) 
    loss_b = tf.nn.softmax_cross_entropy_with_logits(              ### idem de b vers toutes les images
        labels, tf.concat([logits_ba, logits_bb], 1))
    loss = tf.reduce_mean(loss_a + loss_b)     ### moyenne des 2 et loss

    return loss 

def z_med(probas, bin_central_values) :
    cdf = np.cumsum(probas)
    index = np.argmax(cdf>=0.5)
    return bin_central_values[index]

def evaluate_model(model, bins_edges, metric_edges) :

    bins_centres = (bins_edges[1:]+bins_edges[:-1])/2

    z_true = []
    z_pred = []
    ud_smad = np.zeros((bins_centres.shape[0]))
    for file in ["cube_1_UD", "cube_2_UD", "cube_3_UD"] :
        data = np.load(base_path+"data/cleaned_spec/"+file+".npz", allow_pickle=True)
        nb_im = data["info"].shape[0]
        index = 0
        while index < nb_im :
            if index+512 > nb_im :
                im_to_infer = data["cube"][index:]
                z_to_pred = np.array([m["ZSPEC"] for m in data["info"][index:]])
            else :
                im_to_infer = data["cube"][index:index+512]
                z_to_pred = np.array([m["ZSPEC"] for m in data["info"][index:index+512]])
            index+=512
            z_true.append(z_to_pred)
            output = model(im_to_infer)
            probas = output["pdf"]
            z_meds = np.array([z_med(p, bins_centres) for p in probas])
            z_pred.append(z_meds)

    z_true = np.concatenate(z_true, axis=0)
    z_pred = np.exp(np.concatenate(z_pred, axis=0))-1

    deltas_z = (z_pred - z_true) / (1 + z_true) 

    for bin_ in range(len(metric_edges)-1) :
            
            inds = np.where((z_true>=metric_edges[bin_]) & (z_true<metric_edges[bin_+1]))
            selected_deltas = deltas_z[inds]

            ### SMAD
            median_delta_z_norm = np.median(selected_deltas)
            mad = np.median(np.abs(selected_deltas - median_delta_z_norm))
            sigma_mad = 1.4826 * mad
            ud_smad[bin_] = sigma_mad



    z_true = []
    z_pred = []
    d_smad = np.zeros((bins_centres.shape[0]))
    for file in ["cube_1_D"] :
        data = np.load(base_path+"data/cleaned_spec/"+file+".npz", allow_pickle=True)
        nb_im = data["info"].shape[0]
        index = 0
        while index < nb_im :
            if index+512 > nb_im :
                im_to_infer = data["cube"][index:]
                z_to_pred = np.array([m["ZSPEC"] for m in data["info"][index:]])
            else :
                im_to_infer = data["cube"][index:index+512]
                z_to_pred = np.array([m["ZSPEC"] for m in data["info"][index:index+512]])
            index+=512
            z_true.append(z_to_pred)
            output = model(im_to_infer)
            probas = output["pdf"]
            z_meds = np.array([z_med(p, bins_centres) for p in probas])
            z_pred.append(z_meds)


    z_true = np.concatenate(z_true, axis=0)
    z_pred = np.exp(np.concatenate(z_pred, axis=0))-1

    deltas_z = (z_pred - z_true) / (1 + z_true) 

    for bin_ in range(len(metric_edges)-1) :
            
            inds = np.where((z_true>=metric_edges[bin_]) & (z_true<metric_edges[bin_+1]))
            selected_deltas = deltas_z[inds]

            ### SMAD
            median_delta_z_norm = np.median(selected_deltas)
            mad = np.median(np.abs(selected_deltas - median_delta_z_norm))
            sigma_mad = 1.4826 * mad
            d_smad[bin_] = sigma_mad

    return ud_smad, d_smad

    
            
dataset = {"model":[], "plage":[], "ud_smad":[], "d_smad":[]}           
metric_edges = np.linspace(0, 6, 13)
metric_centres = (metric_edges[1:]+metric_edges[:-1])/2
bins_edges = np.linspace(np.log(1), np.log(7), 401)


# ------- PARTIE RANDOM ----------

random_network = ResNet50(include_top=False, weights=None, input_shape=(64, 64, 6), pooling='avg')
random_network(np.random.random((32, 64, 64, 6)))

random_features = random_network.predict(tsne_images)

tsne = TSNE(2)
tsne_coord = tsne.fit_transform(random_features)
all_coords[0] = tsne_coord

model = AstroFinetune(random_network, astro_head(2048, 400))

ud_metric, d_metric = evaluate_model(model, bins_edges, metric_edges)
for i in range(ud_metric.shape[0]) :
    dataset["model"].append("random")
    dataset["plage"].append(metric_centres[i])
    dataset["d_smad"].append(d_metric[i])
    dataset["ud_smad"].append(ud_metric[i])



# ------- ENTRAINEMENT SUPERVISE  -----------


model = AstroFinetune(ResNet50(include_top=False, weights=None, input_shape=(64, 64, 6), pooling='avg'), astro_head(2048, 400), train_back=True)
data_gen = SupervisedGenerator(base_path+"data/finetune/"+finetune_base, batch_size=256, nbins=400, apply_log=True)

model.fit(data_gen, epochs=100, callbacks=[AlternateTreyerScheduler()])

backbone = model.back 
supervised_features = backbone.predict(tsne_images)

tsne = TSNE(2)
tsne_coord = tsne.fit_transform(supervised_features)
all_coords[1] = tsne_coord


ud_metric, d_metric = evaluate_model(model, bins_edges, metric_edges)
for i in range(ud_metric.shape[0]) :
    dataset["model"].append("supervised")
    dataset["plage"].append(metric_centres[i])
    dataset["d_smad"].append(d_metric[i])
    dataset["ud_smad"].append(ud_metric[i])



# --------- SSL ET FINETUNING ----------


## RAW SSL FEATURES
base_simCLR = simCLRcolor1(ResNet50(include_top=False, weights=None, input_shape=(64, 64, 6), pooling='avg'), noregu_projection_mlp(2048), color_mlp(2048))
base_simCLR(np.random.random((32, 64, 64, 6)))
base_simCLR.load_weights(base_path+"model_save/"+ssl_weights+".weights.h5")
backbone = base_simCLR.backbone
raw_ssl_features = backbone.predict(tsne_images)
tsne = TSNE(2)
tsne_coord = tsne.fit_transform(raw_ssl_features)
all_coords[2] = tsne_coord



## FINETUNING
model = AstroFinetune(base_simCLR.backbone, astro_head(2048, 400), train_back=True)
data_gen = SupervisedGenerator(base_path+"data/finetune/"+finetune_base, batch_size=256, nbins=400, apply_log=True)
model(np.random.random((32, 64, 64, 6)))
model.fit(data_gen, epochs=100, callbacks=[AlternateTreyerScheduler()])
backbone = model.back 
finetune_ssl_features = backbone.predict(tsne_images)
tsne = TSNE(2)
tsne_coord = tsne.fit_transform(finetune_ssl_features)
all_coords[3] = tsne_coord


ud_metric, d_metric = evaluate_model(model, bins_edges, metric_edges)
for i in range(ud_metric.shape[0]) :
    dataset["model"].append("finetune ssl")
    dataset["plage"].append(metric_centres[i])
    dataset["d_smad"].append(d_metric[i])
    dataset["ud_smad"].append(ud_metric[i])


## FINETUNING EN MAINTENANT CONTRASTIVE

data_gen = COINGenerator(base_path+"data/finetune/"+finetune_base, batch_size=256, nbins=400, apply_log=True)
base_simCLR = simCLRcolor1(ResNet50(include_top=False, weights=None, input_shape=(64, 64, 6), pooling='avg'), noregu_projection_mlp(2048), color_mlp(2048))
base_simCLR(np.random.random((32, 64, 64, 6)))
base_simCLR.load_weights(base_path+"model_save/"+ssl_weights+".weights.h5")

backbone = base_simCLR.backbone
classifier = astro_head(2048, 400)
proj = base_simCLR.head
optimizer = tf.keras.optimizers.Adam(1e-4)

## train_step à la mano
for epoch in range(100) :

    for batch in data_gen :
        images, labels_dict = batch
        with tf.GradientTape(persistent=True) as tape :
            x = backbone(images, training=True)  # latent space 
            pdf, reg = classifier(x, training=True)
            projection = proj(x, training=True)

            contrastiv_loss = compute_contrastive(projection, temperature=0.1, normalize=True)
            xent = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels_dict["pdf"], pdf))
            rmse = tf.reduce_mean(tf.math.sqrt( tf.reduce_sum( (labels_dict["reg"] - reg)**2, axis=1)))
            classif_loss = xent+rmse

        gradients = tape.gradient(classif_loss, backbone.trainable_variables+classifier.trainable_variables)
        optimizer.apply_gradients(zip(gradients, backbone.trainable_variables+classifier.trainable_variables))
        gradients = tape.gradient(contrastiv_loss, backbone.trainable_variables+proj.trainable_variables)
        optimizer.apply_gradients(zip(gradients, backbone.trainable_variables+proj.trainable_variables))
        del tape

    if epoch in [70, 90]:
        optimizer.learning_rate.assign(optimizer.learning_rate * 0.1)

    data_gen.on_epoch_end()

model = AstroFinetune(backbone, classifier)
finetune_et_contrast_ssl_features = backbone.predict(tsne_images)
tsne = TSNE(2)
tsne_coord = tsne.fit_transform(finetune_et_contrast_ssl_features)
all_coords[4] = tsne_coord


ud_metric, d_metric = evaluate_model(model, bins_edges, metric_edges)
for i in range(ud_metric.shape[0]) :
    dataset["model"].append("finetune ssl + contrast")
    dataset["plage"].append(metric_centres[i])
    dataset["d_smad"].append(d_metric[i])
    dataset["ud_smad"].append(ud_metric[i])




## FINETUNING EN MAINTENANT CONTRASTIVE SUR BEAUCOUP DE DONNEES

data_gen = COINGenerator(base_path+"data/finetune/"+finetune_base, batch_size=256, nbins=400, apply_log=True)
unsup_data_gen = MultiGen([base_path+"data/cleaned_spec/", base_path+"data/cleaned_phot/"], batch_size=256, extensions=["_UD.npz", "_D.npz"], do_color=False, do_seg=False, do_mask_band=False)
base_simCLR = simCLRcolor1(ResNet50(include_top=False, weights=None, input_shape=(64, 64, 6), pooling='avg'), noregu_projection_mlp(2048), color_mlp(2048))
base_simCLR(np.random.random((32, 64, 64, 6)))
base_simCLR.load_weights(base_path+"model_save/"+ssl_weights+".weights.h5")

backbone = base_simCLR.backbone
classifier = astro_head(2048, 400)
proj = base_simCLR.head
optimizer = tf.keras.optimizers.Adam(1e-4)

## train_step à la mano
for epoch in range(100) :

    for i, batch in enumerate(data_gen) :
        images, labels_dict = batch
        unsup_images, unsup_dict = unsup_data_gen[i]

        with tf.GradientTape(persistent=True) as tape :
            x = backbone(images, training=True)  # latent space 
            pdf, reg = classifier(x, training=True)

            xbis = backbone(unsup_images, training=True)
            projection = proj(xbis, training=True)

            contrastiv_loss = compute_contrastive(projection, temperature=0.1, normalize=True)

            xent = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels_dict["pdf"], pdf))
            rmse = tf.reduce_mean(tf.math.sqrt( tf.reduce_sum( (labels_dict["reg"] - reg)**2, axis=1)))
            classif_loss = xent+rmse

        gradients = tape.gradient(classif_loss, backbone.trainable_variables+classifier.trainable_variables)
        optimizer.apply_gradients(zip(gradients, backbone.trainable_variables+classifier.trainable_variables))
        gradients = tape.gradient(contrastiv_loss, backbone.trainable_variables+proj.trainable_variables)
        optimizer.apply_gradients(zip(gradients, backbone.trainable_variables+proj.trainable_variables))
        del tape

    if epoch in [70, 90]:
        optimizer.learning_rate.assign(optimizer.learning_rate * 0.1)

    if epoch % 10 == 0 and epoch > 0 :
        unsup_data_gen._load_data()

    data_gen.on_epoch_end()
    unsup_data_gen.on_epoch_end()


model = AstroFinetune(backbone, classifier)

finetune_et_contrast_all_ssl_features = backbone.predict(tsne_images)
tsne = TSNE(2)
tsne_coord = tsne.fit_transform(finetune_et_contrast_all_ssl_features)
all_coords[5] = tsne_coord



ud_metric, d_metric = evaluate_model(model, bins_edges, metric_edges)
for i in range(ud_metric.shape[0]) :
    dataset["model"].append("finetune ssl + contrast sur tout")
    dataset["plage"].append(metric_centres[i])
    dataset["d_smad"].append(d_metric[i])
    dataset["ud_smad"].append(ud_metric[i])








# -------- COIN ET FINETUNING ----------     50 EPOQUE COIN ET 50 EPOQUE TREYER


data_gen = COINGenerator(base_path+"data/finetune/"+finetune_base, batch_size=256, nbins=400, contrast=True, apply_log=True)
base_simCLR = simCLRcolor1(ResNet50(include_top=False, weights=None, input_shape=(64, 64, 6), pooling='avg'), noregu_projection_mlp(2048), color_mlp(2048))
base_simCLR(np.random.random((32, 64, 64, 6)))
base_simCLR.load_weights(base_path+"model_save/"+ssl_weights+".weights.h5")

backbone = base_simCLR.backbone
classifier = astro_head(2048, 400)
proj = base_simCLR.head
optimizer = tf.keras.optimizers.Adam(1e-4)

## train_step à la mano
for epoch in range(50) :

    for i, batch in enumerate(data_gen) :
        images, labels_dict = batch

        with tf.GradientTape(persistent=True) as tape :
            x = backbone(images, training=True)  # latent space 
            projection = proj(x, training=True)

            consup_loss = compute_coin(projection, labels_dict["reg"])

        gradients = tape.gradient(contrastiv_loss, backbone.trainable_variables+proj.trainable_variables)
        optimizer.apply_gradients(zip(gradients, backbone.trainable_variables+proj.trainable_variables))
        del tape

    if epoch in [70, 90]:
        optimizer.learning_rate.assign(optimizer.learning_rate * 0.1)


    data_gen.on_epoch_end()


coin_features = backbone.predict(tsne_images)
tsne = TSNE(2)
tsne_coord = tsne.fit_transform(coin_features)
all_coords[6] = tsne_coord


### A PARTIR DE LA ON VA RE UTILISER CE QUI A ETE ENTRAINE AVEC COIN


coin_back = ResNet50(include_top=False, weights=None, input_shape=(64, 64, 6), pooling='avg')
coin_proj = noregu_projection_mlp(2048)

coin_back(np.random.random((32, 64, 64, 6)))
coin_back.set_weights(backbone.get_weights())
coin_proj(np.random.random((32,64, 64, 6)))
coin_proj.set_weights(proj.get_weights())
classifier = astro_head(2048, 400)


## PARTIE 1   FINETUNE TREYER BASIQUE

data_gen = COINGenerator(base_path+"data/finetune/"+finetune_base, batch_size=256, nbins=400, contrast=True, apply_log=True)

for epoch in range(50) :

    for i, batch in enumerate(data_gen) :
        images, labels_dict = batch

        with tf.GradientTape(persistent=True) as tape :
            x = coin_back(images, training=True)  # latent space 
            pdf, reg = classifier(x, training=True)
            #projection = proj(x, training=True)

            #contrastiv_loss = compute_contrastive(projection, temperature=0.1, normalize=True)
            #consup_loss = compute_coin(projection, labels_dict["reg"])

            xent = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels_dict["pdf"], pdf))
            rmse = tf.reduce_mean(tf.math.sqrt( tf.reduce_sum( (labels_dict["reg"] - reg)**2, axis=1)))
            classif_loss = xent+rmse

        gradients = tape.gradient(classif_loss, coin_back.trainable_variables+classifier.trainable_variables)
        optimizer.apply_gradients(zip(gradients, coin_back.trainable_variables+classifier.trainable_variables))
        #gradients = tape.gradient(contrastiv_loss, backbone.trainable_variables+proj.trainable_variables)
        #optimizer.apply_gradients(zip(gradients, backbone.trainable_variables+proj.trainable_variables))
        del tape

    if epoch in [35, 45]:
        optimizer.learning_rate.assign(optimizer.learning_rate * 0.1)


    data_gen.on_epoch_end()


model = AstroFinetune(backbone, classifier)

coin_treyer_features = backbone.predict(tsne_images)
tsne = TSNE(2)
tsne_coord = tsne.fit_transform(coin_treyer_features)
all_coords[7] = tsne_coord



ud_metric, d_metric = evaluate_model(model, bins_edges, metric_edges)
for i in range(ud_metric.shape[0]) :
    dataset["model"].append("ssl + coin + treyer")
    dataset["plage"].append(metric_centres[i])
    dataset["d_smad"].append(d_metric[i])
    dataset["ud_smad"].append(ud_metric[i])





## ETAPE 2 : COIN + TREYER + CONTRASTIV

coin_back(np.random.random((32, 64, 64, 6)))
coin_back.set_weights(backbone.get_weights())
coin_proj(np.random.random((32,64, 64, 6)))
coin_proj.set_weights(proj.get_weights())
classifier = astro_head(2048, 400)

data_gen = COINGenerator(base_path+"data/finetune/"+finetune_base, batch_size=256, nbins=400, contrast=True, apply_log=True)

for epoch in range(50) :

    for i, batch in enumerate(data_gen) :
        images, labels_dict = batch

        with tf.GradientTape(persistent=True) as tape :
            x = coin_back(images, training=True)  # latent space 
            pdf, reg = classifier(x, training=True)
            projection = coin_proj(x, training=True)

            contrastiv_loss = compute_contrastive(projection, temperature=0.1, normalize=True)
            #consup_loss = compute_coin(projection, labels_dict["reg"])

            xent = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels_dict["pdf"], pdf))
            rmse = tf.reduce_mean(tf.math.sqrt( tf.reduce_sum( (labels_dict["reg"] - reg)**2, axis=1)))
            classif_loss = xent+rmse

        gradients = tape.gradient(classif_loss, coin_back.trainable_variables+classifier.trainable_variables)
        optimizer.apply_gradients(zip(gradients, coin_back.trainable_variables+classifier.trainable_variables))
        gradients = tape.gradient(contrastiv_loss, coin_back.trainable_variables+coin_proj.trainable_variables)
        optimizer.apply_gradients(zip(gradients, coin_back.trainable_variables+coin_proj.trainable_variables))
        del tape

    if epoch in [35, 45]:
        optimizer.learning_rate.assign(optimizer.learning_rate * 0.1)


    data_gen.on_epoch_end()


model = AstroFinetune(backbone, classifier)

coin_treyer_contrastiv_features = backbone.predict(tsne_images)
tsne = TSNE(2)
tsne_coord = tsne.fit_transform(coin_treyer_contrastiv_features)
all_coords[8] = tsne_coord


ud_metric, d_metric = evaluate_model(model, bins_edges, metric_edges)
for i in range(ud_metric.shape[0]) :
    dataset["model"].append("ssl + coin + treyer + contrastiv")
    dataset["plage"].append(metric_centres[i])
    dataset["d_smad"].append(d_metric[i])
    dataset["ud_smad"].append(ud_metric[i])






## ETAPE 3 : COIN + TREYER + CONSUP

coin_back(np.random.random((32, 64, 64, 6)))
coin_back.set_weights(backbone.get_weights())
coin_proj(np.random.random((32,64, 64, 6)))
coin_proj.set_weights(proj.get_weights())
classifier = astro_head(2048, 400)

data_gen = COINGenerator(base_path+"data/finetune/"+finetune_base, batch_size=256, nbins=400, contrast=True, apply_log=True)

for epoch in range(50) :

    for i, batch in enumerate(data_gen) :
        images, labels_dict = batch

        with tf.GradientTape(persistent=True) as tape :
            x = coin_back(images, training=True)  # latent space 
            pdf, reg = classifier(x, training=True)
            projection = coin_proj(x, training=True)

            #contrastiv_loss = compute_contrastive(projection, temperature=0.1, normalize=True)
            consup_loss = compute_coin(projection, labels_dict["reg"])

            xent = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels_dict["pdf"], pdf))
            rmse = tf.reduce_mean(tf.math.sqrt( tf.reduce_sum( (labels_dict["reg"] - reg)**2, axis=1)))
            classif_loss = xent+rmse

        gradients = tape.gradient(classif_loss, coin_back.trainable_variables+classifier.trainable_variables)
        optimizer.apply_gradients(zip(gradients, coin_back.trainable_variables+classifier.trainable_variables))
        gradients = tape.gradient(contrastiv_loss, coin_back.trainable_variables+coin_proj.trainable_variables)
        optimizer.apply_gradients(zip(gradients, coin_back.trainable_variables+coin_proj.trainable_variables))
        del tape

    if epoch in [35, 45]:
        optimizer.learning_rate.assign(optimizer.learning_rate * 0.1)


    data_gen.on_epoch_end()


model = AstroFinetune(backbone, classifier)

coin_treyer_consup_features = backbone.predict(tsne_images)
tsne = TSNE(2)
tsne_coord = tsne.fit_transform(coin_treyer_consup_features)
all_coords[9] = tsne_coord

ud_metric, d_metric = evaluate_model(model, bins_edges, metric_edges)
for i in range(ud_metric.shape[0]) :
    dataset["model"].append("ssl + coin + treyer + consup")
    dataset["plage"].append(metric_centres[i])
    dataset["d_smad"].append(d_metric[i])
    dataset["ud_smad"].append(ud_metric[i])


import pandas as pd

df = pd.DataFrame(dataset)
df.to_csv(base_path+"data/metrics_save/coin_comp"+save_name+".csv", index=False)


np.savez(base_path+"data/metrics_save/tsnes"+save_name+".npz", array1=all_coords, array2=tsne_z)



















