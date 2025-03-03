import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from regularizers import CosineDistRegularizer, VarRegularizer, TripletCosineRegularizer


class simCLR(keras.Model) :
    def __init__(self, backbone, head, temp=0.7) :
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.temp=temp

    def call(self, input, training=True, return_all=True) :
        if return_all :
            c1, c2, c3, c4, c5, c6, c7, c8, c9, flat, l1, x = self.backbone(input, training=training)
        else :
            x = self.backbone(input, training=training)
        z = self.head(x, training=training)
        return z

    def train_step(self, data) : 
        images, labels = data
        with tf.GradientTape(persistent=True) as tape :
            z = self(images)

            contrastiv_loss = self.loss(z, self.temp)

            regu = sum(self.losses)
            total_loss = contrastiv_loss + regu
        gradients = tape.gradient(total_loss, self.backbone.trainable_variables + self.head.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.backbone.trainable_variables + self.head.trainable_variables))

        del tape
        return {"contrastiv_loss":contrastiv_loss, "regularization" : regu}
    

class simCLRcolor1(keras.Model) :
    def __init__(self, backbone, head, mlp, regularizer=None, temp=0.7) :
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.mlp=mlp
        self.regularizer = regularizer
        self.is_regu = True if regularizer is not None else False
        self.temp=temp
        self.lam = 0.1


    def call(self, input, training=True) :
        x = self.backbone(input, training=training)
        z = self.head(x, training=training)
        c = self.mlp(x, training=True)
        return z, c, x

    def train_step(self, data) : 
        images, labels = data

        with tf.GradientTape(persistent=True) as tape :
            z, c, x = self(images)

            contrastiv_loss = self.loss(z, self.temp)
            color_loss = self.lam * tf.keras.losses.MSE(labels, c)
            
            if self.is_regu :
                regu_loss = self.regularizer(x, labels)

            regularizers = sum(self.losses)
            total_loss = contrastiv_loss+regularizers
  
        gradients = tape.gradient(total_loss, self.backbone.trainable_variables + self.head.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.backbone.trainable_variables + self.head.trainable_variables))

        gradients = tape.gradient(color_loss, self.backbone.trainable_variables + self.mlp.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.backbone.trainable_variables + self.mlp.trainable_variables))

        if self.is_regu :
            gradients = tape.gradient(regu_loss, self.backbone.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.backbone.trainable_variables))
            del tape
            return {"contrastiv_loss":contrastiv_loss, "mse_color_loss":color_loss, "regu_loss":regu_loss}

        del tape
        return {"contrastiv_loss":contrastiv_loss, "activity_regu":regularizers, "mse_color_loss":color_loss}
    

class simCLRcolor2(keras.Model) :
    def __init__(self, backbone, head, temp=0.7) :
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.temp=temp
        self.lam = 0.05

    def call(self, input, training=True) :
        x = self.backbone(input, training=training)
        z = self.head(x, training=training)
        return z, x

    def train_step(self, data) : 
        images, labels = data

        with tf.GradientTape(persistent=True) as tape :
            z, x = self(images)

            contrastiv_loss = self.loss(z, self.temp)

            pairwise_cosine = tf.keras.losses.cosine_similarity(x, x) ## cosine similarity entre tous les éléments : on vise orthogonalité donc minimiser somme de la matrice
            # résultat entre -1 et 1  

            color_cosine_dist = 1 - tf.keras.losses.cosine_similarity(labels, labels)  # distance cosinus dans l'espace couleur entre tous les éléments
            ## Résultat entre 0 et 2 : 0 = vecteurs proches dans l'espace couleur    2 = vecteurs opposés  ==> si proche on diminue pénalité pour l'orthogonalité recherchée

            weighted_pairwise_cosine = tf.multiply(pairwise_cosine, color_cosine_dist)
            cosine_loss = tf.reduce_sum(weighted_pairwise_cosine) * self.lam


  
        gradients = tape.gradient(contrastiv_loss, self.backbone.trainable_variables + self.head.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.backbone.trainable_variables + self.head.trainable_variables))

        gradients = tape.gradient(cosine_loss, self.backbone.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.backbone.trainable_variables))

        del tape
        return {"contrastiv_loss":contrastiv_loss, "cosine_loss":cosine_loss}

import random

class simCLRcolor3(keras.Model) :
    def __init__(self, backbone, head, temp=0.7) :
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.temp=temp
        self.lam = 0.05

    def call(self, input, training=True) :
        x = self.backbone(input, training=training)
        z = self.head(x, training=training)
        return z, x
    
    def kmeans(self, data, k, num_iterations=10):
        centroids = tf.Variable(tf.gather(data, tf.random.shuffle(tf.range(tf.shape(data)[0]))[:k]))

        for i in range(num_iterations):
            # Calcul des distances (Euclidiennes)
            expanded_data = tf.expand_dims(data, 1)  # Shape: (512, 1, 1024)
            expanded_centroids = tf.expand_dims(centroids, 0)  # Shape: (1, k, 1024)
            distances = tf.reduce_sum(tf.square(expanded_data - expanded_centroids), axis=2)  # Shape: (512, k)

            # Assignation des clusters
            cluster_assignments = tf.argmin(distances, axis=1)  # Shape: (512,)

            # Mise à jour des centroides (vectorisée)
            mask = tf.one_hot(cluster_assignments, k)  # Shape: (512, k)
            cluster_sums = tf.matmul(mask, data, transpose_a=True)  # Shape: (k, 1024)
            cluster_counts = tf.reduce_sum(mask, axis=0)  # Shape: (k,)
            new_centroids = cluster_sums / tf.expand_dims(cluster_counts + 1e-8, axis=1)

            centroids.assign(new_centroids)

        return centroids, cluster_assignments

    def train_step(self, data) : 
        images, labels = data

        with tf.GradientTape(persistent=True) as tape :
            z, x = self(images)

            contrastiv_loss = self.loss(z, self.temp)

            centroids, clusters = self.kmeans(tf.nn.l2_normalize(x, axis=1), k=random.randint(5, 20))

            mask = tf.one_hot(clusters, len(centroids))  # Shape: (512, k)
            #cluster_data = tf.matmul(mask, x, transpose_a=True)  # Agrégation des données par cluster
            cluster_labels = tf.matmul(mask, tf.cast(labels, tf.float32), transpose_a=True)

            var_intra = tf.reduce_sum(
                tf.math.reduce_variance(cluster_labels, axis=1)
            ) * self.lam



  
        gradients = tape.gradient(contrastiv_loss, self.backbone.trainable_variables + self.head.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.backbone.trainable_variables + self.head.trainable_variables))

        gradients = tape.gradient(var_intra, self.backbone.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.backbone.trainable_variables))

        del tape
        return {"contrastiv_loss":contrastiv_loss, "var intra":var_intra}


class simCLRmultitask(keras.Model) :
    def __init__(self, backbone, head, do_color, color_head, do_seg, segmentor, do_reco, deconvolutor, temp=0.7) :
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.do_color = do_color
        self.color_head = color_head
        self.do_seg = do_seg
        self.segmentor = segmentor
        self.do_reco = do_reco
        self.deconvolutor = deconvolutor
        self.temp=temp
        self.lam = 0.1

    def call(self, input, training=True) :
        x, flat, c4 = self.backbone(input, training=training)   # retourne fin du backbone, le résultat du flatten et de la conv4
        z = self.head(x, training=training)
        output_dict = {"backbone_output":z}

        if self.do_color :
            c = self.color_head(x, training=training)
            output_dict["color_output"] = c
        if self.do_seg :
            seg_mask = self.segmentor([flat, c4], training=training)
            output_dict["seg_output"] = seg_mask
        if self.do_reco :
            recon = self.deconvolutor(flat, training=training)
            output_dict["recon_output"] = recon      

        return output_dict

    def train_step(self, data) : 
        images, labels_dict = data

        with tf.GradientTape(persistent=True) as tape :
            output_dict = self(images)

            contrastiv_loss = self.loss(output_dict["backbone_output"], self.temp)

            loss_dict = {"contrastiv_loss":contrastiv_loss}

            if self.do_color : 
                color_labels = labels_dict["color"]
                color_loss = self.lam * tf.keras.losses.mean_squared_error(color_labels[0], output_dict["color_output"])
                loss_dict["color_loss"] = tf.reduce_mean(color_loss)

            if self.do_seg : 
                true_seg = labels_dict["seg_mask"]
                pred_seg = output_dict["seg_output"]
                seg_loss = tf.keras.losses.binary_crossentropy(true_seg, pred_seg)
                loss_dict["seg_loss"] = tf.reduce_mean(seg_loss)

            if self.do_reco :
                #masked_indexes = labels_dict["masked_indexes"]
                reconstruction = output_dict["recon_output"]
                diff = tf.abs(images - reconstruction)
                #diff = tf.abs(tf.where(masked_indexes, images, 0) - tf.where(masked_indexes, reconstruction, 0))
                recon_loss = tf.reduce_mean(tf.math.log(diff + 1e-8), axis=(1, 2, 3))  # Stabilisation avec epsilon
                loss_dict["recon_loss"] = tf.reduce_mean(recon_loss) 


        
        gradients = tape.gradient(contrastiv_loss, self.backbone.trainable_variables + self.head.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.backbone.trainable_variables + self.head.trainable_variables))

        if self.do_color :
            gradients = tape.gradient(loss_dict["color_loss"], self.backbone.trainable_variables + self.color_head.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.backbone.trainable_variables + self.color_head.trainable_variables))

        if self.do_seg :
            gradients = tape.gradient(loss_dict["seg_loss"], self.backbone.trainable_variables + self.segmentor.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.backbone.trainable_variables + self.segmentor.trainable_variables))

        if self.do_reco :
            gradients = tape.gradient(loss_dict["recon_loss"], self.backbone.trainable_variables + self.deconvolutor.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.backbone.trainable_variables + self.deconvolutor.trainable_variables))


        del tape
        return loss_dict

class simCLR_adversarial(keras.Model) :
    def __init__(self, backbone, head, adversaire, temp=0.7) :
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.adversaire=adversaire
        self.temp=temp
        self.flat_vector = tf.constant(tf.ones((1, 2), dtype=tf.float32)/tf.cast(2, dtype=tf.float32))
        self.adverse_accuracy = tf.keras.metrics.BinaryAccuracy(name="survey_accuracy")  # flatten

        self.adversarial_layers = []
        for i, lay in enumerate(self.backbone.layers) : 
            if lay.name == 'flatten' :
                self.adversarial_layers.append(lay)
                break
            else :
                self.adversarial_layers.append(lay)
        self.adversarial_variables = [var for layer in self.adversarial_layers for var in layer.trainable_variables]

    def call(self, input, training=True) :
        x, flat = self.backbone(input, training=training)
        survey_predictions = self.adversaire(flat, training=training)
        z = self.head(x, training=training)
        return z, survey_predictions

    def train_step(self, data) : 
        images, true_survey = data
        with tf.GradientTape(persistent=True) as tape :
            z, surv = self(images)

            contrastiv_loss = self.loss(z, self.temp)
            flat_distr = tf.tile(self.flat_vector, [tf.shape(images)[0], 1])
            adversarial_loss = tf.keras.losses.kl_divergence(flat_distr, surv)  # KL divergence entre distribution flat et prédiction de l'adverse
            # on pénalise backbone si la classif est facile pour la tête
            survey_classif = tf.keras.losses.binary_crossentropy(true_survey, surv)  # crossentropy pour MLP de classif sur le survey

            self.adverse_accuracy.update_state(true_survey, surv)
  
        gradients = tape.gradient(contrastiv_loss, self.backbone.trainable_variables + self.head.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.backbone.trainable_variables + self.head.trainable_variables))

        gradients = tape.gradient(survey_classif, self.adversaire.trainable_variables)  # MAJ DU CLASSIFIER SUR SURVEY
        self.optimizer.apply_gradients(zip(gradients, self.adversaire.trainable_variables))

        gradients = tape.gradient(adversarial_loss, self.adversarial_variables)
        self.optimizer.apply_gradients(zip(gradients, self.adversarial_variables))

        del tape
        return {"contrastiv_loss":contrastiv_loss,
                "survey accuracy":self.adverse_accuracy.result(), 
                "adversarial_kl":adversarial_loss,
                "adversarial_crossent":survey_classif}


class simCLR_adversarial2(keras.Model) :
    def __init__(self, backbone, head, adversaire, temp=0.7) :
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.adversaire=adversaire
        self.temp=temp
        #self.flat_vector = tf.constant(tf.ones((1, 2), dtype=tf.float32)/tf.cast(2, dtype=tf.float32))
        self.adverse_accuracy = tf.keras.metrics.BinaryAccuracy(name="survey_accuracy")  # flatten

        self.adversarial_layers = []
        for i, lay in enumerate(self.backbone.layers) :
            if lay.name == 'flatten' :
                self.adversarial_layers.append(lay)
                break
            else :
                self.adversarial_layers.append(lay)
        self.adversarial_variables = [var for layer in self.adversarial_layers for var in layer.trainable_variables]

    def call(self, input, training=True) :
        x, flat = self.backbone(input, training=training)
        survey_predictions = self.adversaire(flat, training=training)
        z = self.head(x, training=training)
        return z, survey_predictions

    def train_step(self, data) :
        images, true_survey = data
        with tf.GradientTape(persistent=True) as tape :
            z, surv = self(images)

            contrastiv_loss = self.loss(z, self.temp)
            #flat_distr = tf.tile(self.flat_vector, [tf.shape(images)[0], 1])
            #adversarial_loss = tf.keras.losses.kl_divergence(flat_distr, surv)  # KL divergence entre distribution flat et prédiction de l'adverse
            # on pénalise backbone si la classif est facile pour la tête
            survey_classif = tf.keras.losses.binary_crossentropy(true_survey, surv)  # crossentropy pour MLP de classif sur le survey
            adversarial_loss = -survey_classif
            self.adverse_accuracy.update_state(true_survey, surv)

        gradients = tape.gradient(contrastiv_loss, self.backbone.trainable_variables + self.head.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.backbone.trainable_variables + self.head.trainable_variables))

        gradients = tape.gradient(survey_classif, self.adversaire.trainable_variables)  # MAJ DU CLASSIFIER SUR SURVEY
        self.optimizer.apply_gradients(zip(gradients, self.adversaire.trainable_variables))

        gradients = tape.gradient(adversarial_loss, self.adversarial_variables)
        self.optimizer.apply_gradients(zip(gradients, self.adversarial_variables))

        del tape
        return {"contrastiv_loss":contrastiv_loss,
                "survey accuracy":self.adverse_accuracy.result(),
                "adversarial_kl":adversarial_loss,
                "adversarial_crossent":survey_classif}


            
class ContrastivLoss(keras.losses.Loss) :
    def __init__(self) :
        super().__init__()
        self.large_num = 1e8

    def call(self, batch, temperature=1) :
        hidden1, hidden2 = tf.split(batch, 2, 0)
        batch_size = tf.shape(hidden1)[0]

        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = tf.one_hot(tf.range(batch_size), batch_size*2)
        masks = tf.one_hot(tf.range(batch_size), batch_size)

        logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / temperature
        logits_aa = logits_aa - masks * self.large_num
        logits_bb = tf.matmul(hidden2, hidden2_large, transpose_b=True) / temperature
        logits_bb = logits_bb - masks * self.large_num
        logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature
        logits_ba = tf.matmul(hidden2, hidden1_large, transpose_b=True) / temperature

        loss_a = tf.nn.softmax_cross_entropy_with_logits(
            labels, tf.concat([logits_ab, logits_aa], 1))
        loss_b = tf.nn.softmax_cross_entropy_with_logits(
            labels, tf.concat([logits_ba, logits_bb], 1))
        loss = tf.reduce_mean(loss_a + loss_b)

        return loss
    






class BarlowTwins(keras.Model) :
    def __init__(self, backbone, head, lam=5e-3) : 
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.lam = lam
        self.bn = layers.BatchNormalization()
    
    def call(self, inputs, training=True) :
        x = self.backbone(inputs, training=training)
        z = self.head(x, training=training)
        return self.bn(z, training=training)

    def train_step(self, data) :
        images, labels = data
        with tf.GradientTape() as tape :
            z = self(images)

            loss = self.loss(z, self.lam)

        gradients = tape.gradient(loss, self.backbone.trainable_variables+self.head.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.backbone.trainable_variables+self.head.trainable_variables))

        return {"barlow_twin_loss":loss}

class BarlowTwinsLoss(keras.losses.Loss) :
    def __init__(self) :
        super().__init__()
    
    def call(self, z, lam) :
        z1, z2 = tf.split(z, 2, 0)
        c = tf.matmul(z1, z2, transpose_a=True)
        batch_size = tf.shape(z1)[0]
        c = c / tf.cast(batch_size, c.dtype)
        on_diag = tf.reduce_sum(tf.square(tf.linalg.diag_part(c) - 1))
        off_diag = tf.reduce_sum(tf.square(c - tf.linalg.diag(tf.linalg.diag_part(c))))
        loss = on_diag + lam * off_diag
        return loss
    


class VICReg(keras.Model) :
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def call(self, inputs, training=True) :
        x = self.backbone(inputs, training=training)
        return self.head(x, training=training)

    def train_step(self, data) :
        images, labels = data
        with tf.GradientTape() as tape :
            z = self(images) 
            loss = self.loss(z, z)

        gradients = tape.gradient(loss, self.backbone.trainable_variables+self.head.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.backbone.trainable_variables+self.head.trainable_variables))
        
        return {"VICReg loss":loss}

class VICRegLoss(keras.losses.Loss) :
    def __init__(self, la=25, mu=25, nu=1) :
        super().__init__()
        self.la = la
        self.mu=mu
        self.nu=nu
        self.gamma=1

    def call(self, z, ypred) :
        z1, z2 = tf.split(z, 2, 0)
        n = tf.cast(tf.shape(z)[0], tf.float32)   
        d = tf.cast(tf.shape(z)[1], tf.float32)
        var1 = tf.reduce_mean(tf.nn.relu(self.gamma - tf.math.reduce_std(z1, axis=0)))  # scalaire - 256   => 1
        var2 = tf.reduce_mean(tf.nn.relu(self.gamma - tf.math.reduce_std(z2, axis=0)))

        invariance = tf.keras.losses.mse(z1, z2)

        mu1 = tf.reduce_mean(z1, axis=0)
        z_centered1 = z1 - mu1
        cov = tf.matmul(z_centered1, z_centered1, transpose_a=True) / (n - 1)
        cov_squared = tf.square(cov)
        off_diag = tf.reduce_sum(cov_squared) - tf.reduce_sum(tf.linalg.diag_part(cov_squared))
        d = tf.cast(tf.shape(z)[1], tf.float32)
        cov1 = off_diag / d

        mu2 = tf.reduce_mean(z2, axis=0)
        z_centered2 = z2 - mu2
        cov = tf.matmul(z_centered2, z_centered2, transpose_a=True) / (n - 1)
        cov_squared = tf.square(cov)
        off_diag = tf.reduce_sum(cov_squared) - tf.reduce_sum(tf.linalg.diag_part(cov_squared))
        d = tf.cast(tf.shape(z)[1], tf.float32)
        cov2 = off_diag / d

        return self.la * invariance + self.mu *(var1+var2) + self.nu*(cov1+cov2)

