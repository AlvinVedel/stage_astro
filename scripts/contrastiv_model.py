import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import random


class simCLR(keras.Model) :
    def __init__(self, backbone, head, regularization={"do":False}, color_head={"do":False}, segmentor={"do":False}, deconvolutor={"do":False}, 
                 adversarial={"do":False, "metric":tf.keras.metrics.BinaryAccuracy()}, intermediaires_outputs=None, temp=0.7) :
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.regu_params = regularization
        self.color_params = color_head
        self.segm_params = segmentor
        self.recon_params = deconvolutor
        self.adversarial_params = adversarial
        if intermediaires_outputs is not None :
            self.backbone = keras.Model(inputs=self.backbone.input, outputs=[self.backbone.layers[-1].output]+[self.backbone.layers[i].output for i in intermediaires_outputs])
        self.temp=temp
        self.lam = 0.1

    def call(self, input, training=True) :
        outputs = self.backbone(input, training=training)   # retourne fin du backbone, le résultat du flatten et de la conv4

        if isinstance(outputs, list) :

            z = self.head(outputs[0], training=training)
            output_dict = {"backbone_output":z}


            if self.regu_params["do"] :
                output_dict["latent_output"] = outputs[0]

            if self.color_params["do"] :
                c = self.color_params["network"](outputs[0], training=training)
                output_dict["color_output"] = c

            if self.segm_params["do"] :
                flat = outputs[self.segm_params["need"][0]]
                c4 = outputs[self.segm_params["need"][1]]
                seg_mask = self.segm_params["network"]([flat, c4], training=training)
                output_dict["seg_output"] = seg_mask

            if self.recon_params["do"] :
                flat = outputs[self.recon_params["need"][0]]
                recon = self.recon_params["network"](flat, training=training)
                output_dict["recon_output"] = recon      


            if self.adversarial_params["do"] :

                flat = outputs[self.adversarial_params["need"][0]]
                probas = self.adversarial_params["network"](flat, training=training)
                output_dict["adversarial_output"] = probas

        else :

            z = self.head(outputs, training=training)
            output_dict = {"backbone_output":z}

            if self.color_params["do"] :
                c = self.color_params["network"](outputs, training=training)
                output_dict["color_output"] = c


        return output_dict

    def train_step(self, data) : 


        images, labels_dict = data

        with tf.GradientTape(persistent=True) as tape :
            output_dict = self(images)

            contrastiv_loss = self.loss(output_dict["backbone_output"], self.temp)

            loss_dict = {"contrastiv_loss":contrastiv_loss}
            loss_dict["regularization"] = sum(self.losses)

            basic_loss = contrastiv_loss + loss_dict["regularization"]

            

            if self.color_params["do"] : 
                color_labels = labels_dict["color"]
                color_loss = tf.keras.losses.mean_squared_error(color_labels, output_dict["color_output"])
                loss_dict["color_loss"] = self.color_params["weight"] * tf.reduce_mean(color_loss)

                if self.regu_params["do"] :
                    additionnal_regu_loss = self.regu_params["weight"] * self.regu_params["regularizer"](output_dict["latent_output"], output_dict["color_output"])
                    loss_dict["backregu_loss"] = additionnal_regu_loss

            if self.segm_params["do"] : 
                true_seg = labels_dict["seg_mask"]
                pred_seg = output_dict["seg_output"]
                seg_loss = tf.keras.losses.binary_crossentropy(true_seg, pred_seg) 
                loss_dict["seg_loss"] = tf.reduce_mean(seg_loss)* self.segm_params["weight"]

            if self.recon_params["do"] :
                #masked_indexes = labels_dict["masked_indexes"]
                reconstruction = output_dict["recon_output"]
                diff = tf.abs(images - reconstruction)
                #diff = tf.abs(tf.where(masked_indexes, images, 0) - tf.where(masked_indexes, reconstruction, 0))
                recon_loss = tf.reduce_mean(tf.math.log(diff + 1e-8), axis=(1, 2, 3))  # Stabilisation avec epsilon
                loss_dict["recon_loss"] = tf.reduce_mean(recon_loss) * self.recon_params["weight"]

            if self.adversarial_params["do"] :
                survey_labels = labels_dict["survey"]
                classif_loss = tf.keras.losses.binary_crossentropy(survey_labels, output_dict["adversarial_output"]) * self.adversarial_params["weight"]
                adversarial_loss = -classif_loss
                loss_dict["classif_loss"] = classif_loss
                loss_dict["adversarial_loss"] = adversarial_loss

                self.adversarial_params["metric"].update_state(survey_labels, output_dict["adversarial_output"])
                loss_dict["adv_acc"] = self.adversarial_params["metric"].result()


        
        gradients = tape.gradient(basic_loss, self.backbone.trainable_variables + self.head.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.backbone.trainable_variables + self.head.trainable_variables))
        

        if self.color_params["do"] :
            gradients = tape.gradient(loss_dict["color_loss"], self.backbone.trainable_variables + self.color_params["network"].trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.backbone.trainable_variables + self.color_params["network"].trainable_variables))

            if self.regu_params["do"] :
                gradients = tape.gradient(loss_dict["backregu_loss"], self.backbone.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.backbone.trainable_variables))

        if self.segm_params["do"] :
            gradients = tape.gradient(loss_dict["seg_loss"], self.backbone.trainable_variables + self.segm_params["network"].trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.backbone.trainable_variables + self.segm_params["network"].trainable_variables))

        if self.recon_params["do"] :
            gradients = tape.gradient(loss_dict["recon_loss"], self.backbone.trainable_variables + self.recon_params["network"].trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.backbone.trainable_variables + self.recon_params["network"].trainable_variables))

        if self.adversarial_params["do"] :
            gradients = tape.gradient(loss_dict["classif_loss"], self.adversarial_params["network"].trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.adversarial_params["network"].trainable_variables))

            gradients = tape.gradient(loss_dict["adversarial_loss"], self.backbone.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.backbone.trainable_variables))


        del tape
        return loss_dict


class simCLR1(keras.Model) :
    def __init__(self, backbone, head, temp=0.7) :
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.temp=temp

    def call(self, input, training=True) :
        x = self.backbone(input, training=training)
        z = self.head(x, training=training)
        return z

    def train_step(self, data) : 
        images, labels = data
        with tf.GradientTape(persistent=True) as tape :
            z = self(images)

            contrastiv_loss = self.loss(z, self.temp)

            regu_loss = sum(self.losses)
            total_loss = regu_loss + contrastiv_loss
  
        gradients = tape.gradient(total_loss, self.backbone.trainable_variables + self.head.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.backbone.trainable_variables + self.head.trainable_variables))


        del tape
        return {"contrastiv_loss":contrastiv_loss, "regu_loss":regu_loss}


class simCLRcolor1(keras.Model) :
    def __init__(self, backbone, head, color_head, temp=0.7) :
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.color_head = color_head
        self.temp=temp

    def call(self, input, training=True) :
        x = self.backbone(input, training=training)
        z = self.head(x, training=training)
        c = self.color_head(x, training=training)
        return z, c

    def train_step(self, data) : 
        images, labels = data
        with tf.GradientTape(persistent=True) as tape :
            z, c = self(images)
            #print(images.shape, c.shape, labels["color"].shape)
            contrastiv_loss = self.loss(z, self.temp)
            color_loss = tf.keras.losses.mean_squared_error(labels["color"], c)

            regu_loss = sum(self.losses)
            total_loss = regu_loss + contrastiv_loss
  
        gradients = tape.gradient(total_loss, self.backbone.trainable_variables + self.head.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.backbone.trainable_variables + self.head.trainable_variables))

        gradients = tape.gradient(color_loss, self.backbone.trainable_variables + self.color_head.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.backbone.trainable_variables + self.color_head.trainable_variables))

        del tape
        return {"contrastiv_loss":contrastiv_loss, "regu_loss":regu_loss, "color_loss":color_loss}
    
class simCLRcolor1_adversarial(keras.Model) :
    def __init__(self, backbone, head, color_head, adversarial_net, temp=0.7) :
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.color_head = color_head
        self.adversarial_net = adversarial_net
        self.survey_accuracy = tf.keras.metrics.BinaryAccuracy()
        self.temp=temp

    def call(self, input, training=True) :
        x = self.backbone(input, training=training)
        z = self.head(x, training=training)
        c = self.color_head(x, training=training)
        classif = self.adversarial_net(x, training=training)
        return z, c, classif

    def train_step(self, data) : 
        images, labels = data
        with tf.GradientTape(persistent=True) as tape :
            z, c, classif = self(images)
            #print(images.shape, c.shape, labels["color"].shape, classif.shape, labels["survey"].shape)
            contrastiv_loss = self.loss(z, self.temp)
            color_loss = tf.keras.losses.mean_squared_error(labels["color"], c)

            classif_xent = tf.keras.losses.sparse_categorical_crossentropy(labels["survey"], classif)
            self.survey_accuracy.update_state(labels["survey"], classif)
            backbone_adversarial_loss = -classif_xent

            regu_loss = sum(self.losses)
            total_loss = regu_loss + contrastiv_loss

        gradients = tape.gradient(backbone_adversarial_loss, self.backbone.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.backbone.trainable_variables))

        gradients = tape.gradient(classif_xent, self.adversarial_net.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.adversarial_net.trainable_variables))
  
        gradients = tape.gradient(total_loss, self.backbone.trainable_variables + self.head.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.backbone.trainable_variables + self.head.trainable_variables))

        gradients = tape.gradient(color_loss, self.backbone.trainable_variables + self.color_head.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.backbone.trainable_variables + self.color_head.trainable_variables))

        del tape
        return {"contrastiv_loss":contrastiv_loss, "regu_loss":regu_loss, "color_loss":color_loss, "adversarial_loss":classif_xent, "acc":self.survey_accuracy.result()}

            
class NTXent(keras.losses.Loss) :
    def __init__(self, normalize=False) :
        super().__init__()
        self.large_num = 1e8
        self.normalize = normalize

    def call(self, batch, temperature=1) :
        # sépare les images x des images x'
        hidden1, hidden2 = tf.split(batch, 2, 0)
        batch_size = tf.shape(hidden1)[0]
        if self.normalize :
            hidden1 = tf.nn.l2_normalize(hidden1, axis=1)
            hidden2 = tf.nn.l2_normalize(hidden2, axis=1)
        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = tf.one_hot(tf.range(batch_size), batch_size*2)    # matrice des des labels,    batch_size x 2*batch_size
        masks = tf.one_hot(tf.range(batch_size), batch_size)       # mask de shape     batch x batch

        logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / temperature       ### si normalisé cela aurait été cosine sim => shape batch, batch    distance x x
        logits_aa = logits_aa - masks * self.large_num    ### on rempli la diagonale de très petite valeur car forcément cosine sim entre vecteurs identique = 1
        logits_bb = tf.matmul(hidden2, hidden2_large, transpose_b=True) / temperature
        logits_bb = logits_bb - masks * self.large_num    ###  idem ici ==> donc là on fait distances entre x' x'
        logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature     ### sim x x'
        logits_ba = tf.matmul(hidden2, hidden1_large, transpose_b=True) / temperature     ### sim x' x 

        loss_a = tf.nn.softmax_cross_entropy_with_logits(              ### matrice labels contient info de où sont les paires positives
            labels, tf.concat([logits_ab, logits_aa], 1))              ### en concaténant ab et aa on obtient similarité de a vers toutes les autres images (en ayant mis sa propre correspondance à 0) 
        loss_b = tf.nn.softmax_cross_entropy_with_logits(              ### idem de b vers toutes les images
            labels, tf.concat([logits_ba, logits_bb], 1))
        loss = tf.reduce_mean(loss_a + loss_b)     ### moyenne des 2 et loss

        return loss
    






###### BarlowTwins #######

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
    



###### VICReg #######

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



###### BYOL #######

class BYOL(keras.Model) :
    def __init__(self, backbone_online, backbone_target, head_online, head_target, clas_online, momentum=0.99) :
        super().__init__()
        self.online_backbone = backbone_online
        self.target_backbone = backbone_target
        self.online_head = head_online
        self.target_head = head_target
        self.online_clas = clas_online
        self.momentum = momentum

    def call(self, images) :
        """
        recoit un batch d'images taille 2N, les N premières sont la transformation t1 et les N suivantes sont la transformation t2 des mêmes images
        """
        x = self.online_backbone(images, training=True)
        x = self.online_head(x, training=True)
        x = self.online_clas(x, training=True)

        y = self.target_backbone(images, training=False)
        y = self.target_head(y, training=False)

        y1, y2 = tf.split(y, 2, 0)   # inversion pour comparer les images à une transformation différente
        y = tf.concat([y2, y1], axis=0)

        return x, y
    
    def update_target_weights(self):
        online_weights = self.online_backbone.weights + self.online_head.weights
        target_weights = self.target_backbone.weights + self.target_head.weights

        for online_weight, target_weight in zip(online_weights, target_weights):
            target_weight.assign(self.momentum * target_weight + (1 - self.momentum) * online_weight)

    def train_step(self, data):
        images = data  

        with tf.GradientTape() as tape:

            x, y = self(images)
            loss = self.loss(y, x) 

        gradients = tape.gradient(loss, self.online_backbone.trainable_variables+self.online_head.trainable_variables+self.online_clas.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.online_backbone.trainable_variables+self.online_head.trainable_variables+self.online_clas.trainable_variables))

        self.update_target_weights()

        return {"loss": loss}

class ByolLoss(keras.losses.Loss) :
    def __init__(self) :
        super().__init__()

    def call(self, ytrue, ypred) :
        """
        pas de ytrue car self supervised
        ypred contient les [x, y] pour online et teacher
        """
        x, y = ytrue, ypred
        x = tf.math.l2_normalize(x, axis=-1)
        y = tf.math.l2_normalize(y, axis=-1)
        loss = 2 - 2 * tf.reduce_sum(x*y, axis=-1)
        return tf.reduce_mean(loss)

