import tensorflow as tf
import tensorflow.keras as keras

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
        images = data  # Assure-toi que data contient tes images

        with tf.GradientTape() as tape:

            x, y = self(images)
            loss = self.loss(y, x) 

        # Calculer les gradients et appliquer l'optimisation
        gradients = tape.gradient(loss, self.online_backbone.trainable_variables+self.online_head.trainable_variables+self.online_clas.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.online_backbone.trainable_variables+self.online_head.trainable_variables+self.online_clas.trainable_variables))

        # Mettre à jour les poids du modèle target
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
