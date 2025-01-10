import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers



class VarRegularizer(keras.losses.Loss) :
    def __init__(self) :
        super().__init__()
    
    def call(self, batch) :
        hiddens = tf.math.l2_normalize(batch, axis=-1)
        var = tf.reduce_variance(hiddens, axis=0)  # variance entre les éléments du batch sur chaque dimension
        avg_var = tf.reduce_mean(var)

        return -avg_var

class TripletCosineRegularizer(keras.losses.Loss) :
    def __init__(self) :
        super().__init__()

    def call(self, features, colors):
        f1, f2 = tf.split(features, 2, 0)
        c1, c2 = tf.split(colors, 2, 0)
        
        def compute_triplet_loss(f, c):
            c = tf.math.l2_normalize(c, axis=-1)

            color_cosine_dist = 1 - tf.matmul(c, c, transpose_b=True)

            positive_indexes = tf.argmin(color_cosine_dist, axis=1)
            negative_indexes = tf.argmax(color_cosine_dist, axis=1)

            positive_features = tf.gather(f, positive_indexes, axis=0)
            negative_features = tf.gather(f, negative_indexes, axis=0)

            positive_pair_loss = tf.reduce_mean(1 - tf.reduce_sum(f * positive_features, axis=-1))
            negative_pair_loss = tf.reduce_mean(tf.reduce_sum(f * negative_features, axis=-1))

            return negative_pair_loss - positive_pair_loss

        loss_f1 = compute_triplet_loss(f1, c1)
        loss_f2 = compute_triplet_loss(f2, c2)

        # Somme des pertes
        total_loss = loss_f1 + loss_f2
        return total_loss



class CosineDistRegularizer(keras.losses.Loss) :
    def __init__(self, use_std=True, const=0.05, spike_factor=1) :
        super().__init__()
        self.use_std = use_std
        self.const = const
        self.spike_factor = spike_factor

    def call(self, features, colors) :
        f = tf.math.l2_normalize(features, axis=-1)
        fcosine_sim_matrix = tf.matmul(f, f, transpose_b=True)

        c = tf.math.l2_normalize(colors, axis=-1)
        ccosine_dist_matrix = 1 - tf.matmul(c, c, transpose_b=True)   ## matrice des distances dans l'espace couleur
        if self.use_std :
            const = 2*tf.reduce_std(ccosine_dist_matrix)
        else :
            const = self.const
        
        cosine_mask = tf.cast(ccosine_dist_matrix > const, tf.float32)

        loss_matrix = tf.math.exp(-self.spike_factor * tf.abs(fcosine_sim_matrix - 1)) * cosine_mask
        return tf.reduce_mean(loss_matrix)







