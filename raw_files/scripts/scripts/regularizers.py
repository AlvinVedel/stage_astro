import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers



class VarRegularizer(keras.losses.Loss) :
    def __init__(self) :
        super().__init__()
    
    def call(self, batch, colors) :
        hiddens = tf.math.l2_normalize(batch, axis=-1)
        var = tf.math.reduce_variance(hiddens, axis=0)  # variance entre les éléments du batch sur chaque dimension
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
            f = tf.math.l2_normalize(f, axis=-1)
            color_cosine_dist = 1 - tf.matmul(c, c, transpose_b=True) 

            negative_pair_dist = tf.reduce_max(color_cosine_dist)
    
            color_cosine_dist += tf.cast(tf.identity(tf.shape(color_cosine_dist)[0])*10000, dtype=tf.float32)
            positive_pair_dist = tf.reduce_min(color_cosine_dist)

            triplet_loss = tf.maximum(positive_pair_dist - negative_pair_dist + 0.05, 0)
            return tf.reduce_mean(triplet_loss)

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
        c = tf.math.l2_normalize(colors, axis=-1)

        def compute_loss(f, c, use_std, const, spike_factor) :
            
            fcosine_sim_matrix = tf.matmul(f, f, transpose_b=True)

        
            ccosine_dist_matrix = 1 - tf.matmul(c, c, transpose_b=True)   ## matrice des distances dans l'espace couleur
            if use_std :
                const = 2*tf.math.reduce_std(ccosine_dist_matrix)
            else :
               const = const
        
            cosine_mask = tf.cast(ccosine_dist_matrix > const, tf.float32)

            loss_matrix = tf.math.exp(-spike_factor * tf.abs(fcosine_sim_matrix - 1)) * cosine_mask
            return tf.reduce_mean(loss_matrix)

        f1, f2 = tf.split(f, 2, 0)
        c1, c2 = tf.split(c, 2, 0)
        l1 = compute_loss(f1, c1, self.use_std, self.const, self.spike_factor)
        l2 = compute_loss(f2, c2, self.use_std, self.const, self.spike_factor)
        return l1+l2







