import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import Regularizer



### IMPLEMENTATION DE REGULARIZERS POUR Regularizing Neural Networks via Minimizing Hyperspherical Energy https://openaccess.thecvf.com/content_CVPR_2020/papers/Lin_Regularizing_Neural_Networks_via_Minimizing_Hyperspherical_Energy_CVPR_2020_paper.pdf
### -> pas fonctionnel 

### ET REGULARIZERS MAISON POUR LA CONTRAINTE DE L ESPACE LATENT


class ThomsonRegularizerFirst(Regularizer):
    def __init__(self, lambda_=1e-3, lambda_l2=0):
        self.lambda_ = lambda_
        self.lambda_l2 = lambda_l2

    def __call__(self, w):
        n_filt = w.shape[-1]  # Nombre de filtres dans la couche
        w_reshaped = tf.reshape(w, [-1, n_filt])
        
        filt_neg = w_reshaped * -1
        w_reshaped = tf.concat([w_reshaped, filt_neg], axis=1)
        
        # Calcul des produits internes normalisés entre tous les filtres
        filt_norm = tf.sqrt(tf.reduce_sum(w_reshaped * w_reshaped, axis=0, keepdims=True) + 1e-4)
        norm_mat = tf.matmul(tf.transpose(filt_norm), filt_norm)
        inner_pro = tf.matmul(tf.transpose(w_reshaped), w_reshaped)
        inner_pro /= norm_mat

        cross_terms = 2.0 - 2.0 * inner_pro + tf.linalg.diag(tf.ones([n_filt * 2]))
        final = tf.pow(cross_terms, -1)  # Répulsion entre filtres
        final -= tf.linalg.band_part(final, -1, 0)

        cnt = n_filt * (n_filt - 1) / 2.0
        loss = tf.reduce_sum(final) / cnt

        l2_loss = self.lambda_l2 * tf.reduce_sum(tf.square(w))
        
        return self.lambda_ * loss + l2_loss

    def get_config(self):
        return {"lambda_": self.lambda_}



class ThomsonRegularizerProject(Regularizer):
    def __init__(self, lambda_=1e-3, pd=30, pn=1, pnd=0, lambda_l2=0):
        self.lambda_ = lambda_
        self.pd = pd  # Dimension de projection
        self.pn = pn  # Nombre de vecteurs gaussiens pour la projection
        self.pnd = pnd  # Proportion de la projection appliquée
        self.lambda_l2 = lambda_l2

    def __call__(self, w):
        n_filt = w.shape[-1]  # Nombre de filtres dans la couche
        w_reshaped = tf.reshape(w, [-1, n_filt])
        
        # Création de vecteurs gaussiens aléatoires pour la projection
        random_vectors = tf.random.normal([self.pd, w.shape[0] * w.shape[1]], mean=0.0, stddev=1.0)

        # Application de la projection : projection des poids sur ces vecteurs
        w_reshaped = tf.matmul(random_vectors, w_reshaped)
        
        # Calcul des produits internes (similarité entre filtres)
        filt_norm = tf.sqrt(tf.reduce_sum(w_reshaped * w_reshaped, axis=0, keepdims=True) + 1e-4)
        norm_mat = tf.matmul(tf.transpose(filt_norm), filt_norm)
        inner_pro = tf.matmul(tf.transpose(w_reshaped), w_reshaped)
        inner_pro /= norm_mat
        
        # Calcul de la répulsion entre les filtres
        cross_terms = 2.0 - 2.0 * inner_pro + tf.linalg.diag(tf.ones([n_filt * 2]))
        final = tf.pow(cross_terms, -1)
        final -= tf.linalg.band_part(final, -1, 0)
        
        cnt = n_filt * (n_filt - 1) / 2.0
        loss = tf.reduce_sum(final) / cnt


        l2_loss = self.lambda_l2 * tf.reduce_sum(tf.square(w))
        
        return self.lambda_ * loss + l2_loss

    def get_config(self):
        return {"lambda_": self.lambda_, "pd": self.pd, "pn": self.pn, "pnd": self.pnd, "lambda_l2":self.lambda_l2}



class ThomsonRegularizerFinal(Regularizer):
    def __init__(self, lambda_=1e-3, lambda_l2=0):
        self.lambda_ = lambda_
        self.lambda_l2 = lambda_l2

    def __call__(self, w):
        n_filt = w.shape[-1]  # Nombre de filtres dans la couche
        w_reshaped = tf.reshape(w, [-1, n_filt])

        filt_norm = tf.sqrt(tf.reduce_sum(w_reshaped * w_reshaped, axis=0, keepdims=True) + 1e-4)
        norm_mat = tf.matmul(tf.transpose(filt_norm), filt_norm)
        inner_pro = tf.matmul(tf.transpose(w_reshaped), w_reshaped)
        inner_pro /= norm_mat
        
        cross_terms = 2.0 - 2.0 * inner_pro + tf.linalg.diag(tf.ones([n_filt]))
        final = tf.pow(cross_terms, -1)
        final -= tf.linalg.band_part(final, -1, 0)
        
        cnt = n_filt * (n_filt - 1) / 2.0
        loss = tf.reduce_sum(final) / cnt

        l2_loss = self.lambda_l2 * tf.reduce_sum(tf.square(w))
        
        return 10 * self.lambda_ * loss + l2_loss

    def get_config(self):
        return {"lambda_": self.lambda_, "lambda_l2":self.lambda_l2}
















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







