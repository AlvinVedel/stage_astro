import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
#import tensorflow_probability as tfp


def compute_median(values):
    # Trier les valeurs
    sorted_values = tf.sort(values)
    n = tf.shape(sorted_values)[0]

    # Vérifier si le nombre d'éléments est pair ou impair
    is_odd = tf.math.floormod(n, 2) == 1

    # Si impair, prendre l'élément du milieu
    median = tf.cond(
        is_odd,
        lambda: tf.gather(sorted_values, n // 2),
        lambda: (tf.gather(sorted_values, n // 2 - 1) + tf.gather(sorted_values, n // 2)) / 2.0
    )
    return median


class Bias(tf.keras.metrics.Metric):
    def __init__(self, inf=0, sup=1e+6, name="bias", **kwargs):
        super(Bias, self).__init__(name=name, **kwargs)
        self.borne_inf = inf
        self.borne_sup = sup
        self.total_bias = self.add_weight(name="total_bias", initializer="zeros")
        self.total_samples = self.add_weight(name="total_samples", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Filtrage des valeurs dans la plage [inf, sup[
        mask = tf.logical_and(y_true >= self.borne_inf, y_true < self.borne_sup)
        y_true_filtered = tf.boolean_mask(y_true, mask)
        y_pred_filtered = tf.boolean_mask(y_pred, mask)

        # Vérifier s'il y a des valeurs dans la plage
        if tf.reduce_sum(tf.cast(mask, tf.float32)) > 0:
            deltas_z = (y_pred_filtered - y_true_filtered) / (1 + y_true_filtered)
            bias = tf.reduce_sum(deltas_z)  # Somme des biais pour les échantillons filtrés
            self.total_bias.assign_add(bias)
            self.total_samples.assign_add(tf.cast(tf.size(deltas_z), tf.float32))

    def result(self):
        # Fraction cumulée des outliers
        return self.total_bias / (self.total_samples + 1e-6)

    def reset_states(self):
        # Réinitialisation des états
        self.total_bias.assign(0.0)
        self.total_samples.assign(0.0)



class SigmaMAD(tf.keras.metrics.Metric):
    def __init__(self, inf=0, sup=1e+6, name="smad", **kwargs):
        super(SigmaMAD, self).__init__(name=name, **kwargs)
        self.borne_inf = inf
        self.borne_sup = sup
        self.total_smad = self.add_weight(name="total_smad", initializer="zeros")
        self.total_samples = self.add_weight(name="total_samples", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Filtrage des valeurs dans la plage [inf, sup[
        mask = tf.logical_and(y_true >= self.borne_inf, y_true < self.borne_sup)
        y_true_filtered = tf.boolean_mask(y_true, mask)
        y_pred_filtered = tf.boolean_mask(y_pred, mask)

        # Vérifier s'il y a des valeurs dans la plage
        if tf.reduce_sum(tf.cast(mask, tf.float32)) > 0:
            deltas_z = (y_pred_filtered - y_true_filtered) / (1 + y_true_filtered)
            #median_delta_z_norm = tfp.stats.percentile(deltas_z, 50.0)  # Médiane
            median_delta_z_norm = compute_median(deltas_z)
            #mad = tfp.stats.percentile(tf.abs(deltas_z - median_delta_z_norm), 50.0)
            mad = compute_median(tf.abs(deltas_z - median_delta_z_norm))
            smad = 1.4826 * mad

            # Mise à jour des métriques
            self.total_smad.assign_add(smad)
            self.total_samples.assign_add(tf.cast(tf.size(deltas_z), tf.float32))

    def result(self):
        # Fraction cumulée des outliers
        return self.total_smad / (self.total_samples + 1e-6)

    def reset_states(self):
        # Réinitialisation des états
        self.total_smad.assign(0.0)
        self.total_samples.assign(0.0)
        



class OutlierFraction(tf.keras.metrics.Metric):
    def __init__(self, inf=0, sup=1e+6, name="outl", **kwargs):
        super(OutlierFraction, self).__init__(name=name, **kwargs)
        self.borne_inf = inf
        self.borne_sup = sup
        self.total_outliers = self.add_weight(name="total_outliers", initializer="zeros")
        self.total_samples = self.add_weight(name="total_samples", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Filtrage des valeurs dans la plage [inf, sup[
        mask = tf.logical_and(y_true >= self.borne_inf, y_true < self.borne_sup)
        y_true_filtered = tf.boolean_mask(y_true, mask)
        y_pred_filtered = tf.boolean_mask(y_pred, mask)

        # Vérifier s'il y a des valeurs dans la plage
        if tf.reduce_sum(tf.cast(mask, tf.float32)) > 0:
            deltas_z = (y_pred_filtered - y_true_filtered) / (1 + y_true_filtered)
            outliers = tf.abs(deltas_z) > 0.05

            # Mise à jour des métriques
            self.total_outliers.assign_add(tf.reduce_sum(tf.cast(outliers, tf.float32)))
            self.total_samples.assign_add(tf.cast(tf.size(deltas_z), tf.float32))

    def result(self):
        # Fraction cumulée des outliers
        return self.total_outliers / (self.total_samples + 1e-6)

    def reset_states(self):
        # Réinitialisation des états
        self.total_outliers.assign(0.0)
        self.total_samples.assign(0.0)




        

