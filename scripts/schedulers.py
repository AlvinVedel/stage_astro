import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np


class CosineDecay(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr, target_lr, max_epochs, current_epoch=0, decay_factor=1.0):
        super().__init__()
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self.max_epochs = max_epochs
        self.decay_factor = decay_factor
        self.epoch_counter = current_epoch  # Compteur pour le nombre d'époques

    def cosine_lr_schedule(self):
        """Calcul de la décroissance cosinus du learning rate"""
        epoch = self.epoch_counter
        decayed_lr = self.initial_lr + (self.target_lr - self.initial_lr)(0.5 * (1 + np.cos(np.pi * epoch / self.max_epochs)))
        return decayed_lr

    def on_epoch_begin(self, epoch, logs=None):
        """Mise à jour du learning rate au début de chaque époque"""
        self.epoch_counter += 1
        #current_lr = self.model.optimizer.learning_rate - self.cosine_lr_schedule()
        if self.epoch_counter % 25 == 0 :
            current_lr = self.model.optimizer.lr.numpy() / 2
            tf.keras.backend.set_value(self.model.optimizer.lr, current_lr)
            print(f"\nEpoch {self.epoch_counter}: Learning rate is set to {current_lr}")

class LinearDecay(tf.keras.callbacks.Callback):
    def __init__(self, ep, factor=2, each=25) :
        super().__init__()
        self.epoch_counter = ep
        self.factor = factor
        self.each = each

    def on_epoch_begin(self, epoch, logs=None) :
        self.epoch_counter+=1
        if self.epoch_counter % self.each == 0 :
            current_lr = self.model.optimizer.lr.numpy() / self.factor
            tf.keras.backend.set_value(self.model.optimizer.lr, current_lr)
            print(f"\nEpoch {self.epoch_counter}: Learning rate is set to {current_lr}")



class TreyerScheduler(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if epoch==35 or epoch==45 :
            old_lr = self.model.optimizer.lr.numpy()  # On récupère le LR actuel
            new_lr = old_lr / 10  # Diviser le LR par 10
            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
            print(f"\nEpoch {epoch+1}: Learning rate is reduced to {new_lr}")
