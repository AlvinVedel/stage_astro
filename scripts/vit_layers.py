from tensorflow.keras import layers
import tensorflow as tf
import tensorflow.keras as keras
from functools import partial
import math





class Block(tf.keras.Model) :
    def __init__(self, embed_dim, num_heads=8, mlp_ratio=4.0) :
        ## PAS DE LayerScale (ils utilisent identity() si pas de init value, pas de init_value ici)
        super().__init__()
        self.embed_dim=embed_dim
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)



        self.qkv = layers.Dense(self.embed_dim*3, activation='linear', use_bias=False)
        self.proj = layers.Dense(self.embed_dim, activation='linear')

        self.mlp1 = layers.Dense(int(self.embed_dim*self.mlp_ratio), activation='gelu')
        self.mlp2 = layers.Dense(self.embed_dim, activation='linear')

    def call(self, x) :
        # layer norm 1
        x_nrom = self.norm1(x)

        ##### ATTENTION PART #####
        b, n, c = tf.shape(x_nrom)[0], tf.shape(x_nrom)[1], tf.shape(x_nrom)[2]
        qkv = self.qkv(x_nrom)  # shape batch, n, c*3
        qkv = tf.reshape(qkv, (b, n, 3, self.num_heads, c // self.num_heads))   # shape batch, n, 3, 8, c/8    batch, 65, 3, 8, 72
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])    # 3, batch, 8, 65, 72
        head_dim = self.embed_dim // self.num_heads
        q, k, v = qkv[0] * head_dim**-.5, qkv[1], qkv[2]   # séparation q k v chacun étant    batch, num_head, n,  c/num_head  
        # 1, 8, 65, 72

        attn = tf.matmul(q, k, transpose_b=True)  # 1, 8, 65, 65
        attn = tf.nn.softmax(attn, axis=-1)
        y = tf.matmul(attn, v)  # 1, 8, 65, 72
        y = tf.transpose(y, perm=[0, 2, 1, 3])   # 1, 65, 8, 72
        y = tf.reshape(y, (b, n, c))  # shape 1, 65, 576
        y = self.proj(y)  # 1, 65, 576
        x = x + y     # 1, 65, 576      y = résultat du x_norm dans le bloc attention, x pas layer normalisé


        x_norm = self.norm2(x) 
        ##### MLP PART #####   -> pas de drop car régression?
        y = self.mlp1(x_norm)   # 1, 65, 576*mlp_ratio
        y = self.mlp2(y)     # 1, 65, 576
        x = x + y   # 1, 65, 576       y = résultat du x_norm dans bloc attention,     x pas layer normalisé
        
        return x


class PatchExtractor(tf.keras.layers.Layer) :
    def __init__(self, patch_size=4, embed_size=512, image_size=64) :
        super().__init__()
        self.patch_size=patch_size
        self.embed_size = embed_size
        self.patch_conv = tf.keras.layers.Conv2D(filters=self.embed_size, kernel_size=(self.patch_size, self.patch_size), strides=(self.patch_size, self.patch_size), padding='valid', activation='linear')
        self.num_patches = (image_size // patch_size) ** 2
        self.positional_embedding = self.add_weight(
                "positional_embedding",
                shape=(self.num_patches, self.embed_size),
                initializer="random_normal",
                trainable=True
            )

    def call(self, inputs) :
        res_conv = self.patch_conv(inputs)  # shape batch, H, W, filtres
        patch_embedding = tf.reshape(res_conv, (tf.shape(res_conv)[0], -1, tf.shape(res_conv)[-1]))  # shape BATCH, N_PATCH, EMBED_DIM
        patch_embedding += self.positional_embedding
        return patch_embedding
    

class ViT_backbone(tf.keras.layers.Layer) :
    def __init__(self, embed_dim=256, num_blocks=4, num_heads=8, patch_size=4, gp='average') :
        super().__init__()
        self.embed_dim=embed_dim
        self.patch_master = PatchExtractor(patch_size, embed_size=self.embed_dim, image_size=64)
        self.blocks = [Block(embed_dim=self.embed_dim, num_heads=num_heads) for i in range(num_blocks)]
        if gp == "average" :
            self.last_pool = layers.GlobalAveragePooling1D()
        else :
            self.last_pool = layers.GlobalMaxPooling1D()

    def call(self, inputs) :
        x = self.patch_master(inputs) 
        for i in range(len(self.blocks)) :
            x = self.blocks[i](x)
        
        return self.last_pool(x)   # sortie B, N



class Backbone(tf.keras.Model) :
    def __init__(self, img_size=64, patch_size=4, embed_dim=576, num_blocks=6) :
        super().__init__()
        self.num_patch = (img_size//patch_size)**2
        self.patch_size=patch_size
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.interpolate_antialias = False
        self.interpolate_offset = 0.1



        self.patch_embed = layers.Conv2D(filters=embed_dim, kernel_size=(patch_size, patch_size), strides=(patch_size, patch_size), padding='same')
        self.flatten = layers.Reshape(target_shape=(-1, self.embed_dim))


        self.cls_token = self.add_weight(shape=(1, 1, self.embed_dim), initializer='zeros', trainable=True)
        self.concat = layers.Concatenate(axis=1)

        self.pos_embed = self.add_weight(shape=(1, self.num_patch+1, embed_dim), trainable=True, initializer='zeros')
        #self.add = layers.Add()

        self.blocks = [Block(self.embed_dim) for _ in range(self.num_blocks)]
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)

        self.mask_token = self.add_weight(shape=(1, embed_dim), initializer='zeros', trainable=True)

        

    def call(self, x, masks=None):

        x = tf.cast(x, tf.float32)
        b, w, h, nc = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x = self.patch_embed(x) # batch, 8, 8, 576
        x = self.flatten(x)  # batch, 64, 576

        if masks is not None :
            # masks censé être de shape batch, N      x de shape batch, N, embed_dim
            masks = tf.expand_dims(masks, axis=-1)  # shape batch, N, 1
            mask_token = tf.cast(self.mask_token, dtype=x.dtype)
            print(masks.shape, mask_token.shape, x.shape)
            x = tf.where(masks, mask_token, x)

        #if masks is not None :
        #    x = tf.where(masks, self.mask_token, x)

        x = self.concat([tf.tile(self.cls_token, (b, 1, 1)), x]) # batch, 65, 576

        # pos embed interpolation
        previous_dtype = x.dtype
        npatch = tf.shape(x)[1] - 1
        """
        N = self.pos_embed.shape[1] - 1
        if N == npatch and w==h:
            x = x + tf.tile(self.pos_embed, (b, 1, 1)) # batch, 65, 576

        else :
            pos_embed = tf.cast(self.pos_embed, dtype=tf.float32)
            class_pos_embed = pos_embed[:, 0]
            patch_pos_embed = pos_embed[:, 1:]
            dim = tf.shape(x)[-1]
            w0 = w // self.patch_size
            h0 = h // self.patch_size
            M = int(tf.sqrt(tf.cast(N, tf.float32)))  # Nombre de patches dans chaque dimension

            
            size = (w0, h0)

            # Redimensionner les patches
            patch_pos_embed = tf.reshape(patch_pos_embed, (1, M, M, dim))
            patch_pos_embed = tf.image.resize(
                patch_pos_embed,
                size,
                method='bicubic' if self.interpolate_antialias else 'bilinear',
                antialias=self.interpolate_antialias
            )

            patch_pos_embed = tf.reshape(patch_pos_embed, (1, -1, dim))
            result = tf.concat([tf.expand_dims(class_pos_embed, 0), patch_pos_embed], axis=1)
            x = x + tf.tile(tf.cast(result, dtype=previous_dtype), (b, 1, 1))
        """



        N = self.pos_embed.shape[1] - 1
        condition = tf.logical_and(tf.equal(N, npatch), tf.equal(w, h))

        def pos_embed_match():
            return x + tf.tile(self.pos_embed, (b, 1, 1))  # Cas où N == npatch et w == h

        def pos_embed_resize():
            pos_embed = tf.cast(self.pos_embed, dtype=tf.float32)
            class_pos_embed = pos_embed[:, 0]
            patch_pos_embed = pos_embed[:, 1:]
            dim = tf.shape(x)[-1]
            w0 = w // self.patch_size
            h0 = h // self.patch_size
            M = tf.cast((tf.sqrt(tf.cast(N, tf.float32))), tf.int32)  # Nombre de patches dans chaque dimension
            size = (w0, h0)

            # Redimensionner les patches
            patch_pos_embed = tf.reshape(patch_pos_embed, (1, M, M, dim))
            patch_pos_embed = tf.image.resize(
                patch_pos_embed,
                size,
                method='bicubic' if self.interpolate_antialias else 'bilinear',
                antialias=self.interpolate_antialias
            )

            patch_pos_embed = tf.reshape(patch_pos_embed, (1, -1, dim))
            result = tf.concat([tf.expand_dims(class_pos_embed, 0), patch_pos_embed], axis=1)
            return x + tf.tile(tf.cast(result, dtype=previous_dtype), (b, 1, 1))

        x = tf.cond(condition, pos_embed_match, pos_embed_resize)


        for blk in self.blocks :
            x = blk(x)
        
        x = self.layer_norm(x)

        return {'cls_token' : x[:, 0],
                'patch_token' : x[:, 1:]}
    


class BackboneAstro(tf.keras.Model) :
    def __init__(self, img_size=64, patch_size=4, embed_dim=576, num_blocks=6) :
        super().__init__()
        self.num_patch = (img_size//patch_size)**2
        self.patch_size=patch_size
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.interpolate_antialias = False
        self.interpolate_offset = 0.1

        self.conv1 = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='tanh')



        self.patch_embed = layers.Conv2D(filters=embed_dim, kernel_size=(patch_size, patch_size), strides=(patch_size, patch_size), padding='same')
        self.flatten = layers.Reshape(target_shape=(-1, self.embed_dim))


        self.cls_token = self.add_weight(shape=(1, 1, self.embed_dim), initializer='zeros', trainable=True)
        self.concat = layers.Concatenate(axis=1)

        self.pos_embed = self.add_weight(shape=(1, self.num_patch+1, embed_dim), trainable=True, initializer='zeros')
        #self.add = layers.Add()

        self.blocks = [Block(self.embed_dim) for _ in range(self.num_blocks)]
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)

        self.mask_token = self.add_weight(shape=(1, embed_dim), initializer='zeros', trainable=True)

        

    def call(self, x, masks=None):

        x = tf.cast(x, tf.float32)
        b, w, h, nc = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.patch_embed(x) # batch, 8, 8, 576
        x = self.flatten(x)  # batch, 64, 576

        if masks is not None :
            # masks censé être de shape batch, N      x de shape batch, N, embed_dim
            masks = tf.expand_dims(masks, axis=-1)  # shape batch, N, 1
            mask_token = tf.cast(self.mask_token, dtype=x.dtype)
            print(masks.shape, mask_token.shape, x.shape)
            x = tf.where(masks, mask_token, x)

        #if masks is not None :
        #    x = tf.where(masks, self.mask_token, x)

        x = self.concat([tf.tile(self.cls_token, (b, 1, 1)), x]) # batch, 65, 576

        # pos embed interpolation
        previous_dtype = x.dtype
        npatch = tf.shape(x)[1] - 1
        N = self.pos_embed.shape[1] - 1
        if N == npatch and w==h:
            x = x + tf.tile(self.pos_embed, (b, 1, 1)) # batch, 65, 576

        else :
            pos_embed = tf.cast(self.pos_embed, dtype=tf.float32)
            class_pos_embed = pos_embed[:, 0]
            patch_pos_embed = pos_embed[:, 1:]
            dim = tf.shape(x)[-1]
            w0 = w // self.patch_size
            h0 = h // self.patch_size
            M = int(tf.sqrt(tf.cast(N, tf.float32)))  # Nombre de patches dans chaque dimension

            
            size = (w0, h0)

            # Redimensionner les patches
            patch_pos_embed = tf.reshape(patch_pos_embed, (1, M, M, dim))
            patch_pos_embed = tf.image.resize(
                patch_pos_embed,
                size,
                method='bicubic' if self.interpolate_antialias else 'bilinear',
                antialias=self.interpolate_antialias
            )

            patch_pos_embed = tf.reshape(patch_pos_embed, (1, -1, dim))
            result = tf.concat([tf.expand_dims(class_pos_embed, 0), patch_pos_embed], axis=1)
            x = x + tf.tile(tf.cast(result, dtype=previous_dtype), (b, 1, 1))


        for blk in self.blocks :
            x = blk(x)
        
        x = self.layer_norm(x)

        return {'cls_token' : x[:, 0],
                'patch_token' : x[:, 1:]}



import tensorflow as tf
from tensorflow.keras import layers

class Head(tf.keras.Model):
    def __init__(
        self,
        in_dim,
        out_dim,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
    ):
        super(Head, self).__init__()
        
        # Construire le MLP
        nlayers = max(nlayers, 1)
        self.mlp = self._build_mlp(nlayers, bottleneck_dim, hidden_dim=hidden_dim, bias=mlp_bias)
        self._init_weights()
        self.last_layer = layers.Dense(out_dim, use_bias=False)
        #self.g = self.add_weight(shape=(1,), initializer="ones", trainable=True, name="weight_g")

    def _init_weights(self):
        """Initialisation des poids des couches linéaires"""
        for layer in self.mlp:
            if isinstance(layer, layers.Dense):
                # Equivalent à trunc_normal_ dans PyTorch
                initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02)
                layer.kernel_initializer = initializer
                if layer.use_bias :#is not None:
                    layer.bias_initializer = tf.keras.initializers.Zeros()

    def _build_mlp(self, nlayers, bottleneck_dim, hidden_dim=None, bias=True):
        layers_list = []
        if nlayers == 1:
            # Une seule couche linéaire
            layers_list.append(layers.Dense(bottleneck_dim, use_bias=bias))
        else:
            # Première couche
            layers_list.append(layers.Dense(hidden_dim, use_bias=bias))
            layers_list.append(layers.Activation('gelu'))

            # Couches intermédiaires
            for _ in range(nlayers - 2):
                layers_list.append(layers.Dense(hidden_dim, use_bias=bias))
                layers_list.append(layers.Activation('gelu'))

            # Dernière couche (linéaire)
            layers_list.append(layers.Dense(bottleneck_dim, use_bias=bias))
        return layers_list

    def call(self, x, training=False):
        
        # Passer à travers le MLP
        for layer in self.mlp:
            x = layer(x, training=training)

        # Normalisation L2 (norme sur les vecteurs)
        eps = 1e-6 if x.dtype == tf.float16 else 1e-12
        x = tf.nn.l2_normalize(x, axis=-1, epsilon=eps)
        # Appliquer la normalisation des poids
        #weights = tf.nn.l2_normalize(self.last_layer.kernel, axis=0) * self.g
        x = self.last_layer(x)
        return x







