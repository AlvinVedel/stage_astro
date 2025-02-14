import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from vit_layers import ViT_backbone
from contrastiv_model import NTXent
from regularizers import ThomsonRegularizerFinal, ThomsonRegularizerFirst, ThomsonRegularizerProject



def inception_block(input):
    c1 = layers.Conv2D(101, activation='relu', kernel_size=1, strides=1, padding='same')(input)
    c2 = layers.Conv2D(101, activation='relu', kernel_size=1, strides=1, padding='same')(input)
    c3 = layers.Conv2D(101, activation='relu', kernel_size=1, strides=1, padding='same')(input)

    c4 = layers.Conv2D(156, activation='relu', kernel_size=3, strides=1, padding='same')(c1)
    c5 = layers.Conv2D(156, activation='relu', kernel_size=5, strides=1, padding='same')(c2)
    c6 = layers.AveragePooling2D((2, 2), strides=1, padding='same')(c3)

    c7 = layers.Conv2D(109, activation='relu', kernel_size=1, strides=1, padding='same')(input)

    conc = layers.Concatenate(axis=-1)([c4, c5, c6, c7])   # 156 + 156 + 101 + 109 = 522
    return conc


def basic_backbone(full_bn=True) :
    inp = keras.Input((64, 64, 6))
    c1 = layers.Conv2D(96, padding='same', strides=1, kernel_size=3)(inp) # 64
    c1 = layers.PReLU()(c1) 
    c2 = layers.Conv2D(96, padding='same', kernel_size=3, strides=1, activation='tanh')(c1)  #64
    if full_bn :
        c2 = layers.BatchNormalization()(c2)
    p1 = layers.AveragePooling2D((2, 2))(c2)  # 32
    c3 = layers.Conv2D(128, padding='same', strides=1, kernel_size=3)(p1)
    c3 = layers.PReLU()(c3)
    if full_bn :
        c3 = layers.BatchNormalization()(c3)
    c4 = layers.Conv2D(128, padding='same', kernel_size=3, strides=1)(c3)  #32
    c4 = layers.PReLU(name='c4')(c4) 
    if full_bn :
        c4 = layers.BatchNormalization()(c4)
    p2 = layers.AveragePooling2D((2, 2))(c4)  # 16
    c5 = layers.Conv2D(256, padding='same', strides=1, kernel_size=3)(p2) #16
    c5 = layers.PReLU()(c5)
    if full_bn :
        c5 = layers.BatchNormalization()(c5)
    c6 = layers.Conv2D(256, padding='same', kernel_size=3, strides=1)(c5)  #16
    c6 = layers.PReLU()(c6)
    if full_bn :
       c6 = layers.BatchNormalization()(c6)
    p3 = layers.AveragePooling2D((2, 2))(c6) # 8
    c7 = layers.Conv2D(256, kernel_size=3, strides=1, padding='valid')(p3) # 6
    c7 = layers.PReLU()(c7)
    if full_bn :
        c7 = layers.BatchNormalization()(c7)
    c8 = layers.Conv2D(256, kernel_size=3, strides=1, padding='valid')(c7) # 4
    c8 = layers.PReLU()(c8)
    if full_bn :
        c8 = layers.BatchNormalization()(c8)
    c9 = layers.Conv2D(256, padding='valid', kernel_size=3, strides=1)(c8) # 2, 2, 256
    c9 = layers.PReLU()(c9)
    if full_bn :
        c9 = layers.BatchNormalization()(c9)
    flat = layers.Flatten(name='flatten')(c9) # 2, 2, 256 = 1024 

    l1 = layers.Dense(1024)(flat) 
    l1 = layers.PReLU()(l1)
    if full_bn :
        l1 = layers.BatchNormalization()(l1)
   
    return keras.Model(inputs=inp, outputs=l1)




def comhe_backbone(l2_value=1e-5) :
    inp = keras.Input((64, 64, 6))
    c1 = layers.Conv2D(96, padding='same', strides=1, kernel_size=3, kernel_regularizer=ThomsonRegularizerFirst(lambda_=1e-3, lambda_l2=l2_value), bias_regularizer=keras.regularizers.L2(l2_value))(inp) # 64
    c1 = layers.PReLU()(c1) 
    c2 = layers.Conv2D(96, padding='same', kernel_size=3, strides=1, activation='tanh', kernel_regularizer=ThomsonRegularizerProject(lambda_=1e-3, lambda_l2=l2_value), bias_regularizer=keras.regularizers.L2(l2_value))(c1)  #64
    p1 = layers.AveragePooling2D((2, 2))(c2)  # 32
    c3 = layers.Conv2D(128, padding='same', strides=1, kernel_size=3, kernel_regularizer=ThomsonRegularizerProject(lambda_=1e-3, lambda_l2=l2_value), bias_regularizer=keras.regularizers.L2(l2_value))(p1)
    c3 = layers.PReLU()(c3)
    c4 = layers.Conv2D(128, padding='same', kernel_size=3, strides=1, kernel_regularizer=ThomsonRegularizerProject(lambda_=1e-3, lambda_l2=l2_value), bias_regularizer=keras.regularizers.L2(l2_value))(c3)  #32
    c4 = layers.PReLU(name='c4')(c4) 
    p2 = layers.AveragePooling2D((2, 2))(c4)  # 16
    c5 = layers.Conv2D(256, padding='same', strides=1, kernel_size=3, kernel_regularizer=ThomsonRegularizerProject(lambda_=1e-3, lambda_l2=l2_value), bias_regularizer=keras.regularizers.L2(l2_value))(p2) #16
    c5 = layers.PReLU()(c5)
    c6 = layers.Conv2D(256, padding='same', kernel_size=3, strides=1, kernel_regularizer=ThomsonRegularizerProject(lambda_=1e-3, lambda_l2=l2_value), bias_regularizer=keras.regularizers.L2(l2_value))(c5)  #16
    c6 = layers.PReLU()(c6)
    p3 = layers.AveragePooling2D((2, 2))(c6) # 8
    c7 = layers.Conv2D(256, kernel_size=3, strides=1, padding='valid', kernel_regularizer=ThomsonRegularizerProject(lambda_=1e-3, lambda_l2=l2_value), bias_regularizer=keras.regularizers.L2(l2_value))(p3) # 6
    c7 = layers.PReLU()(c7)
    c8 = layers.Conv2D(256, kernel_size=3, strides=1, padding='valid', kernel_regularizer=ThomsonRegularizerProject(lambda_=1e-3, lambda_l2=l2_value), bias_regularizer=keras.regularizers.L2(l2_value))(c7) # 4
    c8 = layers.PReLU()(c8)
    c9 = layers.Conv2D(256, padding='valid', kernel_size=3, strides=1, kernel_regularizer=ThomsonRegularizerFinal(lambda_=1e-3, lambda_l2=l2_value), bias_regularizer=keras.regularizers.L2(l2_value))(c8) # 2, 2, 256
    c9 = layers.PReLU()(c9)
    flat = layers.Flatten(name='flatten')(c9) # 2, 2, 256 = 1024 

    l1 = layers.Dense(1024, kernel_regularizer=keras.regularizers.L2(l2_value), bias_regularizers=keras.regularizers.L2(l2_value))(flat) 
    l1 = layers.PReLU()(l1)
   
    return keras.Model(inputs=inp, outputs=l1)


def comhe_projection_head(input_shape, l2_value=1e-5) :
    latent_input = keras.Input((input_shape))
    x = layers.Dense(512, activation='linear', kernel_regularizer=keras.regularizers.L2(l2_value), bias_regularizers=keras.regularizers.L2(l2_value))(latent_input)
    x = layers.PReLU()(x)
    x = layers.Dense(256, activation='linear', kernel_regularizer=keras.regularizers.L2(l2_value), bias_regularizers=keras.regularizers.L2(l2_value))(x)
    return keras.Model(latent_input, x)


def treyer_backbone(bn=False) :

    input_img = keras.Input((64, 64, 6))
    #input_ebv = keras.Input((1,))
    conv1 = layers.Conv2D(96, kernel_size=5, activation='relu', strides=1, padding='same', name='c1')(input_img)
    conv2 = layers.Conv2D(96, kernel_size=3, activation='tanh', strides=1, padding='same', name='c2')(conv1)
    avg_pool = layers.AveragePooling2D(pool_size=(2, 2), strides=2, name='p1')(conv2)  # batch, 32, 32, 96

    incep1 = inception_block(avg_pool)
    incep2 = inception_block(incep1)
    incep3 = inception_block(incep2)

    avg_pool2 = layers.AveragePooling2D((2, 2), strides=2, name='p2')(incep3) # 16, 16, 522

    incep4 = inception_block(avg_pool2)
    incep5 = inception_block(incep4)

    avg_pool3 = layers.AveragePooling2D((2, 2), strides=2, name='p3')(incep5)  # 8, 8, 522

    incep6 = inception_block(avg_pool3) 

    conv3 = layers.Conv2D(96, kernel_size=3, activation='relu', padding='valid', strides=1, name='c3')(incep6)  # 6, 6, 96
    conv4 = layers.Conv2D(96, kernel_size=3, activation='relu', padding='valid', strides=1, name='c4')(conv3)  # 4, 4, 96
    conv5 = layers.Conv2D(96, kernel_size=3, activation='relu', padding='valid', strides=1, name='c5')(conv4)   # 2, 2, 96
    avg_pool4 = layers.AveragePooling2D((2, 2), strides=1, name='p4')(conv5)   # 1, 1, 96
    resh = layers.Reshape(target_shape=(96,))(avg_pool4) # batch, 96
    
    #conc = layers.Concatenate()([resh, input_ebv])
    l1 = layers.Dense(1024, activation='relu', name='l1')(resh)
    if bn :
        l1 = layers.BatchNormalization()(l1)

    
    model = keras.Model(inputs=[input_img], outputs=[l1])
    return model


def projection_mlp(input_shape=1024, bn=True):
    latent_input = keras.Input((input_shape))
    x = layers.Dense(512, activation='linear')(latent_input)
    if bn :
        x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)
    x = layers.Dense(256, activation='linear', activity_regularizer=tf.keras.regularizers.L1L2(l1=1e-3, l2=1e-2))(x)
    return keras.Model(latent_input, x)

def noregu_projection_mlp(input_shape=1024, bn=True):
    latent_input = keras.Input((input_shape))
    x = layers.Dense(512, activation='linear')(latent_input)
    if bn :
        x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)
    x = layers.Dense(256, activation='linear')(x)
    return keras.Model(latent_input, x)


def color_mlp(input_shape=1024) :
    latent_input = keras.Input((input_shape))
    x = layers.Dense(256)(latent_input)
    x = layers.PReLU()(x)
    x = layers.Dense(256)(x)
    x = layers.PReLU()(x)
    output = layers.Dense(5, activation='linear')(x)
    return keras.Model(latent_input, output)

def classif_mlp(input_shape=1024) :
    latent_input = keras.Input((input_shape))
    x = layers.Dense(256, activation='relu')(latent_input)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    output = layers.Dense(2, activation='softmax')(x)
    return keras.Model(latent_input, output)



def astro_head(input_shape=1024, nbins=400) :
    inp = keras.Input((input_shape))
    l1 = layers.Dense(1024)(inp)
    l1 = layers.PReLU()(l1)

    l2 = layers.Dense(1024)(l1)
    l2 = layers.PReLU()(l2)
    pdf = layers.Dense(nbins, activation='softmax', name='pdf')(l2)

    l2b = layers.Dense(512, activation='tanh')(l1)
    reg = layers.Dense(1, activation='linear', name='reg')(l2b)
    return keras.Model(inp, [pdf, reg])


def astro_model(back, head) :
    inp = keras.Input((64, 64, 6))
    x = back(inp)
    pdf, reg = head(x)
    #return keras.Model(inp, [pdf, reg])
    return keras.Model(inp, {"pdf":pdf, "reg":reg})


def adv_network() :
    inp = keras.Input((32, 32, 128))
    c4 = layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(inp)
    c4 = layers.PReLU()(c4)
    c5 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(c4)
    c5 = layers.PReLU()(c5)
    gap = layers.GlobalAveragePooling2D()(c5)
    fc1 = layers.Dense(128)(gap)
    fc1 = layers.PReLU()(fc1)
    drop = layers.Dropout(0.4)(fc1)
    fc2 = layers.Dense(128)(drop)
    fc2 = layers.PReLU()(fc2)
    drop2 = layers.Dropout(0.4)(fc2)
    classif = layers.Dense(2, activation='softmax')(drop2)
    return keras.Model(inp, classif)



class AstroFinetune(tf.keras.Model):
    def __init__(self, back, head, train_back=False) :
        super().__init__()
        self.back = back
        self.head = head
        self.train_back=train_back

    def call(self, inputs, training=True) :
        x = self.back(inputs)
        pdf, reg = self.head(x, training=training)
        return {"pdf":pdf, "reg":reg}

    def train_step(self, inputs) :
        images, labels = inputs
        with tf.GradientTape() as tape :
            pred_dict = self(images, training=True)
            classif_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels["pdf"], pred_dict["pdf"]))
            reg_loss = tf.reduce_mean(tf.keras.losses.mean_absolute_error(labels["reg"], pred_dict["reg"]))
            total_loss = classif_loss + reg_loss
        if self.train_back :
            trainable_vars = self.back.trainable_variables + self.head.trainable_variables
        else :
            trainable_vars = self.head.trainable_variables

        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return {"crossent":classif_loss, "mae":reg_loss}
    



class ContrastivAstroFinetune(tf.keras.Model):
    def __init__(self, back, head, projection_head, train_back=False) :
        super().__init__()
        self.back = back
        self.head = head
        self.projection_head = projection_head
        self.train_back=train_back
        self.contrast_loss = NTXent(normalize=True)


    def call(self, inputs, training=True) :
        x = self.back(inputs, training=self.train_back)
        pdf, reg = self.head(x, training=training)
        proj = self.projection_head(x, training=training)
        return {"pdf":pdf, "reg":reg, "projection":proj}

    def train_step(self, inputs) :
        images, labels = inputs
        with tf.GradientTape() as tape :
            pred_dict = self(images, training=True)
            classif_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels["pdf"], pred_dict["pdf"]))
            reg_loss = tf.reduce_mean(tf.keras.losses.mean_absolute_error(labels["reg"], pred_dict["reg"]))
            total_loss = classif_loss + reg_loss
            if self.train_back :   # sert à rien de maintenir tête de contraste sur représentations figées
                contrastiv_loss = self.contrast_loss(pred_dict["projection"])
                total_loss += contrastiv_loss
            
        if self.train_back :
            trainable_vars = self.back.trainable_variables + self.head.trainable_variables + self.projection_head.trainable_variables
        else :
            trainable_vars = self.head.trainable_variables

        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return {"crossent":classif_loss, "mae":reg_loss}



class AstroModel(tf.keras.Model) :
    def __init__(self, back, head, is_adv, adv_network) :
        super().__init__()
        self.back = back
        self.head = head 
        self.adv = is_adv
        self.survey_accuracy = tf.keras.metrics.BinaryAccuracy()
        self.adv_network = adv_network
        if self.adv :
            self.back = tf.keras.Model(self.back.input, [self.back.layers[6].output, self.back.output])
        for i, lay in enumerate(self.back.layers) :
            print(i, lay.name)

    def call(self, inputs, only_adv=False, training=True) :
        output_dict = {}

        if self.adv :
            c3, x = self.back(inputs, training=training)
            adv_pred = self.adv_network(c3)
            output_dict["adversarial"] = adv_pred


        else :
            x = self.back(inputs, training=training)

        if only_adv :  
            return output_dict
        else :
            pdf, reg = self.head(x, training=training)
            output_dict["pdf"] = pdf
            output_dict["reg"] = reg
            return output_dict



    def train_step(self, inputs) :
        loss_dict = {}

        x, y = inputs
        if self.adv : 
            z_imgs, adv_imgs = x
        else :
            z_imgs = x
        
        batch_size = tf.shape(z_imgs)[0]        

        with tf.GradientTape(persistent=True) as tape :

            out1 = self(z_imgs, only_adv=False, training=True)

            pdf_loss = tf.keras.losses.sparse_categorical_crossentropy(y["pdf"], out1["pdf"])
            reg_loss = tf.keras.losses.mean_absolute_error(y["reg"], out1["reg"])

            z_loss = pdf_loss+reg_loss
            loss_dict["pdf_loss"] = pdf_loss
            loss_dict["reg_loss"] = reg_loss


            if self.adv : 
                out2 = self(adv_imgs, only_adv=True, training=True)

                classif_true_labels = tf.cast(tf.concat([tf.ones(batch_size), tf.zeros(batch_size)], axis=0), dtype=tf.int32)
                predictions = tf.concat([out1["adversarial"], out2["adversarial"]], axis=0)

                adv_loss = tf.keras.losses.sparse_categorical_crossentropy(classif_true_labels, predictions, from_logits=False)

                self.survey_accuracy.update_state(tf.cast(classif_true_labels, dtype=tf.float32), tf.cast(predictions[:, 1], dtype=tf.float32))

                #adv_loss = tf.keras.losses.sparse_categorical_crossentropy(classif_true_labels, predictions)
                inverse_adv_loss = -adv_loss
                loss_dict["adversarial_loss"] = adv_loss
                loss_dict["adversarial_accuracy"] = self.survey_accuracy.result()

        
        gradients = tape.gradient(z_loss, self.back.trainable_variables+self.head.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.back.trainable_variables+self.head.trainable_variables))

        if self.adv :
            gradients = tape.gradient(adv_loss, self.adv_network.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.adv_network.trainable_variables))

            gradients = tape.gradient(inverse_adv_loss, self.back.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.back.trainable_variables))

        return loss_dict

        


        
            




def segmentor(input_shape1=1024, input_shape2=(32, 32, 128)) :

    inp = keras.Input(input_shape1)
    deflat = layers.Reshape((8, 8, 16))(inp)
    c1 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(deflat)  # 8 8 256
    c1 = layers.PReLU()(c1)
    c1_r = layers.Reshape((16, 16, 64))(c1)  # 16 16 64
    c2 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(c1_r)
    c2 = layers.PReLU()(c2)
    c2_r = layers.Reshape((32, 32, 64))(c2)  #32 32 64
    c3 = layers.Conv2D(128, strides=1, kernel_size=3, padding='same')(c2_r)
    c3 = layers.PReLU()(c3)
    inp2 = keras.Input(input_shape2)
    conc = layers.Concatenate()([c3, inp2])
    c4 = layers.Conv2D(256, kernel_size=3, padding='same', strides=1)(conc)  # 32 32 256
    c4 = layers.PReLU()(c4)
    c4_r = layers.Reshape((64, 64, 64))(c4)
    segmentation = layers.Conv2D(1, kernel_size=3, padding='same', strides=1, activation='sigmoid')(c4_r)
    return keras.Model([inp, inp2], segmentation)



def deconvolutor(input_shape1=1024) :

    inp = keras.Input(input_shape1)
    l1 = layers.Dense(1024)(inp)
    l1 = layers.PReLU()(l1)

    deflat = layers.Reshape((8, 8, 16))(inp)
    c1 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(deflat)  # 8 8 256
    c1 = layers.PReLU()(c1)
    c1 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(c1)  # 8 8 256
    c1 = layers.PReLU()(c1)
    c1_r = layers.Reshape((16, 16, 64))(c1)  # 16 16 64
    c2 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(c1_r)
    c2 = layers.PReLU()(c2)
    c2 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(c2)
    c2 = layers.PReLU()(c2)
    c2_r = layers.Reshape((32, 32, 64))(c2)  #32 32 64
    c3 = layers.Conv2D(256, strides=1, kernel_size=3, padding='same')(c2_r)
    c3 = layers.PReLU()(c3)
    c4 = layers.Conv2D(256, kernel_size=3, padding='same', strides=1)(c3)  # 32 32 256
    c4 = layers.PReLU()(c4)
    c4_r = layers.Reshape((64, 64, 64))(c4)
    c5 = layers.Conv2D(256, strides=1, kernel_size=3, padding='same')(c4_r)
    c5 = layers.PReLU()(c5)
    reconstruction = layers.Conv2D(5, kernel_size=3, padding='same', strides=1, activation='linear')(c5)
    return keras.Model(inp, reconstruction)













def ResNet50():
    inp = tf.keras.Input((64, 64, 9))
    c1 = layers.Conv2D(64, kernel_size=7, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-7))(inp)  # 32, 32
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.ReLU()(c1)
    c1 = layers.MaxPooling2D((2, 2))(c1)  # 16, 16

    r1 = bottleneck_block(c1, 64, downsample=True)
    r1 = bottleneck_block(r1, 64)
    r1 = bottleneck_block(r1, 64)

    r2 = bottleneck_block(r1, 128, 2, True)   #8, 8
    r2 = bottleneck_block(r2, 128)
    r2 = bottleneck_block(r2, 128)
    r2 = bottleneck_block(r2, 128)

    r3 = bottleneck_block(r2, 256, 2, True) # 4, 4
    r3 = bottleneck_block(r3, 256)
    r3 = bottleneck_block(r3, 256)
    r3 = bottleneck_block(r3, 256)
    r3 = bottleneck_block(r3, 256)
    r3 = bottleneck_block(r3, 256)

    r4 = bottleneck_block(r3, 512, 2, True)  #2, 2
    r4 = bottleneck_block(r4, 512)
    r4 = bottleneck_block(r4, 512)

    x = layers.GlobalAveragePooling2D()(r4)  # 512
    return tf.keras.Model(inp, x)

def bottleneck_block(x, filters, stride=1, downsample=False) :
    identity = x
    if downsample :
        identity = layers.Conv2D(filters*4, (1, 1), strides=stride, padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(5e-7))(identity)
        identity = layers.BatchNormalization()(identity)

    x = conv_block(x, filters, kernel_size=1, stride=stride)
    x = conv_block(x, filters)

    x = layers.Conv2D(filters*4, (1, 1), use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(5e-7))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, identity])
    x = layers.ReLU()(x)
    return x

def conv_block(x, filters, kernel_size=3, stride=1, padding='same'):
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding=padding, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(5e-7))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x
