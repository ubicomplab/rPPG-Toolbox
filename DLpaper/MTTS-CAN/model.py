'''
Models for Multi-Task Temporal Shift Attention Networks for On-Device Contactless Vitals Measurement
Author: Xin Liu
'''

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Conv2D, Conv3D, Input, AveragePooling2D, \
    multiply, Dense, Dropout, Flatten, AveragePooling3D
from tensorflow.python.keras.models import Model


class Attention_mask(tf.keras.layers.Layer):
    def call(self, x):
        xsum = K.sum(x, axis=1, keepdims=True)
        xsum = K.sum(xsum, axis=2, keepdims=True)
        xshape = K.int_shape(x)
        return x / xsum * xshape[1] * xshape[2] * 0.5

    def get_config(self):
        config = super(Attention_mask, self).get_config()
        return config


class TSM(tf.keras.layers.Layer):
    def call(self, x, n_frame, fold_div=3):
        nt, h, w, c = x.shape
        x = K.reshape(x, (-1, n_frame, h, w, c))
        fold = c // fold_div
        last_fold = c - (fold_div - 1) * fold
        out1, out2, out3 = tf.split(x, [fold, fold, last_fold], axis=-1)

        # Shift left
        padding_1 = tf.zeros_like(out1)
        padding_1 = padding_1[:, -1, :, :, :]
        padding_1 = tf.expand_dims(padding_1, 1)
        _, out1 = tf.split(out1, [1, n_frame - 1], axis=1)
        out1 = tf.concat([out1, padding_1], axis=1)

        # Shift right
        padding_2 = tf.zeros_like(out2)
        padding_2 = padding_2[:, 0, :, :, :]
        padding_2 = tf.expand_dims(padding_2, 1)
        out2, _ = tf.split(out2, [n_frame - 1, 1], axis=1)
        out2 = tf.concat([padding_2, out2], axis=1)

        out = tf.concat([out1, out2, out3], axis=-1)
        out = K.reshape(out, (-1, h, w, c))

        return out

    def get_config(self):
        config = super(TSM, self).get_config()
        return config


def TSM_Cov2D(x, n_frame, nb_filters=128, kernel_size=(3, 3), activation='tanh', padding='same'):
    x = TSM()(x, n_frame)
    x = Conv2D(nb_filters, kernel_size, padding=padding, activation=activation)(x)
    return x


# %%


def CAN(nb_filters1, nb_filters2, input_shape, kernel_size=(3, 3), dropout_rate1=0.25, dropout_rate2=0.5,
        pool_size=(2, 2), nb_dense=128):
    diff_input = Input(shape=input_shape)
    rawf_input = Input(shape=input_shape)

    d1 = Conv2D(nb_filters1, kernel_size, padding='same', activation='tanh')(diff_input)
    d2 = Conv2D(nb_filters1, kernel_size, activation='tanh')(d1)

    r1 = Conv2D(nb_filters1, kernel_size, padding='same', activation='tanh')(rawf_input)
    r2 = Conv2D(nb_filters1, kernel_size, activation='tanh')(r1)

    g1 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r2)
    g1 = Attention_mask()(g1)
    gated1 = multiply([d2, g1])

    d3 = AveragePooling2D(pool_size)(gated1)
    d4 = Dropout(dropout_rate1)(d3)

    r3 = AveragePooling2D(pool_size)(r2)
    r4 = Dropout(dropout_rate1)(r3)

    d5 = Conv2D(nb_filters2, kernel_size, padding='same', activation='tanh')(d4)
    d6 = Conv2D(nb_filters2, kernel_size, activation='tanh')(d5)

    r5 = Conv2D(nb_filters2, kernel_size, padding='same', activation='tanh')(r4)
    r6 = Conv2D(nb_filters2, kernel_size, activation='tanh')(r5)

    g2 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r6)
    g2 = Attention_mask()(g2)
    gated2 = multiply([d6, g2])

    d7 = AveragePooling2D(pool_size)(gated2)
    d8 = Dropout(dropout_rate1)(d7)

    d9 = Flatten()(d8)
    d10 = Dense(nb_dense, activation='tanh')(d9)
    d11 = Dropout(dropout_rate2)(d10)
    out = Dense(1)(d11)
    model = Model(inputs=[diff_input, rawf_input], outputs=out)
    return model


# %% MT_CAN
def MT_CAN(nb_filters1, nb_filters2, input_shape, kernel_size=(3, 3), dropout_rate1=0.25, dropout_rate2=0.5,
           pool_size=(2, 2), nb_dense=128):
    diff_input = Input(shape=input_shape)
    rawf_input = Input(shape=input_shape)

    d1 = Conv2D(nb_filters1, kernel_size, padding='same', activation='tanh')(diff_input)
    d2 = Conv2D(nb_filters1, kernel_size, activation='tanh')(d1)

    r1 = Conv2D(nb_filters1, kernel_size, padding='same', activation='tanh')(rawf_input)
    r2 = Conv2D(nb_filters1, kernel_size, activation='tanh')(r1)

    g1 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r2)
    g1 = Attention_mask()(g1)
    gated1 = multiply([d2, g1])

    d3 = AveragePooling2D(pool_size)(gated1)
    d4 = Dropout(dropout_rate1)(d3)

    r3 = AveragePooling2D(pool_size)(r2)
    r4 = Dropout(dropout_rate1)(r3)

    d5 = Conv2D(nb_filters2, kernel_size, padding='same', activation='tanh')(d4)
    d6 = Conv2D(nb_filters2, kernel_size, activation='tanh')(d5)

    r5 = Conv2D(nb_filters2, kernel_size, padding='same', activation='tanh')(r4)
    r6 = Conv2D(nb_filters2, kernel_size, activation='tanh')(r5)

    g2 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r6)
    g2 = Attention_mask()(g2)
    gated2 = multiply([d6, g2])

    d7 = AveragePooling2D(pool_size)(gated2)
    d8 = Dropout(dropout_rate1)(d7)

    d9 = Flatten()(d8)
    d10_y = Dense(nb_dense, activation='tanh')(d9)
    d11_y = Dropout(dropout_rate2)(d10_y)
    out_y = Dense(1, name='output_1')(d11_y)

    d10_r = Dense(nb_dense, activation='tanh')(d9)
    d11_r = Dropout(dropout_rate2)(d10_r)
    out_r = Dense(1, name='output_2')(d11_r)

    model = Model(inputs=[diff_input, rawf_input], outputs=[out_y, out_r])
    return model


# %% TS_CAN

def TS_CAN(n_frame, nb_filters1, nb_filters2, input_shape, kernel_size=(3, 3), dropout_rate1=0.25, dropout_rate2=0.5,
           pool_size=(2, 2), nb_dense=128):
    diff_input = Input(shape=input_shape)
    rawf_input = Input(shape=input_shape)

    d1 = TSM_Cov2D(diff_input, n_frame, nb_filters1, kernel_size, padding='same', activation='tanh')
    d2 = TSM_Cov2D(d1, n_frame, nb_filters1, kernel_size, padding='valid', activation='tanh')

    r1 = Conv2D(nb_filters1, kernel_size, padding='same', activation='tanh')(rawf_input)
    r2 = Conv2D(nb_filters1, kernel_size, activation='tanh')(r1)

    g1 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r2)
    g1 = Attention_mask()(g1)
    gated1 = multiply([d2, g1])

    d3 = AveragePooling2D(pool_size)(gated1)
    d4 = Dropout(dropout_rate1)(d3)

    r3 = AveragePooling2D(pool_size)(r2)
    r4 = Dropout(dropout_rate1)(r3)

    d5 = TSM_Cov2D(d4, n_frame, nb_filters2, kernel_size, padding='same', activation='tanh')
    d6 = TSM_Cov2D(d5, n_frame, nb_filters2, kernel_size, padding='valid', activation='tanh')

    r5 = Conv2D(nb_filters2, kernel_size, padding='same', activation='tanh')(r4)
    r6 = Conv2D(nb_filters2, kernel_size, activation='tanh')(r5)

    g2 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r6)
    g2 = Attention_mask()(g2)
    gated2 = multiply([d6, g2])

    d7 = AveragePooling2D(pool_size)(gated2)
    d8 = Dropout(dropout_rate1)(d7)

    d9 = Flatten()(d8)
    d10 = Dense(nb_dense, activation='tanh')(d9)
    d11 = Dropout(dropout_rate2)(d10)
    out = Dense(1)(d11)
    model = Model(inputs=[diff_input, rawf_input], outputs=out)
    return model


# %% MTTS-CAN

def MTTS_CAN(n_frame, nb_filters1, nb_filters2, input_shape, kernel_size=(3, 3), dropout_rate1=0.25,
             dropout_rate2=0.5, pool_size=(2, 2), nb_dense=128):
    diff_input = Input(shape=input_shape)
    rawf_input = Input(shape=input_shape)

    d1 = TSM_Cov2D(diff_input, n_frame, nb_filters1, kernel_size, padding='same', activation='tanh')
    d2 = TSM_Cov2D(d1, n_frame, nb_filters1, kernel_size, padding='valid', activation='tanh')

    r1 = Conv2D(nb_filters1, kernel_size, padding='same', activation='tanh')(rawf_input)
    r2 = Conv2D(nb_filters1, kernel_size, activation='tanh')(r1)

    g1 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r2)
    g1 = Attention_mask()(g1)
    gated1 = multiply([d2, g1])

    d3 = AveragePooling2D(pool_size)(gated1)
    d4 = Dropout(dropout_rate1)(d3)

    r3 = AveragePooling2D(pool_size)(r2)
    r4 = Dropout(dropout_rate1)(r3)

    d5 = TSM_Cov2D(d4, n_frame, nb_filters2, kernel_size, padding='same', activation='tanh')
    d6 = TSM_Cov2D(d5, n_frame, nb_filters2, kernel_size, padding='valid', activation='tanh')

    r5 = Conv2D(nb_filters2, kernel_size, padding='same', activation='tanh')(r4)
    r6 = Conv2D(nb_filters2, kernel_size, activation='tanh')(r5)

    g2 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r6)
    g2 = Attention_mask()(g2)
    gated2 = multiply([d6, g2])

    d7 = AveragePooling2D(pool_size)(gated2)
    d8 = Dropout(dropout_rate1)(d7)

    d9 = Flatten()(d8)

    d10_y = Dense(nb_dense, activation='tanh')(d9)
    d11_y = Dropout(dropout_rate2)(d10_y)
    out_y = Dense(1, name='output_1')(d11_y)

    d10_r = Dense(nb_dense, activation='tanh')(d9)
    d11_r = Dropout(dropout_rate2)(d10_r)
    out_r = Dense(1, name='output_2')(d11_r)

    model = Model(inputs=[diff_input, rawf_input], outputs=[out_y, out_r])
    return model


# %%
def CAN_3D(n_frame, nb_filters1, nb_filters2, input_shape, kernel_size=(3, 3, 3), dropout_rate1=0.25, dropout_rate2=0.5,
           pool_size=(2, 2, 2), nb_dense=128):
    diff_input = Input(shape=input_shape)
    rawf_input = Input(shape=input_shape)

    d1 = Conv3D(nb_filters1, kernel_size, padding='same', activation='tanh')(diff_input)
    d2 = Conv3D(nb_filters1, kernel_size, activation='tanh')(d1)

    # Appearance Branch
    r1 = Conv3D(nb_filters1, kernel_size, padding='same', activation='tanh')(rawf_input)
    r2 = Conv3D(nb_filters1, kernel_size, activation='tanh')(r1)
    g1 = Conv3D(1, (1, 1, 1), padding='same', activation='sigmoid')(r2)
    g1 = Attention_mask()(g1)
    gated1 = multiply([d2, g1])
    d3 = AveragePooling3D(pool_size)(gated1)

    d4 = Dropout(dropout_rate1)(d3)
    d5 = Conv3D(nb_filters2, kernel_size, padding='same', activation='tanh')(d4)
    d6 = Conv3D(nb_filters2, kernel_size, activation='tanh')(d5)

    r3 = AveragePooling3D(pool_size)(r2)
    r4 = Dropout(dropout_rate1)(r3)
    r5 = Conv3D(nb_filters2, kernel_size, padding='same', activation='tanh')(r4)
    r6 = Conv3D(nb_filters2, kernel_size, activation='tanh')(r5)
    g2 = Conv3D(1, (1, 1, 1), padding='same', activation='sigmoid')(r6)
    g2 = Attention_mask()(g2)
    gated2 = multiply([d6, g2])

    d7 = AveragePooling3D(pool_size)(gated2)
    d8 = Dropout(dropout_rate1)(d7)
    d9 = Flatten()(d8)
    d10 = Dense(nb_dense, activation='tanh')(d9)
    d11 = Dropout(dropout_rate2)(d10)
    out = Dense(n_frame)(d11)
    model = Model(inputs=[diff_input, rawf_input], outputs=out)
    return model


# input_shape = (36, 36, 10, 3)
# model = DeepPhy_3DCNN(10, 32, 64, input_shape)
# print('==========================')


# %%
def MT_CAN_3D(n_frame, nb_filters1, nb_filters2, input_shape, kernel_size=(3, 3, 3), dropout_rate1=0.25,
              dropout_rate2=0.5, pool_size=(2, 2, 2), nb_dense=128):
    diff_input = Input(shape=input_shape)
    rawf_input = Input(shape=input_shape)

    d1 = Conv3D(nb_filters1, kernel_size, padding='same', activation='tanh')(diff_input)
    d2 = Conv3D(nb_filters1, kernel_size, activation='tanh')(d1)

    # Appearance Branch
    r1 = Conv3D(nb_filters1, kernel_size, padding='same', activation='tanh')(rawf_input)
    r2 = Conv3D(nb_filters1, kernel_size, activation='tanh')(r1)
    g1 = Conv3D(1, (1, 1, 1), padding='same', activation='sigmoid')(r2)
    g1 = Attention_mask()(g1)
    gated1 = multiply([d2, g1])

    d3 = AveragePooling3D(pool_size)(gated1)
    d4 = Dropout(dropout_rate1)(d3)
    d5 = Conv3D(nb_filters2, kernel_size, padding='same', activation='tanh')(d4)
    d6 = Conv3D(nb_filters2, kernel_size, activation='tanh')(d5)

    r3 = AveragePooling3D(pool_size)(r2)
    r4 = Dropout(dropout_rate1)(r3)
    r5 = Conv3D(nb_filters2, kernel_size, padding='same', activation='tanh')(r4)
    r6 = Conv3D(nb_filters2, kernel_size, activation='tanh')(r5)
    g2 = Conv3D(1, (1, 1, 1), padding='same', activation='sigmoid')(r6)
    g2 = Attention_mask()(g2)
    gated2 = multiply([d6, g2])
    d7 = AveragePooling3D(pool_size)(gated2)
    d8 = Dropout(dropout_rate1)(d7)

    d9 = Flatten()(d8)
    d10_y = Dense(nb_dense, activation='tanh')(d9)
    d11_y = Dropout(dropout_rate2)(d10_y)
    out_y = Dense(n_frame, name='output_1')(d11_y)

    d10_r = Dense(nb_dense, activation='tanh')(d9)
    d11_r = Dropout(dropout_rate2)(d10_r)
    out_r = Dense(n_frame, name='output_2')(d11_r)

    model = Model(inputs=[diff_input, rawf_input], outputs=[out_y, out_r])

    return model


# %%

def Hybrid_CAN(n_frame, nb_filters1, nb_filters2, input_shape_1, input_shape_2, kernel_size_1=(3, 3, 3),
               kernel_size_2=(3, 3), dropout_rate1=0.25, dropout_rate2=0.5, pool_size_1=(2, 2, 2), pool_size_2=(2, 2),
               nb_dense=128):
    diff_input = Input(shape=input_shape_1)
    rawf_input = Input(shape=input_shape_2)

    # Motion branch
    d1 = Conv3D(nb_filters1, kernel_size_1, padding='same', activation='tanh')(diff_input)
    d2 = Conv3D(nb_filters1, kernel_size_1, activation='tanh')(d1)

    # App branch
    r1 = Conv2D(nb_filters1, kernel_size_2, padding='same', activation='tanh')(rawf_input)
    r2 = Conv2D(nb_filters1, kernel_size_2, activation='tanh')(r1)

    # Mask from App (g1) * Motion Branch (d2)
    g1 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r2)
    g1 = Attention_mask()(g1)
    g1 = K.expand_dims(g1, axis=-1)
    gated1 = multiply([d2, g1])

    # Motion Branch
    d3 = AveragePooling3D(pool_size_1)(gated1)
    d4 = Dropout(dropout_rate1)(d3)
    d5 = Conv3D(nb_filters2, kernel_size_1, padding='same', activation='tanh')(d4)
    d6 = Conv3D(nb_filters2, kernel_size_1, activation='tanh')(d5)

    # App branch
    r3 = AveragePooling2D(pool_size_2)(r2)
    r4 = Dropout(dropout_rate1)(r3)
    r5 = Conv2D(nb_filters2, kernel_size_2, padding='same', activation='tanh')(r4)
    r6 = Conv2D(nb_filters2, kernel_size_2, activation='tanh')(r5)

    # Mask from App (g2) * Motion Branch (d6)
    g2 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r6)
    g2 = Attention_mask()(g2)
    g2 = K.repeat_elements(g2, d6.shape[3], axis=-1)
    g2 = K.expand_dims(g2, axis=-1)
    gated2 = multiply([d6, g2])

    # Motion Branch
    d7 = AveragePooling3D(pool_size_1)(gated2)
    d8 = Dropout(dropout_rate1)(d7)

    # Motion Branch
    d9 = Flatten()(d8)
    d10 = Dense(nb_dense, activation='tanh')(d9)
    d11 = Dropout(dropout_rate2)(d10)
    out = Dense(n_frame)(d11)

    model = Model(inputs=[diff_input, rawf_input], outputs=out)
    return model


# %%
def MT_Hybrid_CAN(n_frame, nb_filters1, nb_filters2, input_shape_1, input_shape_2, kernel_size_1=(3, 3, 3),
                  kernel_size_2=(3, 3), dropout_rate1=0.25, dropout_rate2=0.5, pool_size_1=(2, 2, 2),
                  pool_size_2=(2, 2), nb_dense=128):
    diff_input = Input(shape=input_shape_1)
    rawf_input = Input(shape=input_shape_2)

    # Motion branch
    d1 = Conv3D(nb_filters1, kernel_size_1, padding='same', activation='tanh')(diff_input)
    d2 = Conv3D(nb_filters1, kernel_size_1, activation='tanh')(d1)

    # App branch
    r1 = Conv2D(nb_filters1, kernel_size_2, padding='same', activation='tanh')(rawf_input)
    r2 = Conv2D(nb_filters1, kernel_size_2, activation='tanh')(r1)

    # Mask from App (g1) * Motion Branch (d2)
    g1 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r2)
    g1 = Attention_mask()(g1)
    g1 = K.expand_dims(g1, axis=-1)
    gated1 = multiply([d2, g1])

    # Motion Branch
    d3 = AveragePooling3D(pool_size_1)(gated1)
    d4 = Dropout(dropout_rate1)(d3)
    d5 = Conv3D(nb_filters2, kernel_size_1, padding='same', activation='tanh')(d4)
    d6 = Conv3D(nb_filters2, kernel_size_1, activation='tanh')(d5)

    # App branch
    r3 = AveragePooling2D(pool_size_2)(r2)
    r4 = Dropout(dropout_rate1)(r3)
    r5 = Conv2D(nb_filters2, kernel_size_2, padding='same', activation='tanh')(r4)
    r6 = Conv2D(nb_filters2, kernel_size_2, activation='tanh')(r5)

    # Mask from App (g2) * Motion Branch (d6)
    g2 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r6)
    g2 = Attention_mask()(g2)
    g2 = K.repeat_elements(g2, d6.shape[3], axis=-1)
    g2 = K.expand_dims(g2, axis=-1)
    gated2 = multiply([d6, g2])

    # Motion Branch
    d7 = AveragePooling3D(pool_size_1)(gated2)
    d8 = Dropout(dropout_rate1)(d7)

    # Motion Branch
    d9 = Flatten()(d8)

    d10_y = Dense(nb_dense, activation='tanh')(d9)
    d11_y = Dropout(dropout_rate2)(d10_y)
    out_y = Dense(n_frame, name='output_1')(d11_y)

    d10_r = Dense(nb_dense, activation='tanh')(d9)
    d11_r = Dropout(dropout_rate2)(d10_r)
    out_r = Dense(n_frame, name='output_2')(d11_r)

    model = Model(inputs=[diff_input, rawf_input], outputs=[out_y, out_r])
    return model


# %%
class HeartBeat(keras.callbacks.Callback):
    def __init__(self, train_gen, test_gen, args, cv_split, save_dir):
        super(HeartBeat, self).__init__()
        self.train_gen = train_gen
        self.test_gen = test_gen
        self.args = args
        self.cv_split = cv_split
        self.save_dir = save_dir

    def on_epoch_end(self, epoch, logs={}):
        print('PROGRESS: 0.00%')
