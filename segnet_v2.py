import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization, ZeroPadding2D, MaxPooling2D, Reshape, \
    Concatenate, Lambda
from keras.models import Model
from keras.utils import multi_gpu_model
from keras.utils import plot_model

from custom_layers.unpooling_layer import Unpooling

ATROUS_RATES = [6, 12, 18]

def build_encoder_decoder():
    kernel = 3

    # Encoder
    #
    input_tensor = Input(shape=(320, 320, 4))
    input_tensor_shape = input_tensor.get_shape()[1:3]
    x = ZeroPadding2D((1, 1))(input_tensor)
    x = Conv2D(64, (kernel, kernel), activation='relu', name='conv1_1')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(64, (kernel, kernel), activation='relu', name='conv1_2')(x)
    orig_1 = x
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(128, (kernel, kernel), activation='relu', name='conv2_1')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(128, (kernel, kernel), activation='relu', name='conv2_2')(x)
    orig_2 = x
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, (kernel, kernel), activation='relu', name='conv3_1')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, (kernel, kernel), activation='relu', name='conv3_2')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, (kernel, kernel), activation='relu', name='conv3_3')(x)
    orig_3 = x
    low_level_feature = x
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    inputs_size = x.get_shape()[1:3]
    # Atrous convolution
    conv_4_1x1 = Conv2D(256, (1, 1), activation='relu', padding='same', name='conv4_1x1')(x)
    conv_4_3x3_1 = Conv2D(256, (kernel, kernel), activation='relu', padding='same', dilation_rate=ATROUS_RATES[0], name='conv4_3x3_1')(x)
    conv_4_3x3_2 = Conv2D(256, (kernel, kernel), activation='relu', padding='same', dilation_rate=ATROUS_RATES[1], name='conv4_3x3_2')(x)
    conv_4_3x3_3 = Conv2D(256, (kernel, kernel), activation='relu', padding='same', dilation_rate=ATROUS_RATES[2], name='conv4_3x3_3')(x)  
    # Image average pooling
    image_level_features = Lambda(lambda x: tf.reduce_mean(x, [1, 2], keepdims=True), name='global_average_pooling')(x)
    image_level_features = Conv2D(256, (1, 1), activation='relu', padding='same', name='image_level_features_conv_1x1')(image_level_features)
    image_level_features = Lambda(lambda x: tf.image.resize_bilinear(x, inputs_size), name='upsample_1')(image_level_features)
    # Concat
    x = Concatenate(axis=3)([conv_4_1x1, conv_4_3x3_1, conv_4_3x3_2, conv_4_3x3_3, image_level_features])
    x = Conv2D(256, (1,1), activation='relu', padding='same', name='conv_1x1_concat')(x)    

    # Decoder
    #
    low_level_feature_shape = low_level_feature.get_shape()[1:3]
    x = Lambda(lambda x: tf.image.resize_bilinear(x, low_level_feature_shape),name="upsample_2")(x)

    x = Concatenate(axis=3)([x, low_level_feature])
    x = Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer='he_normal', bias_initializer='zeros', name="dev_conv3_1")(x)
    x = Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer='he_normal', bias_initializer='zeros', name="dev_conv3_2")(x)

    x = Lambda(lambda x: tf.image.resize_bilinear(x, input_tensor_shape), name="upsample_3")(x)
    x = Conv2D(1, (3,3), activation='sigmoid', padding='same', kernel_initializer='he_normal', bias_initializer='zeros',name='pred')(x)
    model = Model(inputs=input_tensor, outputs=x)
    return model


def build_refinement(encoder_decoder):
    input_tensor = encoder_decoder.input

    input = Lambda(lambda i: i[:, :, :, 0:3])(input_tensor)

    x = Concatenate(axis=3)([input, encoder_decoder.output])
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='refinement_pred', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)

    model = Model(inputs=input_tensor, outputs=x)
    return model


if __name__ == '__main__':
    with tf.device("/cpu:0"):
        encoder_decoder = build_encoder_decoder()
    print(encoder_decoder.summary())
    plot_model(encoder_decoder, to_file='encoder_decoder.svg', show_layer_names=True, show_shapes=True)

    with tf.device("/cpu:0"):
        refinement = build_refinement(encoder_decoder)
    print(refinement.summary())
    plot_model(refinement, to_file='refinement.svg', show_layer_names=True, show_shapes=True)

    parallel_model = multi_gpu_model(refinement, gpus=None)
    print(parallel_model.summary())
    plot_model(parallel_model, to_file='parallel_model.svg', show_layer_names=True, show_shapes=True)

    K.clear_session()
