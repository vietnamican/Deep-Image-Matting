import tensorflow.keras.backend as K
import tensorflow.compat.v1 as tf
from tensorflow.keras.layers import Add,  Activation, DepthwiseConv2D, Input, Conv2D, SeparableConv2D, UpSampling2D, BatchNormalization, ZeroPadding2D, MaxPooling2D, Reshape, \
    Concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.utils import plot_model

from custom_layers.unpooling_layer import Unpooling

ATROUS_RATES = [6, 12, 18]

def xception_block(x,channels):
    ##separable conv1
    x=Activation("relu")(x)
    x=SeparableConv2D(channels, (3,3), padding='same')(x)
    x=BatchNormalization()(x)
    
    ##separable conv2
    x=Activation("relu")(x)
    x=SeparableConv2D(channels, (3,3), padding='same')(x)
    x=BatchNormalization()(x)
    
    ##separable conv3
    x=Activation("relu")(x)
    x=SeparableConv2D(channels, (3,3), padding='same')(x)
    x=BatchNormalization()(x)
    return x
def downsample_xception_block(x, channels, top_relu=False):
    if(top_relu):
        x = Activation("relu")(x)
    x=SeparableConv2D(channels, (3,3), padding='same')(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    
    ##separable conv2
    x=DepthwiseConv2D((3,3),padding="same")(x)
    x=BatchNormalization()(x)
    x=Conv2D(channels,(1,1),padding="same")(x)
    skip=BatchNormalization()(x)
    x=Activation("relu")(skip)
    
    ##separable conv3
    x=SeparableConv2D(channels, (3,3), strides=2, padding='same')(x)
    x=BatchNormalization()(x)
    return x, skip

def res_xception_block(x, channels):
    res=x
    x=xception_block(x,channels)
    x=Add()([x,res])
    return x
def res_downsample_xception_block(x, channels, top_relu= False):
    res=Conv2D(channels,(1,1),strides=(2,2),padding="same")(x)
    res=BatchNormalization()(res)
    x,skip=downsample_xception_block(x, channels, top_relu)
    x=Add()([x,res])
    return x,skip

def build_encoder_decoder():
    kernel = 3

    # Encoder
    #
    input_tensor = Input(shape=(320, 320, 4))
    input_tensor_shape = input_tensor.get_shape()[1:3]

    # Entry flow
    x=Conv2D(32,(3,3),padding="same")(input_tensor)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    x=Conv2D(64,(3,3),padding="same")(x)
    orig_1=BatchNormalization()(x)
    x=Activation("relu")(orig_1)
    x=x=Conv2D(64,(3,3),strides=(2,2),padding="same")(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    x, orig_2 = res_downsample_xception_block(x, 128)
    x, orig_3 = res_downsample_xception_block(x, 256, top_relu=True)
    x, orig_4 = res_downsample_xception_block(x, 728, top_relu=True)

    # Middle flow
    for i in range(8):
        x=res_xception_block(x,728)

    # Exit flow
    res=Conv2D(1024,(1,1),padding="same")(x)
    res=BatchNormalization()(res)    
    x=Activation("relu")(x)
    x=DepthwiseConv2D((3,3),padding="same")(x)
    x=BatchNormalization()(x)
    x=Conv2D(728,(1,1),padding="same")(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    x=DepthwiseConv2D((3,3),padding="same")(x)
    x=BatchNormalization()(x)
    x=Conv2D(1024,(1,1),padding="same")(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    x=DepthwiseConv2D((3,3),padding="same")(x)
    x=BatchNormalization()(x)
    x=Conv2D(1024,(1,1),padding="same")(x)
    x=BatchNormalization()(x)    
    x=Add()([x,res])
    
    x=DepthwiseConv2D((3,3),padding="same")(x)
    x=BatchNormalization()(x)
    x=Conv2D(1536,(1,1),padding="same")(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    x=DepthwiseConv2D((3,3),padding="same")(x)
    x=BatchNormalization()(x)
    x=Conv2D(1536,(1,1),padding="same")(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    x=DepthwiseConv2D((3,3),padding="same")(x)
    x=BatchNormalization()(x)
    x=Conv2D(2048,(1,1),padding="same")(x)
    x=BatchNormalization()(x)    
    x=Activation("relu")(x)

    inputs_size = x.get_shape()[1:3]
    # Atrous convolution
    conv_4_1x1 = SeparableConv2D(256, (1, 1), activation='relu', padding='same', name='conv4_1x1')(x)
    conv_4_3x3_1 = SeparableConv2D(256, (kernel, kernel), activation='relu', padding='same', dilation_rate=ATROUS_RATES[0], name='conv4_3x3_1')(x)
    conv_4_3x3_2 = SeparableConv2D(256, (kernel, kernel), activation='relu', padding='same', dilation_rate=ATROUS_RATES[1], name='conv4_3x3_2')(x)
    conv_4_3x3_3 = SeparableConv2D(256, (kernel, kernel), activation='relu', padding='same', dilation_rate=ATROUS_RATES[2], name='conv4_3x3_3')(x)  
    # Image average pooling
    image_level_features = Lambda(lambda x: tf.reduce_mean(x, [1, 2], keepdims=True), name='global_average_pooling')(x)
    image_level_features = Conv2D(256, (1, 1), activation='relu', padding='same', name='image_level_features_conv_1x1')(image_level_features)
    image_level_features = Lambda(lambda x: tf.image.resize_bilinear(x, inputs_size), name='upsample_1')(image_level_features)
    # Concat
    x = Concatenate(axis=3)([conv_4_1x1, conv_4_3x3_1, conv_4_3x3_2, conv_4_3x3_3, image_level_features])
    x = Conv2D(256, (1,1), activation='relu', padding='same', name='conv_1x1_concat')(x)    

    # Decoderg
    #
    x = UpSampling2D(size=(2, 2))(x)
    the_shape = K.int_shape(orig_4)
    shape = (1, the_shape[1], the_shape[2], the_shape[3])
    origReshaped = Reshape(shape)(orig_4)
    xReshaped = Reshape(shape)(x)
    together = Concatenate(axis=1)([origReshaped, xReshaped])
    x = Unpooling()(together)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='deconv4_1',
               kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='deconv4_2',
               kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='deconv4_3',
               kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2))(x)
    the_shape = K.int_shape(orig_3)
    shape = (1, the_shape[1], the_shape[2], the_shape[3])
    origReshaped = Reshape(shape)(orig_3)
    xReshaped = Reshape(shape)(x)
    together = Concatenate(axis=1)([origReshaped, xReshaped])
    x = Unpooling()(together)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='deconv3_1',
               kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='deconv3_2',
               kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='deconv3_3',
               kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2))(x)
    the_shape = K.int_shape(orig_2)
    shape = (1, the_shape[1], the_shape[2], the_shape[3])
    origReshaped = Reshape(shape)(orig_2)
    xReshaped = Reshape(shape)(x)
    together = Concatenate(axis=1)([origReshaped, xReshaped])
    x = Unpooling()(together)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='deconv2_1',
               kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='deconv2_2',
               kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2))(x)
    the_shape = K.int_shape(orig_1)
    shape = (1, the_shape[1], the_shape[2], the_shape[3])
    origReshaped = Reshape(shape)(orig_1)
    xReshaped = Reshape(shape)(x)
    together = Concatenate(axis=1)([origReshaped, xReshaped])
    x = Unpooling()(together)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='deconv1_1',
               kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='deconv1_2',
               kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)

    x = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='pred', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    model = Model(inputs=input_tensor, outputs=x)
    return model


def build_refinement(encoder_decoder):
    input_tensor = encoder_decoder.input

    input = Lambda(lambda i: i[:, :, :, 0:3])(input_tensor)

    x = Concatenate(axis=3)([input, encoder_decoder.output])
    x = SeparableConv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = SeparableConv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = SeparableConv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = SeparableConv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = SeparableConv2D(1, (3, 3), activation='sigmoid', padding='same', name='refinement_pred', kernel_initializer='he_normal',
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
