import tensorflow.keras.backend as K
import numpy as np

from config import channel
from model_v16 import build_encoder_decoder
from vgg16 import vgg16_model

vgg_16 = [4,7,9,12,14,16,19,21,26,28,30]
v_16 = [2,4,5,7,8,9,11,12,24,25,26]

layer_names=[layer.name for layer in build_encoder_decoder().layers]
for layer in layer_names:
    print(layer)

def migrate_model(new_model):
    old_model = vgg16_model(224, 224, 3)
    print(old_model.summary())
    old_layers = [l for l in old_model.layers]
    new_layers = [l for l in new_model.layers]

    old_conv1_1 = old_model.get_layer('conv1_1')
    old_weights = old_conv1_1.get_weights()[0]
    old_biases = old_conv1_1.get_weights()[1]
    new_weights = np.zeros((3, 3, channel, 64), dtype=np.float32)
    new_weights[:, :, 0:3, :] = old_weights
    new_weights[:, :, 3:channel, :] = 0.0
    new_conv1_1 = new_model.get_layer('conv1_1')
    new_conv1_1.set_weights([new_weights, old_biases])

    # for i in range(2, 31):
    #     old_layer = old_layers[i]
    #     new_layer = new_layers[i + 1]
    #     new_layer.set_weights(old_layer.get_weights())
    for i in range(11):
        index_vgg = vgg_16[i]
        index_model = v_16[i]
        old_layer = old_layers[index_vgg-1]
        new_layer = new_layers[index_model]
        new_layer.set_weights(old_layer.get_weights())
    # flatten = old_model.get_layer('flatten')
    # f_dim = flatten.input_shape
    # print('f_dim: ' + str(f_dim))
    # old_dense1 = old_model.get_layer('dense1')
    # input_shape = old_dense1.input_shape
    # output_dim = old_dense1.get_weights()[1].shape[0]
    # print('output_dim: ' + str(output_dim))
    # W, b = old_dense1.get_weights()
    # shape = (7, 7, 512, output_dim)
    # new_W = W.reshape(shape)
    # new_conv6 = new_model.get_layer('conv6')
    # new_conv6.set_weights([new_W, b])

    del old_model


if __name__ == '__main__':
    model = build_encoder_decoder()
    migrate_model(model)
    print(model.summary())
    model.save_weights('models/model_v16_weights.h5')

    K.clear_session()
