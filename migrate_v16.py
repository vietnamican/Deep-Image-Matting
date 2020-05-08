import tensorflow.keras.backend as K
import numpy as np

from config import channel
import model_v16 
import model
from vgg16 import vgg16_model

def migrate_model(new_model):
    vgg_16 = [4,7,9,12,14,16,19,21,26,28,30]
    v_16 = [2,4,5,7,8,9,11,12,24,25,26]
    old_model = vgg16_model(224, 224, 3)
    # print(old_model.summary())
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

    for i in range(11):
        index_vgg = vgg_16[i]
        index_model = v_16[i]
        old_layer = old_layers[index_vgg-1]
        new_layer = new_layers[index_model]
        new_layer.set_weights(old_layer.get_weights())
    del old_model


def migrate_model_2(new_model):
    old_model = model.build_encoder_decoder()
    old_model = model.build_refinement(old_model)
    old_model.load_weights("models/final.42-0.0398.hdf5")
    old = [2,4,7,9,12,14,16,19,21,26,28,30]
    new = [1,2,4,5,7,8,9,11,12,24,25,26]
    print(old_model.summary())
    old_layers = [l for l in old_model.layers]
    new_layers = [l for l in new_model.layers]

    for i in range(12):
        index_vgg = old[i]
        index_model = new[i]
        old_layer = old_layers[index_vgg]
        new_layer = new_layers[index_model]
        new_layer.set_weights(old_layer.get_weights())

    for i in range(28, 74):
        # print(i)
        old_layer = old_layers[i+4]
        new_layer = new_layers[i]
        new_layer.set_weights(old_layer.get_weights())
    del old_model

if __name__ == '__main__':
    new_model = model_v16.build_encoder_decoder()
    new_model = model_v16.build_refinement(new_model)
    migrate_model_2(new_model)
    print(new_model.summary())
    new_model.save_weights('models/model_v16_2_42_weights.h5')

    K.clear_session()
