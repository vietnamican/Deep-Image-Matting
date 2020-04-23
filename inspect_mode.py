import segnet_v2
import segnet
import custom_layers
import tensorflow.keras as keras

inputs = keras.layers.Input((224,224,3))

x = custom_layers.Scale()(inputs)

model = keras.models.Model(inputs=inputs, outputs=x)
model.summary()