import argparse
import os

import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.utils import multi_gpu_model

from config import patience, batch_size, epochs, num_train_samples, num_valid_samples
from data_generator_2 import train_gen, valid_gen
from migrate import migrate_model
from segnet_v6 import build_encoder_decoder, build_refinement
from utils import overall_loss, get_available_cpus, get_available_gpus

log_dir = './logs_6'
checkpoint_models_path = './checkpoints_6/cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_models_path)

if __name__ == '__main__':
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--pretrained", help="path to save pretrained model files")
    args = vars(ap.parse_args())
    pretrained_path = args["pretrained"]

    # Callbacks
    tensor_board = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
    # model_names = checkpoint_models_path + 'final.{epoch:02d}-{val_loss:.4f}.hdf5'
    model_checkpoint = ModelCheckpoint(filepath=checkpoint_models_path, monitor='val_loss', verbose=1, save_weights_only=True)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 4), verbose=1)


    class MyCbk(keras.callbacks.Callback):
        def __init__(self, model):
            keras.callbacks.Callback.__init__(self)
            self.model_to_save = model

        def on_epoch_end(self, epoch, logs=None):
            fmt = checkpoint_models_path + 'final.%02d-%.4f.hdf5'
            self.model_to_save.save(fmt % (epoch, logs['val_loss']))


    # Load our model, added support for Multi-GPUs
    num_gpu = len(get_available_gpus())
    if num_gpu >= 2:
        with tf.device("/cpu:0"):
            model = build_encoder_decoder()
            model = build_refinement(model)
            # if pretrained_path is not None:
            #     model.load_weights(pretrained_path)

        final = multi_gpu_model(model, gpus=num_gpu)
        # rewrite the callback: saving through the original model and not the multi-gpu model.
        model_checkpoint = MyCbk(model)
    else:
        model = build_encoder_decoder()
        final = build_refinement(model)
        # if pretrained_path is not None:
        #     final.load_weights(pretrained_path)
    if len(os.listdir(checkpoint_dir)) > 0:
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        final.load_weights(latest)
    final.compile(optimizer='nadam', loss=overall_loss)

    print(final.summary())

    # Final callbacks
    callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]

    # Start Fine-tuning
    final.fit_generator(train_gen(),
                        steps_per_epoch=num_train_samples // batch_size,
                        validation_data=valid_gen(),
                        validation_steps=num_valid_samples // batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks,
                        # use_multiprocessing=True,
                        # workers=2
                        )
