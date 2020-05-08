import os
from model_v16 import build_encoder_decoder, build_refinement
from utils import overall_loss, get_available_cpus, get_available_gpus, get_initial_epoch, get_latest_checkpoint
import tensorflow as tf

checkpoint_models_path = './checkpoints_16_4/cp-{epoch:04d}-{loss:.4f}-{val_loss:.4f}.h5'
checkpoint_dir = os.path.dirname(checkpoint_models_path)
model = build_encoder_decoder()
final = build_refinement(model)
with tf.device("/cpu:0"):
    if len(os.listdir(checkpoint_dir)) > 0:
        print(checkpoint_dir)
        # latest = get_latest_checkpoint(checkpoint_dir)
        final.load_weights("./checkpoints_16_4/cp-0020-0.0460-0.0909.ckpt")
        # initial_epoch = get_initial_epoch(latest)
    else:
        initial_epoch = 0

final.save_weights(checkpoint_models_path)