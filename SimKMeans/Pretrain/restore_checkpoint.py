from absl import flags
from absl import logging
import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS

def try_restore_from_checkpoint(model, global_step, optimizer):
    #Restores the latest ckpt if it exist, otherwise check FLAGS.checkpoint
    checkpoint = tf.train.Checkpoint(model=model, global_step=global_step, optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=FLAGS.model_dir, max_to_keep=FLAGS.keep_checkpoint_max)
    latest_ckpt = checkpoint_manager.latest_checkpoint
    if latest_ckpt:
        #Restore model weights, global step, optimizer states
        logging.info('Restoring from latest checkpoint: %s', latest_ckpt)
        checkpoint_manager.checkpoint.restore(latest_ckpt).expect_partial()
    elif FLAGS.checkpoint:
        #Restore model weigths only, but not global step and optimizer states
        logging.info('Restoring from given checkpoint: %s', FLAGS.checkpoint)
        checkpoint_manager2 = tf.train.CheckpointManager(tf.train.Checkpoint(model,model), directory=FLAGS.model_dir, max_to_keep=FLAGS.keep_checkpoint_max)
        checkpoint_manager2.checkpoint.restore(FLAGS.checkpoint).expect_patial()
        if FLAGS.zero_init_logits_layer:
            model = checkpoint_manager2.checkpoint.model
            ##Funcion del model.py?
            output_layer_parameters = model.supervised_head.trainable_weights
            logging.info('Initializing output layer params %s to zero', [x.op.name for x in output_layer_parameters])
            for x in output_layer_parameters:
                x.assign(tf.zeros_like(x))
    return checkpoint_manager