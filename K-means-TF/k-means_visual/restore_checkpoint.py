from absl import flags
from absl import logging
import os
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
        checkpoint_manager2 = tf.train.CheckpointManager(tf.train.Checkpoint(model=model), directory=FLAGS.model_dir, max_to_keep=FLAGS.keep_checkpoint_max)
        checkpoint_manager2.checkpoint.restore(FLAGS.checkpoint).expect_partial()

        if FLAGS.zero_init_logits_layer:
            model = checkpoint_manager2.checkpoint.model
            output_layer_parameters = model.supervised_head.trainable_weights
            logging.info('Initializing output layer parameters %s to zero',
                    [x.op.name for x in output_layer_parameters])
            for x in output_layer_parameters:
                x.assign(tf.zeros_like(x))

    return checkpoint_manager


def save(model, global_step):
    #Export as SavedModel for finetuning and inference.
    saved_model = build_saved_model(model)
    export_dir = os.path.join(FLAGS.model_dir, 'saved_model')
    checkpoint_export_dir = os.path.join(export_dir, str(global_step))
    if tf.io.gfile.exists(checkpoint_export_dir):
        tf.io.gfile.rmtree(checkpoint_export_dir)
    tf.saved_model.save(saved_model, checkpoint_export_dir)

    if FLAGS.keep_hub_module_max > 0:
        # Delete old exported SavedModels.
        exported_steps = []
        for subdir in tf.io.gfile.listdir(export_dir):
            if not subdir.isdigit():
                continue
            exported_steps.append(int(subdir))
        exported_steps.sort()
        for step_to_delete in exported_steps[:-FLAGS.keep_hub_module_max]:
            tf.io.gfile.rmtree(os.path.join(export_dir, str(step_to_delete)))


def build_saved_model(model):
    #Returns a tf.Module for saving to SavedModel.
    class SimCLRModel(tf.Module):
       #Saved model for exporting to hub.

        def __init__(self, model):
            self.model = model
            # This can't be called `trainable_variables` because `tf.Module` has a getter with the same name.
            self.trainable_variables_list = model.trainable_variables
        
        @tf.function
        def __call__(self, inputs, trainable):
            self.model(inputs, training=trainable)
            return get_salient_tensors_dict()

    module = SimCLRModel(model)
    input_spec = tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.float32)
    module.__call__.get_concrete_function(input_spec, trainable=True)
    module.__call__.get_concrete_function(input_spec, trainable=False)
    return module


def get_salient_tensors_dict():
    #Returns a dictionary of tensors.
    graph = tf.compat.v1.get_default_graph()
    result = {}
    for i in range(1, 5):
        result['block_group%d' % i] = graph.get_tensor_by_name('resnet/block_group%d/block_group%d:0' % (i, i))
    
    result['initial_conv'] = graph.get_tensor_by_name('resnet/initial_conv/Identity:0')
    result['initial_max_pool'] = graph.get_tensor_by_name('resnet/initial_max_pool/Identity:0')
    result['final_avg_pool'] = graph.get_tensor_by_name('resnet/final_avg_pool:0')
    #result['logits_sup'] = graph.get_tensor_by_name('head_supervised/logits_sup:0')

    return result