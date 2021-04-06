from absl import app
from absl import flags
from absl import logging
import os
import math
import json
import galaxies_data as galax
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import data as data_lib
import data_astro
import metrics
import model as model_lib
import galaxies_data as galax


FLAGS = flags.FLAGS



def perform_evaluation(model, builder, eval_steps, ckpt, strategy, topology):
    #perform evaluation

    #check wich dataset to use and instanciate it
    if FLAGS.dataset == 'astro':
        print('Performing Evaluation on Astro Dataset')
        ds = data_astro.build_distributed_dataset(FLAGS.eval_batch_size, False, strategy)

    else:
        ds = data_lib.build_distributed_dataset(builder, FLAGS.eval_batch_size, False, strategy, topology)

    summary_writer = tf.summary.create_file_writer(FLAGS.model_dir)

    #Build metrics
    with strategy.scope():
        regularization_loss = tf.keras.metrics.Mean('eval/regularization_loss')
        label_top_1_accuracy = tf.keras.metrics.Accuracy('eval/label_top_1_accuracy')
        label_top_3_accuracy = tf.keras.metrics.TopKCategoricalAccuracy(3, 'eval/label_top_5_accuracy')
        all_metrics = [regularization_loss, label_top_1_accuracy, label_top_3_accuracy]

        #Restore checkpoint
        logging.info('Restoring from %s', ckpt)
        checkpoint = tf.train.Checkpoint(model=model, global_step=tf.Variable(0, dtype=tf.int64))
        checkpoint.restore(ckpt).expect_partial()
        global_step = checkpoint.global_step
        logging.info('Performing eval at step %d', global_step.numpy())

    def single_step(features, labels):
        _, supervised_head_outputs = model(features, training=False)
        assert supervised_head_outputs is not None
        outputs = supervised_head_outputs
        l = labels['labels']
        metrics.update_finetune_metrics_eval(label_top_1_accuracy, label_top_3_accuracy, outputs, l)
        reg_loss = model_lib.add_weight_decay(model, adjust_per_optimizer=True)
        regularization_loss.update_state(reg_loss)

    with strategy.scope():

        @tf.function
        def run_single_step(iterator):
            images, labels = next(iterator)
            features, labels = images, {'labels': labels}
            strategy.run(single_step, (features, labels))

        iterator = iter(ds)
        for i in range(eval_steps):
            run_single_step(iterator)
            logging.info('Completed eval for %d/%d steps', i+1, eval_steps)
        logging.info('Finished eval for %s', ckpt)
    
    #Write summaries
    cur_step = global_step.numpy()
    logging.info('Writing summaries for %d step', cur_step)
    with summary_writer.as_default():
        metrics.log_and_write_metrics_to_summary(all_metrics, cur_step)
        summary_writer.flush()

    #Record results as JSON.
    result_json_path = os.path.join(FLAGS.model_dir, 'result.json')
    result = {metric.name: metric.result().numpy() for metric in all_metrics}
    result['global_step'] = global_step.numpy()
    logging.info(result)
    with tf.io.gfile.GFile(result_json_path, 'w') as f:
        json.dump({k: float(v) for k, v in result.items()}, f)
    result_json_path = os.path.join(FLAGS.model_dir, 'result_%d.json'%result['global_step'])
    with tf.io.gfile.GFile(result_json_path, 'w') as f:
        json.dump({k: float(v) for k, v in result.items()}, f)
    flag_json_path = os.path.join(FLAGS.model_dir, 'flags.json')
    with tf.io.gfile.GFile(flag_json_path, 'w') as f:
        serializable_flags = {}
        for key, val in FLAGS.flag_values_dict().items():
            #Some flag value types e.g. datetime.timedelta are not json serializable,
            #filter those out
            if json_serializable(val):
                serializable_flags[key] = val
            
        json.dump(serializable_flags, f)

    if FLAGS.train_mode == 'finetune':
        save(model, global_step=result['global_step'])
    
    return result

def json_serializable(val):
    try:
        json.dumps(val)
        return True
    except TypeError:
        return False

def save(model, global_step):
    """Export as SavedModel for finetuning and inference."""
    saved_model = build_saved_model(model)
    export_dir = os.path.join(FLAGS.model_dir, 'saved_model')
    checkpoint_export_dir = os.path.join(export_dir, str(global_step))
    if tf.io.gfile.exist(checkpoint_export_dir):
        tf.io.gfile.rmtree(checkpoint_export_dir)
    tf.saved_model.save(saved_model, checkpoint_export_dir)

    if FLAGS.keep_hub_module_max > 0:
        #Delete old exported SavedModels
        exported_steps = []
        for subdir in tf.io.gfile.listdir(export_dir):
            if not subdir.isdigit():
                continue
            exported_steps.append(int(subdir))
        exported_steps.sort()
        for step_to_delete in exported_steps[:-FLAGS.keep_hub_module_max]:
            tf.io.gfile.rmtree(os.path.join(export_dir, str(step_to_delete)))

def build_saved_model(model):
    #Returns a tf.Module for saving to SavedModel

    class SimCLRModel(tf.Module):

        def __init__(self, model):
            self.model = model
            #this cant be called 'trainable_variables' because 'tf.module' has a getter with the same name
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
    #returns a dictionary of tensors
    graph = tf.compat.v1.get_default_graph()
    result = {}
    for i in range(1, 5):
        result['block_group%d' % i] = graph.get_tensor_by_name('resnet/block_group%d/block_group%d:0' % (i,i))
    result['initial_conv'] = graph.get_tensor_by_name('resnet/initial_conv/Identity:0')
    result['initial_max_pool'] = graph.get_tensor_by_name('resnet/initial_max_pool/Identity:0')
    result['final_avg_pool'] = graph.get_tensor_by_name('resnet/final_avg_pool:0')
    result['logits_sup'] = graph.get_tensor_by_name('head_supervised/logits_sup:0')

    return result

def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments')

    if FLAGS.dataset == 'astro':
        print('Using astro dataset')
        builder = None
        num_train_examples = galax.get_number_of_images('galaxies_train.csv')
        num_eval_examples = galax.get_number_of_images('nair_unbalanced_valid.csv')
        num_classes = 5

    else:
        print('Using one dataset from TensorFlow Datasets')
        builder = tfds.builder(FLAGS.dataset, data_dir=FLAGS.data_dir)
        builder.download_and_prepare()
        num_train_examples = builder.info.splits[FLAGS.train_split].num_examples
        num_eval_examples = builder.info.splits[FLAGS.eval_split].num_examples
        num_classes = builder.info.features['label'].num_classes

    train_steps = model_lib.get_train_steps(num_train_examples)
    eval_steps = FLAGS.eval_steps or int(math.ceil(num_eval_examples / FLAGS.eval_batch_size))
    epoch_steps = int(round(num_train_examples / FLAGS.train_batch_size))

    logging.info('# train examples: %d', num_train_examples)
    logging.info('# train_steps: %d', train_steps)
    logging.info('# eval examples: %d', num_eval_examples)
    logging.info('# eval steps: %d', eval_steps)

    checkpoint_steps = (FLAGS.checkpoint_steps or (FLAGS.checkpoint_epochs * epoch_steps))
    topology = None

    strategy = tf.distribute.MirroredStrategy()
    logging.info('Running using MirroredStrategy on %d replicas', strategy.num_replicas_in_sync)

    #instanciating model
    with strategy.scope():
        model = model_lib.Model(num_classes)
    
    for ckpt in tf.train.checkpoints_iterator(FLAGS.model_dir, min_interval_secs = 15):
        result = perform_evaluation(model, builder, eval_steps, ckpt, strategy, topology)

        if result['global_step'] >= train_steps:
            logging.info('Eval complete. Exiting')
            return

if __name__ == '__main__':
    tf.compat.v1.enable_v2_behavior()
    #necessary for GPU // For outside compilation of summaries on TPU.
    tf.config.set_soft_device_placement(True)
    app.run(main)