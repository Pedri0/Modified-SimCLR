from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

import data_tfds_pretrain as data_lib
import data_astro_pretrain
import metrics
import model_pretrain as model_lib
import objective as obj_lib
import galaxies_data_pretrain as galax

######new .py's #####
import restore_checkpoint


FLAGS = flags.FLAGS


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments')

    
    builder = tfds.builder(FLAGS.dataset, data_dir=FLAGS.data_dir)
    builder.download_and_prepare()
    num_train_examples = builder.info.splits[FLAGS.train_split].num_examples
    num_eval_examples = builder.info.splits[FLAGS.eval_split].num_examples
    num_classes = builder.info.features['label'].num_classes

    train_steps = FLAGS.train_steps or (num_train_examples * FLAGS.train_epochs // FLAGS.train_batch_size + 1)
    epoch_steps = int(round(num_train_examples / FLAGS.train_batch_size))

    logging.info('# train examples: %d', num_train_examples)
    logging.info('# train_steps: %d', train_steps)

    checkpoint_steps = (FLAGS.checkpoint_steps or (FLAGS.checkpoint_epochs * epoch_steps))
    

    #instanciating model
    model = model_lib.Model(num_classes)
    
    #"write" model in path model_dir
    summary_writer = tf.summary.create_file_writer(FLAGS.model_dir)

    #build input pipeline

    ds = data_lib.build_input_fn(builder, FLAGS.train_batch_size)
    
    #Build LR schedule and optimizer
    learning_rate = model_lib.WarmUpAndCosineDecay(FLAGS.learning_rate, num_train_examples)
    optimizer = model_lib.build_optimizer(learning_rate)

    #Build Metrics for pretrain
    all_metrics = []
    weight_decay_metric = tf.keras.metrics.Mean('train/weight_decay')
    total_loss_metric = tf.keras.metrics.Mean('train/total_loss')
    contrast_loss_metric = tf.keras.metrics.Mean('train/contrast_loss')
    contrast_acc_metric = tf.keras.metrics.Mean('train/contrast_acc')
    contrast_entropy_metric = tf.keras.metrics.Mean('train/contrast_entropy')
    all_metrics.extend([weight_decay_metric, total_loss_metric,
        contrast_loss_metric, contrast_acc_metric, contrast_entropy_metric])

    #Restore checkpoint if avalaible
    checkpoint_manager = restore_checkpoint.try_restore_from_checkpoint(model, optimizer.iterations, optimizer)

    def single_step(features, labels):
        with tf.GradientTape() as tape:
            # Log summaries on the last step of the training loop to match logging frequency of other scalar summaries.
            # Notes:
            # 1. Summary ops on TPUs get outside compiled so they do not affect performance
            # 2. Summaries are recorded only on replica 0. So effectively this
            #    summary would be written once per host when should_record == True.
            # 3. optimizer.iterations is incremented in the call to apply_gradients.
            #    So we use  `iterations + 1` here so that the step number matches those of scalar summaries
            # 4. We intentionally run the summary op before the actual model training so that it can run in parallel.
            should_record = tf.equal((optimizer.iterations + 1) % checkpoint_steps, 0)
            with tf.summary.record_if(should_record):
                #only log augmented images for the first tower.
                tf.summary.image('image', features[:, :, :, :3], step=optimizer.iterations+1)

            #get only projection_head_outputs since this is only for pretrain
            projection_head_outputs = model(features, training=True)
            loss = None

            outputs = projection_head_outputs
            con_loss, logits_con, labels_con = obj_lib.add_contrastive_loss(
                outputs, hidden_norm=FLAGS.hidden_norm, temperature=FLAGS.temperature)
            #first epoch
            if loss is None:
                loss = con_loss
            #not first epoch
            else:
                loss += con_loss

            metrics.update_pretrain_metrics_train(
                contrast_loss_metric, contrast_acc_metric, contrast_entropy_metric,con_loss,logits_con,labels_con)
            weight_decay = model_lib.add_weight_decay(model)
            weight_decay_metric.update_state(weight_decay)
            loss += weight_decay
            total_loss_metric.update_state(loss)
            #The default behaviour of 'apply_gradients' is to sum gradients from all replicas so we divide the loss
            #by te number of replicas so that the mean gradient is applied
            grads= tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

    with strategy.scope():
        @tf.function
        def train_multiple_steps(iterator):
            # `tf.range` is needed so that this runs in a `tf.while_loop` and is not unrolled
            for _ in tf.range(checkpoint_steps):
                with tf.name_scope(''):
                    images, labels = next(iterator)
                    features, labels = images, {'labels': labels}
                    strategy.run(single_step, (features, labels))
        
        global_step = optimizer.iterations
        cur_step = global_step.numpy()
        iterator = iter(ds)

        while cur_step < train_steps:
           #Calls to tf.summary .xyz lookup the summary writer resource wich is set by the summary writer's context manager
            with summary_writer.as_default():
                train_multiple_steps(iterator)
                cur_step = global_step.numpy()
                checkpoint_manager.save(cur_step)
                logging.info('Completed: %d / %d steps', cur_step, train_steps)
                metrics.log_and_write_metrics_to_summary(all_metrics, cur_step)
                tf.summary.scalar('learning_rate', learning_rate(tf.cast(global_step, dtype=tf.float32)),global_step)
                summary_writer.flush()

            for metric in all_metrics:
                metric.reset_states()
        
        logging.info('Training complete :)')


if __name__ == '__main__':
    tf.compat.v1.enable_v2_behavior()
    #necessary for GPU // For outside compilation of summaries on TPU.
    tf.config.set_soft_device_placement(True)
    app.run(main)