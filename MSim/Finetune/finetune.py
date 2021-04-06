from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v2 as tf
import pandas as pd
import math

import data_fine as data_lib
import model_fine as model_lib
import restore_checkpoint
import data_val

FLAGS = flags.FLAGS

############ Used in restore_checkpoint
flags.DEFINE_string('model_dir', None, 'Model directory for training.')
flags.DEFINE_integer('keep_checkpoint_max', 1, 'Maximum number of checkpoints to keep.')
flags.DEFINE_string('checkpoint', None, 'Loading from the given checkpoint for fine-tuning if a finetuning checkpoint does not already exist in model_dir.')
flags.DEFINE_bool('zero_init_logits_layer', True, 'If True, zero initialize layers after avg_pool for supervised learning.')
flags.DEFINE_integer('keep_hub_module_max', 1,'Maximum number of Hub modules to keep.')

############ Used in resnet_fine
flags.DEFINE_boolean('global_bn', True, 'Whether to aggregate BN statistics across distributed cores.')
flags.DEFINE_float('batch_norm_decay', 0.9, 'Batch norm decay parameter.')
flags.DEFINE_integer('fine_tune_after_block', -1, 'The layers after which block that we will fine-tune. -1 means fine-tuning everything.'
        '0 means fine-tuning after stem block. 4 means fine-tuning just the linear head.')

############ Used in model_fine
flags.DEFINE_enum('optimizer', 'lars', ['momentum', 'adam', 'lars', 'novograd'], 'Optimizer to use.')
flags.DEFINE_float('momentum', 0.9, 'Momentum parameter.')
flags.DEFINE_float('weight_decay', 0, 'Amount of weight decay to use.')
flags.DEFINE_float('warmup_epochs', 0, 'Number of epochs of warmup.')
flags.DEFINE_integer('train_batch_size', 64, 'Batch size for training.')
flags.DEFINE_integer('train_epochs', 90, 'Number of epochs to train for.')
flags.DEFINE_integer('num_proj_layers', 3, 'Number of non-linear head layers.')
flags.DEFINE_integer('proj_out_dim', 128, 'Number of head projection dimension.')
flags.DEFINE_integer('ft_proj_selector', 0, 'Which layer of the projection head to use during fine-tuning. '
    '0 means no projection head, and -1 means the final layer.')
flags.DEFINE_integer('resnet_depth', 50, 'Depth of ResNet.')
flags.DEFINE_integer('image_size', 330, 'Input image size.')

############  Used in pretrain (here)
flags.DEFINE_integer('checkpoint_epochs', 1, 'Number of epochs between checkpoints/summaries.')
flags.DEFINE_float('learning_rate', 0.002, 'Initial learning rate per batch size of 256.')

def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments')

    num_train_examples = data_lib.get_number_of_images('nair_unbalanced_fine.csv')
    num_classes = 5

    train_steps = num_train_examples * FLAGS.train_epochs // FLAGS.train_batch_size + 1
    epoch_steps = int(round(num_train_examples / FLAGS.train_batch_size))

    logging.info('# train examples: %d', num_train_examples)
    logging.info('# train_steps: %d', train_steps)

    checkpoint_steps = (FLAGS.checkpoint_epochs * epoch_steps)

    strategy = tf.distribute.MirroredStrategy()
    logging.info('Running using MirroredStrategy on %d replicas', strategy.num_replicas_in_sync)

    #instanciating model
    with strategy.scope():
        model = model_lib.Model(num_classes)
        #build input pipeline
        ds = data_lib.build_distributed_dataset(FLAGS.train_batch_size, strategy)
        ds_val = data_val.build_distributed_dataset(4, strategy)
        
        #Build LR schedule and optimizer
        learning_rate = model_lib.WarmUpAndCosineDecay(FLAGS.learning_rate, num_train_examples)
        optimizer = model_lib.build_optimizer(learning_rate)

        #Build Metrics for finetuning
        all_metrics = []
        weight_decay_metric = tf.keras.metrics.Mean('train/weight_decay')
        total_loss_metric = tf.keras.metrics.Mean('train/total_loss')
        supervised_loss_metric = tf.keras.metrics.Mean('train/supervised_loss')
        supervised_acc_metric = tf.keras.metrics.Mean('train/supervised_acc')
        all_metrics.extend([weight_decay_metric, total_loss_metric,
            supervised_loss_metric, supervised_acc_metric])

        #Build list for store history of metrics
        df = pd.DataFrame(columns=['Epoch', 'Step', 'WeightDecay', 'TotalLoss', 'SupervisedLoss',
                'SupervisedAccuracy', 'ValTop2', 'ValTop1'])

        #Restore checkpoint if avalaible
        checkpoint_manager = restore_checkpoint.try_restore_from_checkpoint(model, optimizer.iterations, optimizer)
        #Create new checkpoint manager to save best model weights
        checkpoint =  tf.train.Checkpoint(optimizer=optimizer, global_step=optimizer.iterations, model=model)
        checkpoint_manager2 = tf.train.CheckpointManager(checkpoint, directory=FLAGS.model_dir+'/best', max_to_keep=1)

        @tf.function
        def train_multiple_steps(iterator):
            # `tf.range` is needed so that this runs in a `tf.while_loop` and is not unrolled
            for _ in tf.range(checkpoint_steps):
                with tf.name_scope(''):
                    images, labels = next(iterator)
                    features, labels = images, {'labels': labels}
                    strategy.run(single_step, (features, model, labels, optimizer, strategy, supervised_loss_metric,
                        supervised_acc_metric, weight_decay_metric, total_loss_metric))
        
        epoch = 0
        best_eval_acc = 0
        global_step = optimizer.iterations
        cur_step = global_step.numpy()
        iterator = iter(ds)

        while cur_step < train_steps:
            train_multiple_steps(iterator)
            cur_step = global_step.numpy()
            checkpoint_manager.save(cur_step)
            epoch = epoch + 1
            #save train statics in local variables
            weight_decay = weight_decay_metric.result().numpy()
            total_loss = total_loss_metric.result().numpy()
            sup_loss = supervised_loss_metric.result().numpy()
            sup_acc = supervised_acc_metric.result().numpy() * 100
            logging.info('Completed: %d/%d steps. Epoch: %d. Sup Acc: %0.2f. Total Loss: %0.2f',
                    cur_step, train_steps, epoch, sup_acc, total_loss)

            top1, top2 = validation(strategy, model, ds_val)
            logging.info('Validation at step: %d. Top-2: %0.2f. Top-1: %0.2f.',
                    cur_step, top2, top1)

            d_list = [epoch, cur_step, weight_decay, total_loss, sup_loss, sup_acc, top2, top1]
            df.loc[len(df), :] = d_list

            if top1 > best_eval_acc:
                best_eval_acc = top1
                checkpoint_manager2.save(cur_step)
                nc = cur_step
            
            for metric in all_metrics:
                metric.reset_states()

        logging.info('Training complete :)')
        df.to_csv('finetune_{}_Epoch_lr{}.csv'.format(epoch, FLAGS.learning_rate), index=False)

    save_best_model(model, nc, strategy)

@tf.function
def single_step(features, model, labels, optimizer, strategy, supervised_loss_metric,
                supervised_acc_metric, weight_decay_metric, total_loss_metric):

    with tf.GradientTape() as tape:
        loss = None
        supervised_head_outputs = model(features, training = True)
        l = labels['labels']
        sup_loss = tf.reduce_mean(
            tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(l, supervised_head_outputs))
        
        if loss is None:
            loss = sup_loss
        else:
            loss += sup_loss

        ############ Update metrics #######################
        supervised_loss_metric.update_state(sup_loss)

        label_acc = tf.equal(tf.argmax(l, 1), tf.argmax(supervised_head_outputs, axis=1))
        label_acc = tf.reduce_mean(tf.cast(label_acc, tf.float32))
        supervised_acc_metric.update_state(label_acc)
        ##################################################

        weight_decay = model_lib.add_weight_decay(model)
        weight_decay_metric.update_state(weight_decay)
        loss += weight_decay
        total_loss_metric.update_state(loss)
        loss = loss / strategy.num_replicas_in_sync
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))


def validation(strategy, model, ds_val):
    #ds_val = data_val.build_distributed_dataset(4, strategy)

    with strategy.scope():
        top_1_acc = tf.keras.metrics.Accuracy('valid/top_1_acc')
        top_2_acc = tf.keras.metrics.TopKCategoricalAccuracy(2, 'valid/top_2_acc')
    
    @tf.function
    def single_step(features, labels, model):
        supervised_head_outputs = model(features, training=False)
        l = labels['labels']
        top_1_acc.update_state(tf.argmax(l, 1), tf.argmax(supervised_head_outputs, axis=1))
        top_2_acc.update_state(l, supervised_head_outputs)
    
    with strategy.scope():

        @tf.function
        def run_single_step(img, lbl, model):
            features, labels = img, {'labels': lbl}
            strategy.run(single_step, (features, labels, model))

        for img, lbl in ds_val:
            run_single_step(img, lbl, model)
        
        top_1_acc = top_1_acc.result().numpy()
        top_2_acc = top_2_acc.result().numpy()

    return top_1_acc*100, top_2_acc*100

def save_best_model(model, step, strategy):
    with strategy.scope():
        checkpoint = tf.train.Checkpoint(model=model, global_step=tf.Variable(0, dtype=tf.int64))
        checkpoint.restore(FLAGS.model_dir+'/best/ckpt-'+str(step)).expect_partial()

    restore_checkpoint.save(model, global_step=step)

if __name__ == '__main__':
    tf.compat.v1.enable_v2_behavior()
    #necessary for GPU // For outside compilation of summaries on TPU.
    tf.config.set_soft_device_placement(True)
    app.run(main)