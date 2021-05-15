from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v2 as tf
import pandas as pd
import numpy as np

import data_pretrain as data_lib
import model_pretrain as model_lib
import restore_checkpoint
import contrastive_loss
from sklearn.cluster import MiniBatchKMeans

FLAGS = flags.FLAGS
########################## Flags ##################################

############ Used in restore_checkpoint
flags.DEFINE_string('model_dir', None, 'Model directory for training.')
flags.DEFINE_integer('keep_checkpoint_max', 2, 'Maximum number of checkpoints to keep.')
flags.DEFINE_string('checkpoint', None, 'Loading from the given checkpoint for fine-tuning if a finetuning checkpoint does not already exist in model_dir.')
flags.DEFINE_bool('zero_init_logits_layer', False, 'If True, zero initialize layers after avg_pool for supervised learning.')

############ Used in resnet_pretrain
flags.DEFINE_boolean('global_bn', True, 'Whether to aggregate BN statistics across distributed cores.')
flags.DEFINE_float('batch_norm_decay', 0.9, 'Batch norm decay parameter.')

############ Used in model_pretrain
flags.DEFINE_enum('optimizer', 'lars', ['momentum', 'adam', 'lars', 'novograd'], 'Optimizer to use.')
flags.DEFINE_float('momentum', 0.9, 'Momentum parameter.')
flags.DEFINE_float('weight_decay', 1e-4, 'Amount of weight decay to use.')
flags.DEFINE_float('warmup_epochs', 5, 'Number of epochs of warmup.')
flags.DEFINE_integer('train_batch_size', 128, 'Batch size for training.')
flags.DEFINE_integer('train_epochs', 100, 'Number of epochs to train for.')
flags.DEFINE_integer('num_proj_layers', 3, 'Number of non-linear head layers.')
flags.DEFINE_integer('proj_out_dim', 128, 'Number of head projection dimension.')
flags.DEFINE_integer('ft_proj_selector', 0, 'Which layer of the projection head to use during fine-tuning. '
    '0 means no projection head, and -1 means the final layer.')
flags.DEFINE_integer('resnet_depth', 50, 'Depth of ResNet.')
flags.DEFINE_boolean('use_blur', True, 'Whether or not to use Gaussian blur for augmentation during pretraining.')
flags.DEFINE_integer('image_size', 330, 'Input image size.')

############ Used in data_util
flags.DEFINE_float('color_jitter_strength', 1.0, 'The strength of color jittering.')

############  Used in pretrain (here)
flags.DEFINE_integer('checkpoint_epochs', 1, 'Number of epochs between checkpoints/summaries.')
flags.DEFINE_float('learning_rate', 0.075, 'Initial learning rate per batch size of 256.')
flags.DEFINE_boolean('hidden_norm', True, 'Temperature parameter for contrastive loss.')
flags.DEFINE_float('temperature', 0.1, 'Temperature parameter for contrastive loss.')
######################## End Flags ################################

def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments')

    num_train_examples = data_lib.get_number_of_images('nair_unbalanced_train.csv')

    train_steps = num_train_examples * FLAGS.train_epochs // FLAGS.train_batch_size + 1
    epoch_steps = int(round(num_train_examples / FLAGS.train_batch_size))

    logging.info('# train examples: %d', num_train_examples)
    logging.info('# train_steps: %d', train_steps)

    checkpoint_steps = (FLAGS.checkpoint_epochs * epoch_steps)
    
    strategy = tf.distribute.MirroredStrategy()
    logging.info('Running using MirroredStrategy on %d replicas', strategy.num_replicas_in_sync)

    def single_step(images, labels):

        with tf.GradientTape() as tape:
            loss = None
            projection_head_outputs, representations = model(images, training = True)

            con_loss, logits_con, labels_con = contrastive_loss.contrastive_loss(projection_head_outputs, hidden_norm=FLAGS.hidden_norm,
                temperature=FLAGS.temperature, strategy = strategy)

            if loss is None:
                loss = con_loss
            else:
                loss += con_loss

            ############ Update metrics #######################
            contrast_loss_metric.update_state(con_loss)

            contrast_acc_val = tf.equal(tf.argmax(labels_con, 1), tf.argmax(logits_con, axis=1))
            contrast_acc_val = tf.reduce_mean(tf.cast(contrast_acc_val, tf.float32))
            contrast_acc_metric.update_state(contrast_acc_val)

            prob_con = tf.nn.softmax(logits_con)
            entropy_con = -tf.reduce_mean(tf.reduce_sum(prob_con * tf.math.log(prob_con + 1e-8), -1))
            contrast_entropy_metric.update_state(entropy_con)
            ##################################################

            weight_decay = model_lib.add_weight_decay(model)
            weight_decay_metric.update_state(weight_decay)
            loss += weight_decay
            total_loss_metric.update_state(loss)
            loss = loss / strategy.num_replicas_in_sync
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        return tf.identity(representations)

    def get_new_balanced_ds(pseudolabels, csv_file):
        #calculate the size of the clusters
        clusters = [np.where(pseudolabels==c)[0].shape[0] for c in range(5)]
        #get the maximum size
        clusters = max(clusters)
        
        #convert pseudolabels numpy array to pandas dataframe
        pseudolabels = pd.DataFrame(pseudolabels)
        #read last csv file into pandas dataframe
        data = pd.read_csv(csv_file)
        #join data dataframe with pseudolabels dataframe
        data['pseudolabels'] = pseudolabels

        #do a list from original data dataframe s.t. each element in
        #the list belong to one unique pseudolabel
        auxiliary_df = [data[data['pseudolabels']==s].reset_index(drop=True) for s in range(5)]

        #make a new list which will store the balanced dataset
        balanced_datasets = []

        #Iterate over each element in auxiliary_df
        for dataframe in auxiliary_df:
            #get the proportion size between max cluster size and the actual cluster
            ratio = clusters/dataframe.shape[0]
            #duplicate the rows of datame according to the proportion size
            new_df = dataframe.loc[dataframe.index.repeat(1 if ratio==1.0 else round(ratio)+1)]
            #shuffle the new dataframe and get only n=maxcluster size samples
            new_df = new_df.sample(n=clusters if clusters<2560 else 2560).reset_index(drop=True)
            #append new_df to balanced_datasets list
            balanced_datasets.append(new_df)
        
        #delete auxiliary_df for memory reasons
        del auxiliary_df
        #concatenate every dataframe in balanced_datasets into a single one
        balanced_datasets = pd.concat(balanced_datasets).reset_index(drop=True)
        #now shuffle the dataset
        balanced_datasets = balanced_datasets.sample(frac=1).reset_index(drop=True)
        name = 'new_balanced_dataset.csv'
        #save the dataset
        balanced_datasets.to_csv(name, index=False)

        return name, clusters

    #instanciating model
    with strategy.scope():
        model = model_lib.Model()

        #build input pipeline
        #ds = data_lib.build_distributed_dataset(FLAGS.train_batch_size, strategy)

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
        #Build list for store history of metrics
        df = pd.DataFrame(columns=['Epoch', 'WeightDecay', 'TotalLoss', 'ContrastiveLoss',
                'ContrastiveEntropy', 'ContrastiveAccuracy'])

        #Restore checkpoint if avalaible
        checkpoint_manager = restore_checkpoint.try_restore_from_checkpoint(model, optimizer.iterations, optimizer)

        @tf.function
        def distributed_train(images, labels):
            return strategy.run(single_step, args=(images, labels))

        global_step = optimizer.iterations
        cur_step = global_step.numpy()
        EPOCHS = FLAGS.train_epochs
        start_epoch = 0
        if cur_step != 0:
            checkpoints_names = checkpoint_manager.checkpoints
            last_checkpoint_name = checkpoints_names[-1]
            start_epoch = [int(word) for word in last_checkpoint_name if word.isdigit()]
            start_epoch = start_epoch[0]

        if start_epoch == 0:
            name = 'nair_unbalanced_train.csv'
        else:
            name = 'new_balanced_dataset.csv'

        for epoch in range(start_epoch, EPOCHS):
            #build input pipeline
            ds = data_lib.build_distributed_dataset(FLAGS.train_batch_size, strategy, name)
            mini_kmeans = MiniBatchKMeans(n_clusters=5)
            pseudo_labels_per_epoch = []
            for image, label in ds:
                tensor = distributed_train(image, label)
                mini_kmeans.partial_fit(tensor.numpy())
                pseudo_labels_per_epoch.append(mini_kmeans.labels_)
                #print('minibatch size--> {}'.format(image.shape))
            pseudo_labels_per_epoch = np.concatenate(np.array(pseudo_labels_per_epoch, dtype=object), axis=0)
            name, cluster_size = get_new_balanced_ds(pseudo_labels_per_epoch, name)
            
            checkpoint_manager.save(epoch)
            weight_decay = weight_decay_metric.result().numpy()
            total_loss = total_loss_metric.result().numpy()
            contrast_loss = contrast_loss_metric.result().numpy()
            contrast_acc = contrast_acc_metric.result().numpy() * 100
            contrast_entropy = contrast_entropy_metric.result().numpy()
            logging.info('Completed Epoch: %d. Cluster Size: %d Contrastive Acc: %0.2f. Total Loss: %0.2f'
            ,epoch+1,cluster_size, contrast_acc,total_loss)
            d_list = [epoch, weight_decay, total_loss, contrast_loss, contrast_entropy, contrast_acc]
            df.loc[len(df), :] = d_list
            #reset metrics
            for metric in all_metrics:
                metric.reset_states()
        logging.info('Training complete :)')
        df.to_csv('pretrain_{}_Epochs_lr{}'.format(EPOCHS, FLAGS.learning_rate), index=False)


if __name__ == '__main__':
    tf.compat.v1.enable_v2_behavior()
    app.run(main)