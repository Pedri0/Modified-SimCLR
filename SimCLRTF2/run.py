import json
import math
import os

from absl import app
from absl import flags
from absl import logging

import data as data_lib
import metrics
import model as model_lib
import objective as obj_lib

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

FLAGS = flags.FLAGS
##########################Flags##################################
flags.DEFINE_float('learning_rate', 0.3, 'Initial learning rate per batch size of 256.')

#changed linear (default) to sqrt because our batch_size is not to big (16) 
flags.DEFINE_enum('learning_rate_scaling', 'sqrt', ['linear', 'sqrt'], 'How to scale the learning rate as a function of batch size.')

flags.DEFINE_float('warmup_epochs', 10, 'Number of epochs of warmup.')

flags.DEFINE_float('weight_decay', 1e-6, 'Amount of weight decay to use.')

flags.DEFINE_float('batch_norm_decay', 0.9, 'Batch norm decay parameter.')

#changed 512 (default) to 16 because our hardware
flags.DEFINE_integer('train_batch_size', 16, 'Batch size for training.')

flags.DEFINE_string('train_split', 'train', 'Split for training.')

flags.DEFINE_integer('train_epochs', 100, 'Number of epochs to train for.')

flags.DEFINE_integer('train_steps', 0, 'Number of steps to train for. If provided, overrides train_epochs.')

flags.DEFINE_integer('eval_steps', 0, 'Number of steps to eval for. If not provided, evals over entire dataset.')

flags.DEFINE_integer('eval_batch_size', 256, 'Batch size for eval.')

flags.DEFINE_integer('checkpoint_epochs', 1, 'Number of epochs between checkpoints/summaries.')

flags.DEFINE_integer('checkpoint_steps', 0, 'Number of steps between checkpoints/summaries. If provided, overrides checkpoint_epochs.')

flags.DEFINE_string('eval_split', 'validation', 'Split for evaluation.')

#changed 'imagenet2012' (default) to 'galaxies' because we'll not use imagenet2012 but its supported
flags.DEFINE_string('dataset', 'galaxies', 'Name of a dataset.')

flags.DEFINE_enum('mode', 'train', ['train', 'eval', 'train_then_eval'],
    'Whether to perform training or evaluation.')

flags.DEFINE_enum('train_mode', 'pretrain', ['pretrain', 'finetune'],
    'The train mode controls different objectives and trainable components.')

#changed True (default) to False
flags.DEFINE_bool('lineareval_while_pretraining', False, 'Whether to finetune supervised head while pretraining.')

flags.DEFINE_string('checkpoint', None, 'Loading from the given checkpoint for fine-tuning if a finetuning checkpoint does not already exist in model_dir.')

flags.DEFINE_bool('zero_init_logits_layer', False, 'If True, zero initialize layers after avg_pool for supervised learning.')

flags.DEFINE_integer('fine_tune_after_block', -1, 'The layers after which block that we will fine-tune. -1 means fine-tuning'
    'everything. 0 means fine-tuning after stem block. 4 means fine-tuning '
    'just the linear head.')

flags.DEFINE_string('model_dir', None, 'Model directory for training.')

flags.DEFINE_string('data_dir', None, 'Directory where dataset is stored.')

flags.DEFINE_enum('optimizer', 'lars', ['momentum', 'adam', 'lars'], 'Optimizer to use.')

flags.DEFINE_float('momentum', 0.9, 'Momentum parameter.')

flags.DEFINE_string('eval_name', None, 'Name for eval.')

#changed 5 (default) to 20 just for save
flags.DEFINE_integer('keep_checkpoint_max', 20, 'Maximum number of checkpoints to keep.')

flags.DEFINE_integer('keep_hub_module_max', 1, 'Maximum number of Hub modules to keep.')

flags.DEFINE_float('temperature', 0.1, 'Temperature parameter for contrastive loss.')

flags.DEFINE_enum('proj_head_mode', 'nonlinear', ['none', 'linear', 'nonlinear'], 'How the head projection is done.')

flags.DEFINE_integer('proj_out_dim', 128, 'Number of head projection dimension.')

flags.DEFINE_integer('num_proj_layers', 3, 'Number of non-linear head layers.')

flags.DEFINE_integer('ft_proj_selector', 0, 'Which layer of the projection head to use during fine-tuning. '
    '0 means no projection head, and -1 means the final layer.')

flags.DEFINE_boolean('global_bn', True, 'Whether to aggregate BN statistics across distributed cores.')

flags.DEFINE_integer('width_multiplier', 1, 'Multiplier to change width of network.')

flags.DEFINE_integer('resnet_depth', 50, 'Depth of ResNet.')

flags.DEFINE_float('sk_ratio', 0., 'If it is bigger than 0, it will enable SK. Recommendation: 0.0625.')

flags.DEFINE_float('se_ratio', 0., 'If it is bigger than 0, it will enable SE.')

#changed 224 (default) to 330 because our experiments
flags.DEFINE_integer('image_size', 330, 'Input image size.')

flags.DEFINE_float('color_jitter_strength', 1.0, 'The strength of color jittering.')

flags.DEFINE_boolean('use_blur', True, 'Whether or not to use Gaussian blur for augmentation during pretraining.')
##########################End Flags##################################

def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments')

    if FLAGS.dataset == 'astro':
        print('Using astro dataset')
        builder = None
        

    else:
        builder = tfds.builder(FLAGS.dataset, data_dir=FLAGS.data_dir)
        builder.download_and_prepare()
        num_train_examples = builder.info.splits[FLAGS.train_split].num_examples
        num_eval_examples = builder.info.splits[FLAGS.eval_split].num_examples
        num_classes = builder.info.features['label'].num_classes


if __name__ == '__main__':
    tf.compat.v1.enable_v2_bahavior()
    #necessary for GPU // For outside compilation of summaries on TPU.
    tf.config.set_soft_device_placement(True)
    app.run(main)