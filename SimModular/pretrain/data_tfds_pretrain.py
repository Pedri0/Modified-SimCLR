# coding=utf-8
# Copyright 2020 The SimCLR Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific simclr governing permissions and
# limitations under the License.
# ==============================================================================
"""Data pipeline."""

import functools
from absl import flags
from absl import logging
import tensorflow.compat.v2 as tf

import data_util_pretrain

FLAGS = flags.FLAGS


def build_input_fn(builder, global_batch_size):
  """Build input function.

  Args:
    builder: TFDS builder for specified dataset.
    global_batch_size: Global batch size.
  Returns:
    A function that accepts a dict of params and returns a tuple of images and
    features, to be used as the input_fn in TPUEstimator.
  """

  def _input_fn(input_context):
    """Inner input function."""
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    logging.info('Global batch size: %d', global_batch_size)
    logging.info('Per-replica batch size: %d', batch_size)
    preprocess_fn_pretrain = get_preprocess_fn(color_distortion=True)
    num_classes = builder.info.features['label'].num_classes

    def map_fn(image, label):
      """Produces multiple transformations of the same batch."""
      xs = []
      for _ in range(2):  # Two transformations
        xs.append(preprocess_fn_pretrain(image))
      image = tf.concat(xs, -1)
      label = tf.one_hot(label, num_classes)
      return image, label

    dataset = builder.as_dataset(
        split=FLAGS.train_split,
        shuffle_files=True,
        as_supervised=True)
    logging.info('num_input_pipelines: %d', input_context.num_input_pipelines)
    # The dataset is always sharded by number of hosts.
    # num_input_pipelines is the number of hosts rather than number of cores.
    if input_context.num_input_pipelines > 1:
      dataset = dataset.shard(input_context.num_input_pipelines,
                              input_context.input_pipeline_id)

    buffer_multiplier = 50 if FLAGS.image_size <= 32 else 10
    dataset = dataset.shuffle(batch_size * buffer_multiplier)
    dataset = dataset.repeat(-1)
    dataset = dataset.map(
        map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

  return _input_fn


def build_distributed_dataset(builder, batch_size, strategy):
  input_fn = build_input_fn(builder, batch_size)
  return strategy.experimental_distribute_datasets_from_function(input_fn)


def get_preprocess_fn(color_distortion):
  
  return functools.partial(
      data_util_pretrain.preprocess_image,
      height=FLAGS.image_size,
      width=FLAGS.image_size,
      color_distort=color_distortion)