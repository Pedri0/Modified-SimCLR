import functools
from absl import flags
from absl import logging
import tensorflow.compat.v2 as tf
import pandas as pd

import data_util_val

FLAGS = flags.FLAGS

def build_input_fn(global_batch_size):
    #Build input function.
    #Args:
    #    global_batch_size: Global batch size.
    #Returns:
    #    A function that accepts a dict of params and returns a tuple of images and
    #    features, to be used as the input_fn in TPUEstimator.
    def _input_fn_(input_context):
        #Inner input function
        batch_size = input_context.get_per_replica_batch_size(global_batch_size)
        preprocess_fn_val = get_preprocess_fn()
        num_classes = 5

        def map_fn(image, label):
            image = preprocess_fn_val(image)
            label = tf.one_hot(label, num_classes)
            return image, label

        dataset = get_data_validation()

        if input_context.num_input_pipelines > 1:
            dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)

        dataset = dataset.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    return _input_fn_

def build_distributed_dataset(batch_size, strategy):
    input_fn = build_input_fn(batch_size)
    return strategy.distribute_datasets_from_function(input_fn)

def get_preprocess_fn():
    #Get function that accepts an image and returns a preprocessed image
    return functools.partial(
        data_util_val.preprocess_image,
        height = FLAGS.image_size,
        width= FLAGS.image_size)


def get_data_validation():
    data_dir = 'imagenes_clasificadas_nair/'
    def read_images(image_file, label):
        image = tf.io.read_file(data_dir + image_file)
        image = tf.image.decode_jpeg(image, channels = 3)
        return image, label
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    df = pd.read_csv('nairValid.csv')
    file_paths = df['name'].values
    labels = df['new_class'].values
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    ds_valid = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds_valid = ds_valid.map(read_images, num_parallel_calls =AUTOTUNE)
    return ds_valid