import functools
from absl import flags
from absl import logging
import tensorflow.compat.v2 as tf
import pandas as pd

import data_util_pretrain
FLAGS = flags.FLAGS

def bulid_input_fn(global_batch_size, name):
    #Build input function.
    #Args:
    #    global_batch_size: Global batch size.
    #Returns:
    #    A function that accepts a dict of params and returns a tuple of images and
    #    features, to be used as the input_fn in TPUEstimator.
    def _input_fn_(input_context):
        #Inner input function
        batch_size = input_context.get_per_replica_batch_size(global_batch_size)
        logging.info('Global batch size: %d', global_batch_size)
        logging.info('Per-replica batch size: %d', batch_size)
        preprocess_fn_pretrain = get_preprocess_fn(color_distortion=True)
        num_classes = 5

        def map_fn(image, label):
            #Produces multiple transformations of the same batch for pretraining
            xs = []
            for _ in range(2):  # Two transformations
                xs.append(preprocess_fn_pretrain(image))
            image = tf.concat(xs, -1)
            label = tf.one_hot(label, num_classes)
            return image, label

        #logging.info('Using Astro pretrain data')
        dataset = get_data_pretrain(name)

        if input_context.num_input_pipelines > 1:
            dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)

        #dataset = dataset.shuffle(batch_size * 10)
        dataset = dataset.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        #dataset = dataset.repeat(-1)
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    return _input_fn_

def build_distributed_dataset(batch_size, strategy, name):
    input_fn = bulid_input_fn(batch_size, name)
    return strategy.distribute_datasets_from_function(input_fn)

def get_preprocess_fn(color_distortion):
    #Get function that accepts an image and returns a preprocessed image
    return functools.partial(
        data_util_pretrain.preprocess_image,
        height = FLAGS.image_size,
        width= FLAGS.image_size,
        color_distort=color_distortion)


def get_data_pretrain(name):
    logging.info('Loading Astro pretrain data')
    data_dir = 'imagenes_clasificadas_nair/'
    def read_images(image_file, label):
        image = tf.io.read_file(data_dir + image_file)
        image = tf.image.decode_jpeg(image, channels = 3)
        return image, label
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    df = pd.read_csv(name)
    file_paths = df['name'].values
    labels = tf.zeros([df.shape[0]], dtype=tf.int32)
    ds_train = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds_train = ds_train.map(read_images, num_parallel_calls =AUTOTUNE)
    return ds_train

def get_number_of_images(path_to_csv):
    df = pd.read_csv(path_to_csv)
    return df.shape[0]