import functools
from absl import flags
from absl import logging
from absl.app import FLAGS

import data_util
import galaxies_data_fine as galax
import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS

def bulid_input_fn(global_batch_size):
    """Build input function.
    Args:
        global_batch_size: Global batch size.
        is_training: Whether to build in training mode.
    Returns:
        A function that accepts a dict of params and returns a tuple of images and
        features, to be used as the input_fn in TPUEstimator.
    """
    def _input_fn_(input_context):
        #Inner input function
        batch_size = input_context.get_per_replica_batch_size(global_batch_size)
        #deleted loggins
        preprocess_fn_finetune = get_preprocess_fn(True, is_pretrain=False)
        num_classes = 5

        def map_fn(image, label):
            #Produces multiple transformations of the same batch for pretraining
            image = preprocess_fn_finetune(image)
            label = tf.one_hot(label, num_classes)
            return image, label

        logging.info('Using Astro finetuning data')
        dataset = galax.get_data_fine()

        if input_context.num_input_pipelines > 1:
            dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)

        buffer_multiplier = 10
        dataset = dataset.shuffle(batch_size * buffer_multiplier)
        dataset = dataset.repeat(-1)
        dataset = dataset.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        prefetch_buffer_size = 2
        dataset = dataset.prefetch(prefetch_buffer_size)
        return dataset

    return _input_fn_

def build_distributed_dataset(batch_size, strategy):
    input_fn = bulid_input_fn(batch_size)
    return strategy.distribute_datasets_from_function(input_fn)

def get_preprocess_fn(is_training, is_pretrain):
    #Get function that accepts an image and returns a preprocessed image
    test_crop=True
    return functools.partial(
        data_util.preprocess_image,
        height = FLAGS.image_size,
        width= FLAGS.image_size,
        is_training=is_training,
        color_distort=is_pretrain,
        test_crop=test_crop
    )