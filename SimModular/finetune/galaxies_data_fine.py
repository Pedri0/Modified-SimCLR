import tensorflow.compat.v2 as tf
import pandas as pd
from absl import logging

def get_data_fine():
    logging.info('Loading Astro finetuning data')
    data_dir = '/home/pedri0/Documents/imagenes_clasificadas_nair/'
    def read_images(image_file, label):
        image = tf.io.read_file(data_dir + image_file)
        image = tf.image.decode_jpeg(image, channels = 3)
        return image, label
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    df = pd.read_csv('/home/pedri0/Documents/GitHub/Modified-SimCLR/SimCLRTF2/nair_unbalanced_train.csv')
    file_paths = df['name'].values
    labels = df['new_class'].values
    labels = tf.convert_to_tensor(labels, dtype=tf.int64)
    ds_fine = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds_fine = ds_fine.map(read_images, num_parallel_calls=AUTOTUNE)
    return ds_fine

def get_number_of_images(path_to_csv):
    df = pd.read_csv(path_to_csv)
    return df.shape[0]
