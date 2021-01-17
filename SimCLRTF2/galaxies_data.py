import tensorflow.compat.v2 as tf
import pandas as pd

def get_data_train(data_dir, path_to_csv):
    print('Loading train data')

    def read_images(image_file, label):
        image = tf.io.read_file(data_dir + image_file)
        image = tf.image.decode_jpeg(image, channels = 3)
        return image, label

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    df = pd.read_csv(path_to_csv)
    file_paths = df['name'].values
    labels = tf.zeros([df.shape[0]], dtype=tf.int64)
    ds_train = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds_train = ds_train.map(read_images, num_parallel_calls =AUTOTUNE)
    return ds_train

def get_data_test(data_dir, path_to_csv):
    print('Loading test data')

    def read_images(image_file, label):
        image = tf.io.read_file(data_dir + image_file)
        image = tf.image.decode_jpeg(image, channels = 3)
        return image, label

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    df = pd.read_csv(path_to_csv)
    file_paths = df['name'].values
    labels = df['new_class'].values
    labels = tf.convert_to_tensor(labels, dtype=tf.int64)
    ds_test = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds_test = ds_test.map(read_images, num_parallel_calls=AUTOTUNE)
    return ds_test

def get_data_valid(data_dir, path_to_csv):
    print('Loading valid data')

    def read_images(image_file, label):
        image = tf.io.read_file(data_dir + image_file)
        image = tf.image.decode_jpeg(image, channels = 3)
        return image, label
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    df = pd.read_csv(path_to_csv)
    file_paths = df['name'].values
    labels = df['new_class'].values
    labels = tf.convert_to_tensor(labels, dtype=tf.int64)
    ds_valid = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds_valid = ds_valid.map(read_images, num_parallel_calls=AUTOTUNE)
    return ds_valid

    ##TODO metodos para regresar la cantidad de muestras por dataset
    ## arreglar lo de las labels para el dataset de pretrain o ver si es importante arreglarlo ya que da la clase 1,0,0,0,0