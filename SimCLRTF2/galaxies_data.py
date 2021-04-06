import tensorflow.compat.v2 as tf
import pandas as pd

def get_data_train():
    print('Loading train data')
    data_dir = '/home/pedri0/Documents/imagenes_no_clasificadas_desi/'
    def read_images(image_file, label):
        image = tf.io.read_file(data_dir + image_file)
        image = tf.image.decode_jpeg(image, channels = 3)
        return image, label

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    df = pd.read_csv('/home/pedri0/Documents/GitHub/Modified-SimCLR/SimCLRTF2/galaxies_train.csv')
    file_paths = df['name'].values
    labels = tf.zeros([df.shape[0]], dtype=tf.int64)
    ds_train = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds_train = ds_train.map(read_images, num_parallel_calls =AUTOTUNE)
    return ds_train

def get_data_test():
    print('Loading test data')
    data_dir = '/home/pedri0/Documents/imagenes_clasificadas_nair/'
    def read_images(image_file, label):
        image = tf.io.read_file(data_dir + image_file)
        image = tf.image.decode_jpeg(image, channels = 3)
        return image, label

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    df = pd.read_csv('/home/pedri0/Documents/GitHub/Modified-SimCLR/SimCLRTF2/nair_unbalanced_test.csv')
    file_paths = df['name'].values
    labels = df['new_class'].values
    labels = tf.convert_to_tensor(labels, dtype=tf.int64)
    ds_test = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds_test = ds_test.map(read_images, num_parallel_calls=AUTOTUNE)
    return ds_test

def get_data_valid():
    print('Loading valid data')
    data_dir = '/home/pedri0/Documents/imagenes_clasificadas_nair/'
    def read_images(image_file, label):
        image = tf.io.read_file(data_dir + image_file)
        image = tf.image.decode_jpeg(image, channels = 3)
        return image, label
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    df = pd.read_csv('/home/pedri0/Documents/GitHub/Modified-SimCLR/SimCLRTF2/nair_unbalanced_valid.csv')
    file_paths = df['name'].values
    labels = df['new_class'].values
    labels = tf.convert_to_tensor(labels, dtype=tf.int64)
    ds_valid = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds_valid = ds_valid.map(read_images, num_parallel_calls=AUTOTUNE)
    return ds_valid

def get_data_fine():
    print('Loading data for finetuning')
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

    ##TODO arreglar lo de las labels para el dataset de pretrain o ver si es importante arreglarlo ya que da la clase 1,0,0,0,0

def get_number_of_images(path_to_csv):
    df = pd.read_csv(path_to_csv)
    print('This Dataset has {} images'.format(df.shape[0]))
    return df.shape[0]
