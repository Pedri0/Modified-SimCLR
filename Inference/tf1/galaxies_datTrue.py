import tensorflow as tf
import pandas as pd

#funncion para poner la extension .jpg al nombre de cada imagen
def get_image_path(df):
        for name in range(len(df.index)):
            df['name'][name] = df['name'][name] + str('.jpg')
        return df
#funcion para regresar el dataset de train
def get_data_train():
    #directorio de las imagenes
    directory = '/home/pedri0/Documents/astrotf2/imagenes_no_clasificadas_desi/'
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    #lectura de imagenes con tf.io
    def read_image(image_file, label):
        image = tf.io.read_file(directory + image_file)
        image = tf.image.decode_jpeg(image, channels = 3)
        #es necesario regresar el resize ya que si no tira error
        #return tf.image.resize(image, [800, 800]) , label ##cambiar label por tf.zeros(numero de clases)
        #ACTUALIZACION 22/11/20 me di cuenta que no se muestran las imagenes correctamente si
        #se hace el resize a 800, 800 por lo que voy a cambiar esto por el siguiente return
        #y tengo que volver a entrenar ya que la perdida se estancaba en 8.96 aprox
        #y ademas en el fine tuning divergia veremos si esta es la solucion al problema.
        #return tf.image.resize(image, [800, 800]) , label ##cambiar label por tf.zeros(numero de clases)
        #funcion para regresar el dataset de test
        return image, label
    #leemos el archivo .csv con los nombres de las imagenes
    df = pd.read_csv('data_train.csv')
    #agregamos las extensiones .jpg
    #df = get_image_path(df)
    #obtenemos el nombre de las imagenes con su respectiva extension    
    file_paths = df['name'].values
    #como es train y no tenemos etiquetas, por default dejamos las etiquetas en cero
    labels = tf.zeros([df.shape[0]],dtype=tf.int64) ##Aqui se hace el cambio a tf.zeros
    #creamos el dataset con tf.data.Dataset que incluye el path de las imagenes y sus respectivas etiquetas (todas en cero)
    ds_train = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    #hacemos el mapeo
    ds_train = ds_train.map(read_image, num_parallel_calls=AUTOTUNE)
    #regresamos el dataset recien creado
    return ds_train

def get_data_test():
    #directorio de las imagenes de test
    directory = 'imagenes_clasificadas_nair/'#'/home/pedri0/Documents/astrotf2/imagenes_clasificadas_nair/'
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    #lectura de imagenes con tf.io
    def read_image(image_file, label):
        image = tf.io.read_file(directory + image_file)
        image = tf.image.decode_jpeg(image, channels = 3)
        #es necesario regresar el resize ya que si no tira error
        #return tf.image.resize(image, [800, 800]) , label
        #ver ACTUALIZACION en el get_data_train
        return image, label
    #leemos el archivo .csv con los nombres de las imagenes
    df = pd.read_csv('nairTest.csv')
    #agregamos las extensiones .jpg    
    #df = get_image_path(df)
    #obtenemos el nombre de las imagenes con su respectiva extension    
    file_paths = df['name'].values
    #las etiquetas las obtenemos directamente del csv
    labels = df['new_class'].values
    labels = tf.convert_to_tensor(labels, dtype=tf.int64)
    #creamos el dataset con tf.data.Dataset que incluye el path de las imagenes y sus respectivas etiquetas
    ds_test = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    #hacemos el mapeo
    ds_test = ds_test.map(read_image, num_parallel_calls=AUTOTUNE)
    #regresamos el dataset recien creado
    return ds_test