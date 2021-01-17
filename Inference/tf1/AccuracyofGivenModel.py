import tensorflow.compat.v1 as tf
import galaxies_datTrue as galax
import PreprocessImagesforInference as prep
from absl import app
from absl import flags
import numpy as np
import tensorflow_hub as hub

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model_dir', None,
    'Finetuned Model directory.')


flags.DEFINE_integer(
    'batch_size', 16,
    'Batch size.')


def main(argv):

    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    #configure hardware
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    #define the dataset in this case correspond to test
    ds = galax.get_data_test()
    ds = ds.map(prep.map_fn,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(FLAGS.batch_size, drop_remainder=False)
    ds = prep.pad_to_batch(ds, FLAGS.batch_size)
    #get images and labels from ds
    images, labels = tf.data.make_one_shot_iterator(ds).get_next()
    #open model
    hub_path = FLAGS.model_dir
    module = hub.Module(hub_path, trainable=False)
    key = module(inputs=images, signature="default", as_dict=True)
    logits_t = key['logits_sup'][:, :]
    #set lists to store predicted clases and real clases
    predicted = []
    real_label = []
    #do inference
    with tf.compat.v1.Session() as ses:
        ses.run(tf.global_variables_initializer())
        try:
            while True:
                image, label, logits = ses.run((images, labels, logits_t))
                pred = logits.argmax(-1)
                predicted.append(pred)
                for i in range(FLAGS.batch_size):
                    if np.sum(label[i]!=0):
                        real_label.append(label[i])
                    else:
                        continue
        
        except tf.errors.OutOfRangeError:
            pass
    #convert lists to numpy arrays        
    predicted = np.array(predicted)
    predicted = predicted.flatten()

    real_label = np.array(real_label)
    #one hot for predicted array
    with tf.compat.v1.Session() as ses:
        prediction_one_hot = ses.run((tf.one_hot(predicted,5)))
    #count how many correct predictions
    count = 0
    for i in range(real_label.shape[0]):
        is_true = np.array_equal(prediction_one_hot[i], real_label[i])
        if is_true:
            count += 1
    print('Accuracy : {} %'.format((count*100/real_label.shape[0])))

if __name__ == "__main__":
    tf.disable_v2_behavior()
    app.run(main)