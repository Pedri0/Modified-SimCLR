import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import tensorflow as tf

def get_new_ds(representations):

    representations = representations.numpy()

    pseudolabels = MiniBatchKMeans(n_clusters = 5).fit_predict(representations)
    cluster_length = [np.where(pseudolabels==c)[0].shape[0] for c in range(5)]
    larger_cluster_length = max(cluster_length)

    df = pd.read_csv('nair_unbalanced_train.csv')
    df['pseudolabels'] = pseudolabels

    by_psudolabels = [df[df['pseudolabels']==label].reset_index(drop=True)
                        for label in range(5)]

    extended_dataframes = []
    for dataframe in by_psudolabels:
        ratio = larger_cluster_length/dataframe.shape[0]
        extended_dataframe = dataframe.loc[dataframe.index.repeat(1 if ratio==1.0 else round(ratio)+1)]
        extended_dataframe = extended_dataframe.sample(n=larger_cluster_length).reset_index(drop=True)
        extended_dataframes.append(extended_dataframe)
    
    del by_psudolabels

    extended_dataframes = pd.concat(extended_dataframes).reset_index(drop=True)
    extended_dataframes = extended_dataframes.sample(frac=1).reset_index(drop=True)

    return extended_dataframes


def save_rep_per_step(outputs):
    representations = outputs.numpy()
    df = np.load('data.npy')
    df = np.concatenate((df,representations))
    np.save('data.npy', df)
    return 1


#def save_rep_per_step(outputs):
#    representations = pd.DataFrame(outputs.numpy())
#    df = pd.read_csv('representations_per_epoch.csv')
#    df = pd.concat((df,representations)).reset_index(drop=True)
#    df.to_csv('dr.csv', index=False)
#    return 1

#def update_mini_batch(mini_batch_kmeans, outputs, append_list):
def update_mini_batch(mini_batch_kmeans,outputs, pseudo_labels_per_epoch):
    outputss = tf.py_function(func=tensor_to_numpy, inp=[outputs], Tout=np.float32)
    #print('hello')
    #pseudo_labels_per_epoch.append(np.array(outputss))
    #print(pseudo_labels_per_epoch)
    mini_batch_kmeans.partial_fit(np.array(outputss))
    append_list.append(mini_batch_kmeans.labels_)
    return None

def tensor_to_numpy(outputs):
    arrar = outputs.numpy()
    return arrar

def dss(tensor_arr):
    for i in range(tensor_arr.size().numpy()):
        l.append(tensor_arr.read(i).numpy())
    
    l = np.concatenate(l, axis=0)
    f = pd.DataFrame(l)
    f.to_csv('w.csv', index=False)

    return 1

def save_array(counter, tensors):
    counter = counter.numpy()

    if counter != 0:
        tensors = tensors.numpy()
        data = np.load('outputs.npy')
        data = np.concatenate((data, tensors), axis=0)
        np.save('outputs.npy', data)

    else:
        tensors = tensors.numpy()
        np.save('outputs.npy', tensors)
    
    return 1


