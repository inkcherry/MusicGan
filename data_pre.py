import tensorflow as tf
import numpy as np
from global_variable import global_config

def load_traindata_from_npz(filename):        #invisible fun
    #load traindata from npz file
    npz_file=np.load(filename)
    print("load npz file and the file shape is ↓")
    print(npz_file['shape'])
    #npz_file.files--------->['nonzero', 'shape']
    data = np.zeros(npz_file['shape'],np.bool_)
    # sparse matrix to normal matrix
    data[[x for x in npz_file['nonzero']]] = True
    return data


def yeid_data_for_getgataset(initial_data):
    #yield one phrase from data,for get_dataset to generator,
    for item in initial_data:
        yield  item*2.-1.
        #yield a [n_bars,n_timesteps_inbar,n_pitches,n_tracks] format ndarray iter set.



def get_dataset(initial_data,data_shape,batch_size):
    dataset=tf.data.Dataset.from_generator(
        lambda:yeid_data_for_getgataset(initial_data),tf.float32)

    dataset = dataset.map(lambda pianoroll: set_pianoroll_shape(
        pianoroll, data_shape), num_parallel_calls=1)


    dataset = dataset.shuffle(global_config['shuffle_size']).repeat().batch(batch_size)
    return dataset.prefetch(global_config['prefetch_size'])


def set_pianoroll_shape(pianoroll, data_shape):
    """Set the pianoroll shape and return the pianoroll."""
    pianoroll.set_shape(data_shape)
    return pianoroll





