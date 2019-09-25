from global_variable import  global_config
import data_pre
import scipy.stats
import numpy as np
import tensorflow as tf
def load_data(filename):                     #for train
    initial_data = data_pre.load_traindata_from_npz(filename)

    print(initial_data.shape)

    dataset=data_pre.get_dataset(initial_data, global_config['data_shape'], global_config['batch_size'])

    # print(type(dataset))



    train_x,train_y=dataset.make_one_shot_iterator().get_next(), None
    # print(tf.shape(train_x))
    return train_x,train_y




def get_samples():
    print("get sample----------------------")
    sample_z = scipy.stats.truncnorm.rvs(
            -2, 2, size=(np.prod(global_config['sample_grid']), global_config['latent_dim']))  #[8,8]    128

    return sample_z







