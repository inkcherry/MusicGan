from global_variable import  global_config
import data
import scipy.stats
import numpy as np
import tensorflow as tf
def load_data():
    initial_data = data.load_traindata_from_npz(global_config['traindata_filename'])

    dataset=data.get_dataset(initial_data,global_config['data_shape'],global_config['batch_size'])
    # print(type(data_set))

    train_x,train_y=dataset.make_one_shot_iterator().get_next(), None
    return train_x,train_y




def get_samples():
    print("get sample----------------------")
    sample_z = scipy.stats.truncnorm.rvs(
            -2, 2, size=(np.prod(global_config['sample_grid']), global_config['latent_dim']))  #[8,8]    128

    return sample_z







