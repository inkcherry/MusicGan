from global_variable import  global_config
import data
import tensorflow as tf
def load_data():
    initial_data = data.load_traindata_from_npz(global_config['traindata_filename'])

    dataset=data.get_dataset(initial_data,global_config['data_shape'],global_config['batch_size'])
    # print(type(data_set))

    train_x,train_y=dataset.make_one_shot_iterator().get_next(), None
    return train_x,train_y

tx,ty=load_data()
print(type(tx))
print(tf.shape(tx))
