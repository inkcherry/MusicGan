import tensorflow as tf
import numpy as np
from global_variable import  global_config
def load_traindata_from_npz(filename):

    npz_file=np.load(filename)
    #print(npz_file.files)          ['nonzero', 'shape']
    data = np.zeros(npz_file['shape'])
    # 稀疏矩阵
    data[[x for x in npz_file['nonzero']]] = True
    return data



data_=load_traindata_from_npz(global_config['traindata_filename'])
print(data_.shape)

# nonzero_=data_['nonzero']
# print(nonzero_.shape)
# shape_=data_['shape']
# print(shape_.shape)
# print(data_.size)



