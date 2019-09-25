# from global_variable import  global_config
# import data_pre
# import os
# import numpy as np
# from generator import Generator
# from discriminitor import  Discriminator
# import discriminitor
# import tensorflow as tf
# class Model:
#     def __init__(self,scope_name='Model'):
#         self.scope_name = scope_name
#         with tf.variable_scope(self.scope_name,reuse=tf.AUTO_REUSE)as scope:
#             self.scope=scope
#             self.gen=Generator(global_config['data_shape'][-1])
#             self.dis=Discriminator(global_config['data_shape'][-1])
#             self.components = [self.gen, self.dis]
#
#     def __call__(self, x=None, z=None, y=None, c=None, mode=None):
#         if mode == 'predict':
#             return self.get_predict_nodes(z, y, c)
#         if mode == 'train':
#             return None



    # def get_predict_nodes(self, z, y, c):
    #     with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
    #         nodes = {'z': z}
    #
    #         nodes['slope'] = tf.get_variable(
    #             'slope', [], tf.float32, tf.constant_initializer(1.0),
    #             trainable=False)
    #
    #         nodes['fake_x']=self.gen(nodes['z'],y,False)
    #         #self,tensor_in,training=None,slope=None
    #
    #
    #
    #         def _get_filepath(folder_name, name, suffix, ext):
    #             """Return the filename."""
    #             if suffix:
    #                 return os.path.join(
    #                     global_config['result_dir'], folder_name, name,
    #                     '{}_{}.{}'.format(name, str(suffix, 'utf8'), ext))
    #             return os.path.join(
    #                 global_config['result_dir'], folder_name, name,
    #                 '{}.{}'.format(name, ext))
    #
    #         def _save_array(array, suffix, name):
    #             """Save the input array."""
    #             filepath = _get_filepath('arrays', name, suffix, 'npy')
    #             np.save(filepath, array.astype(np.float16))
    #             return np.array([0], np.int32)
    #
    #         arrays = {'fake_x': nodes['fake_x']}
    #
    #
    #         save_array_ops = []
    #         for key, value in arrays.items():
    #             save_array_ops.append(tf.py_func(
    #                 lambda array, suffix, k=key: _save_array(
    #                     array, suffix, k),
    #                 [value, global_config['suffix']], tf.int32))
    #
    #
    #         nodes['save_arrays_op'] = tf.group(save_array_ops)
    #
    #
    #     return nodes






