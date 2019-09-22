from global_variable import  global_config
import data
from generator import Generator
from discriminitor import  Discriminator
import discriminitor
import tensorflow as tf
class Model:
    def __init__(self,scope_name='Model'):
        self.scope_name = scope_name
        with tf.variable_scope(self.scope_name,reuse=tf.AUTO_REUSE)as scope:
            self.scope=scope
            self.gen=Generator(global_config['data_shape'][-1])
            self.dis=Discriminator(global_config['data_shape'][-1])
            self.components = [self.gen, self.dis]

    def __call__(self, x=None, z=None, y=None, c=None, mode=None):
        if mode == 'predict':
            return self.get_predict_nodes(z, y, c)
        if mode == 'train':
            return None


    def get_predict_nodes(self, z, y, c):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            nodes = {'z': z}

            nodes['slope'] = tf.get_variable(
                'slope', [], tf.float32, tf.constant_initializer(1.0),
                trainable=False)

            nodes['fake_x']=self.gen(nodes['z'],y,False)
            #self,tensor_in,training=None,slope=None
        return nodes






