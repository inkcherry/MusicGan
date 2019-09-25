from discriminitor import Discriminator
from generator import  Generator
import  tensorflow as tf
from data import  load_data   #one by one batch
from data_pre import  load_traindata_from_npz,get_dataset
from global_variable import  global_config
from util import  wloss,get_scheduled_variable
dis=Discriminator()  #all default
gen=Generator()      #all default

# init_data=load_traindata_from_npz(global_config['traindata_filename'])
# dataset=get_dataset(init_data,global_config['data_shape'],global_config['batch_size'])

train_x, _ = load_data(global_config['traindata_filename'])

z=tf.truncated_normal(( global_config['batch_size'], global_config['latent_dim']))

slope = tf.get_variable(
    'slope', [], tf.float32, tf.constant_initializer(1.0),
    trainable=False)

fake_x=gen(z,training=True)
# print("fake_x")
# print(tf.shape(fake_x))
# print(fake_x)

dis_real=dis(train_x,training=True)
dis_fake=dis(fake_x,training=True)

g_loss,d_loss =wloss(dis_real,dis_fake)

total_loss =d_loss+g_loss

learning_rate= tf.get_variable(
    'learning_rate', [], tf.float32,
    tf.constant_initializer(global_config['initial_learning_rate']),
    trainable=False)
scheduled_learning_rate = get_scheduled_variable(
    global_config['initial_learning_rate'],
    global_config['learning_rate_schedule']['end_value'],
    global_config['learning_rate_schedule']['start'],
    global_config['learning_rate_schedule']['end'])

tf.add_to_collection(
    tf.GraphKeys.UPDATE_OPS,
    tf.assign(learning_rate, scheduled_learning_rate))


gen_opt=tf.train.AdamOptimizer(learning_rate,
                         global_config['adam']['beta1'],
                         global_config['adam']['beta2'])
dis_opt=tf.train.AdamOptimizer(learning_rate,
                         global_config['adam']['beta1'],
                         global_config['adam']['beta2'])

global_step=tf.train.get_or_create_global_step()
gen_step=tf.get_variable(
                'gen_step', [], tf.int32, tf.constant_initializer(0),
                trainable=False)


train_op_dis=dis_opt.minimize(d_loss,global_step,tf.trainable_variables(dis.scope_name))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
gen_step_increment = tf.assign_add(gen_step, 1)

# print("7777777777777777777777777777")
# print(gen.scope_name)
# print(g_loss)
# print((global_step))
# exit()
# train_op_gen=gen_opt.minimize(g_loss,global_step,tf.trainable_variables(gen.scope_name))

vars=tf.trainable_variables(gen.scope_name)
print(vars)
with tf.control_dependencies(update_ops + [gen_step_increment]):
    train_op_gen = gen_opt.minimize(g_loss, global_step,tf.trainable_variables(gen.scope_name))
#

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True


# print(type(train_x))
# print(tf.shape(train_x))
# print(train_x)
