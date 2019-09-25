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

# vars=tf.trainable_variables(gen.scope_name)
# print(vars)
with tf.control_dependencies(update_ops + [gen_step_increment]):
    train_op_gen = gen_opt.minimize(g_loss, global_step,tf.trainable_variables(gen.scope_name))
#

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True


global_step=tf.train.get_global_step()
steps_per_iter = global_config['n_dis_updates_per_gen_update']+1  #5dis +1gen
hooks = [tf.train.NanTensorHook(total_loss)]

tensor_logger={
    'gen_step':gen_step,
    'g_loss':g_loss,
    'd_loss':d_loss
}

with tf.train.MonitoredTrainingSession(
    save_checkpoint_steps=5000*steps_per_iter,   #per save 5000
    save_summaries_steps=0,
    checkpoint_dir=global_config['checkpoint_dir'],log_step_count_steps=0,
    hooks=hooks,config=tf_config) as sess:

    step=tf.train.global_step(sess,global_step)
    print("now start training")
    print("step="+str(step))
    if step==0:
        print("print format-------> step, g_loss,d_loss")
    if step>=global_config['steps']:
        print('finish')
        exit()
    while step<global_config['steps']:
        if step < 10:
            n_dis_updates = 10 * global_config['n_dis_updates_per_gen_update']  #
        else:
            n_dis_updates = global_config['n_dis_updates_per_gen_update']
        for _ in range(n_dis_updates):
            sess.run(train_op_dis)

        log_loss_steps=1
        if(step+1)%log_loss_steps==0:
            step, _, tensor_logger_values = sess.run([
                gen_step, train_op_gen,
                tensor_logger])
            # print(tensor_logger['gen_step'],tensor_logger['g_loss'],tensor_logger['d_loss'])
            print("{}, {: 10.6E}, {: 10.6E}\n".format(
                tensor_logger_values['gen_step'],
                tensor_logger_values['g_loss'],
                tensor_logger_values['d_loss']))
        else:
            step,_ = sess.run([gen_step,train_op_gen])

        if sess.should_stop():
            break
        #here to sampler

print("Training end")






# print(type(train_x))
# print(tf.shape(train_x))
# print(train_x)
