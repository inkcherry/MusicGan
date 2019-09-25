import  tensorflow as tf
import data
import data_pre
from global_variable import global_config
def wloss(d_real,d_fake):
    g_loss= -tf.reduce_mean(d_fake)
    d_loss=-g_loss-tf.reduce_mean(d_real)
    return g_loss,d_loss
#
# def load_train_data():
#     data=data_pre.load_traindata_from_npz(global_config['filename'])
#
def get_scheduled_variable(start_value, end_value, start_step, end_step):
    """Return a scheduled decayed/growing variable."""
    if start_step > end_step:
        raise ValueError("`start_step` must be smaller than `end_step`.")
    if start_step == end_step:
        return tf.constant(start_value)
    global_step = tf.train.get_or_create_global_step()
    zero_step = tf.constant(0, dtype=global_step.dtype)
    schedule_step = tf.maximum(zero_step, global_step - start_step)
    return tf.train.polynomial_decay(
        start_value, schedule_step, end_step - start_step, end_value)