import tensorflow_ranking as tfr
import tensorflow as tf

loss_d = tfr.keras.losses.MeanSquaredLoss() #tf.keras.losses.mean_squared_error
loss_r = tf.keras.losses.CategoricalCrossentropy()
loss_l = tfr.keras.losses.ListMLELoss()
loss_s = tfr.keras.losses.SoftmaxLoss()
loss_n = tfr.keras.losses.ApproxNDCGLoss()


def cross_entropy(y_true, y_pred):

  return -1 * tf.matmul(y_true, tf.math.log(y_pred))


def combined_loss(y_true, y_pred):

  # return tf.reduce_mean(-1 * tf.matmul(y_true, tf.transpose(tf.math.log(y_pred)))) + tf.multiply(loss_d(y_true, y_pred), A) 

  # return loss_r(y_true, y_pred) + loss_d(y_true, y_pred)
#   return loss_d(y_true, y_pred)
  
  
  # return loss_n(tf.reshape(y_true, (-1,48)), tf.reshape(y_pred, (-1,48))) / 48
  # return loss_l(tf.reshape(y_true, (-1,48)), tf.reshape(y_pred, (-1,48))) / 48
  return loss_s(tf.reshape(y_true, (-1,48)), tf.reshape(y_pred, (-1,48))) / 480 + loss_d(y_true, y_pred)
  # return loss_s(tf.reshape(y_true, (-1,48)), tf.reshape(y_pred, (-1,48))) / 48

  # return loss_r(y_true, y_pred) / 48 / 48