from DataPropcessing import *
from OPR_LTR import *
from losses import *

if tf.__version__ >= '2.0.0':
    tf.compat.v1.disable_eager_execution()
else:
    from tensorflow.python.keras import backend as K
    K.set_learning_phase(True)

try:
    from tensorflow.python.keras.optimizers import Adam
except ImportError:
    tf.compat.v1.disable_eager_execution()
    from tensorflow.python.keras.optimizer_v1 import Adam

root = "/home/hanyu/TITS/ES"
x0_in_memory, x_in_memory, y_in_memory = load_input_data(root)
x0_scale, x_scale, y_scale = data_normalization(x0_in_memory, x_in_memory, y_in_memory)
adj_path = "/home/hanyu/TITS/link/link_49_hk_diag.pt"
A = torch.load(adj_path)
Adj = ES_adjacnt_matrix(A)
Y = rank_label_matrix(A, y_scale)

ndcg = tfr.keras.metrics.NDCGMetric
# X = tf.stack([tf.divide(tf.matmul(x0_scale, A.float()), tf.reduce_sum(A,0)), x_scale],2)
x0_scale = x0_scale.reshape((x0_scale.shape[0],x0_scale.shape[1],1))
# y_train = y_train.reshape((y_train.shape[0],y_train.shape[1],1))
X = [x0_scale, Adj]
print(x0_scale.shape)
print(Adj.shape)
model_input = X
# print(X.shape)
# Compile model
model = GCN()
model.compile(optimizer=Adam(0.002), loss= combined_loss,  metrics=[combined_loss,'mse'])#[]

print(model.summary())

x0_scale = x0_scale[102:28902,:,:]
Adj = Adj[102:28902,:,:]
Y = Y[102:28902,:,:]

x_train = [x0_scale[:int(28800*0.9),:,:].numpy(), Adj[:int(28800*0.9),:,:]]
y_train = Y[:int(28800*0.9),:,:]

X_test = [x0_scale[int(28800*0.9):,:,:].numpy(), Adj[int(28800*0.9):,:,:]]
y_test = Y[int(28800*0.9):,:,:]

print("x_train shape: ", x_train[0].shape)
print("y_train shape: ", y_train.shape)
print("X_test shape: ", X_test[0].shape)
print("y_test shape: ", y_test.shape)

# Xtrain, Xtest, ytrain, ytest = train_test_split(X,y_train,test_size=0.2,random_state=42)
import numpy as np
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.models import Model

import time

class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

# mc_callback = ModelCheckpoint('/content/drive/MyDrive/polyu/GNN/proposed/checkpoint_ab/best_model_softmax_mse_loss_mask_y100.h5',
#                               monitor='val_loss',
#                               save_best_only=True,
#                               save_weights_only=True)
time_callback = TimeHistory()
mc_callback = ModelCheckpoint('/home/hanyu/TITS/models/checkpoint/test.h5',
                              monitor='val_loss',
                              save_best_only=True,
                              save_weights_only=True)
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="/content/drive/MyDrive/polyu/GNN/proposed/checkpoint/mse_loss_mask_logs")

model.fit(
    x=x_train,
    y=y_train,
    batch_size=128,
    epochs=1000,
    validation_split = 0.1,
    callbacks=[mc_callback,time_callback]
)

model.load_weights('/home/hanyu/TITS/models/checkpoint/test.h5')
test_result = model.predict(X_test, batch_size=1)
savepath = '/home/hanyu/TITS/record/'
result = torch.stack((torch.from_numpy(test_result), y_test),3)
torch.save(result,os.path.join(savepath, 'test.pt'))