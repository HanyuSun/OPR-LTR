import os
import os.path as osp
import torch
import scipy.sparse as sp
import numpy as np
from tqdm import tqdm

def find_max_min(data):
  num_node, num_time = data.shape
  num_node, num_time = data.shape
  maxmin = torch.zeros(num_node,2)

  for i in tqdm(range(num_node)):
    dur_list = []
    datab = data[i,0]
    count = 1

    for j in range(1, num_time):
      dataa = data[i,j]
  
      if datab == dataa:
        count = count + 1
        if j == num_time-1:
          dur_list.append(count)
      else:
        dur_list.append(count)
        count = 1
      datab = dataa

    maxmin[i,0] = max(dur_list)
    maxmin[i,1] = min(dur_list)

  return maxmin,dur_list

def norm_mean_std(x_in_memory):
    x_scale = torch.zeros_like(x_in_memory)
    for i in range(x_in_memory.shape[0]):
        x_in_memoryi = x_in_memory[i,:,:]
        mean =  x_in_memoryi.mean(dim=0)
        std = x_in_memoryi.std(dim=0)
        x_scale[i,:,:] = (x_in_memoryi - mean)/std
    return x_scale

def norm_minmax_1(x_in_memory):
    x_scale = torch.zeros_like(x_in_memory)
    stdata = torch.load("/home/hanyu/TITS/ST/hkall.pt")
    graph_maxmin,_ = find_max_min(stdata)
    for i in range(x_in_memory.shape[0]):
        x_in_memoryi = x_in_memory[i,:,:]
        xup = torch.sub(abs(x_in_memoryi), graph_maxmin[:,1].T) * (x_in_memoryi / abs(x_in_memoryi))
        std = x_in_memoryi.std(dim=0)
        x_scale[i,:,:] = torch.div(xup, std)
    return x_scale

def norm_minmax_2(x_in_memory):
    x_scale = torch.zeros_like(x_in_memory)
    for i in range(x_in_memory.shape[0]):
        x_in_memoryi = x_in_memory[i,:,:]
        mean =  x_in_memoryi.mean(dim=0)
        std = x_in_memoryi.std(dim=0)
        xup = torch.sub(x_in_memoryi, mean)
        x_scale[i,:,:] = torch.div(xup, std)
    return x_scale

def norm_tanh(x_in_memory):
    x_scale = torch.tanh(x_in_memory)
    return x_scale 

#  **input data**


root = "/home/hanyu/TITS/ES"


def load_input_data(root):

    # x0_in_memory: real-time data;
    # x_in_memory: history turn-over event;
    # y_in_memory: label

    x0 = []
    x = []
    y = []

    num_time = len(os.listdir(root))
    for i in tqdm(range(num_time)):
      datai = torch.load(osp.join(root, '%d.pt'%i))
      b, n = datai.shape
      x0i = datai[b,:]
      xi = datai[:b,:]
      yi = datai[b+1,:]

      x0.append(x0i)
      x.append(xi)
      y.append(yi)

    x0_in_memory = torch.stack(x0,axis=1).float()
    x_in_memory = torch.stack(x,axis=1).float()
    y_in_memory = torch.stack(y,axis=1).float()

    print("x0_in_memory shape: ", x0_in_memory.shape)
    print("x_in_memory shape: ", x_in_memory.shape)
    print("y_in_memory shape: ", y_in_memory.shape)

    return x0_in_memory, x_in_memory, y_in_memory

# data_normalization
def data_normalization(x0_in_memory, x_in_memory, y_in_memory):
    x0_scale = x0_in_memory.T
    x0_scale = torch.stack([x0_scale], axis=2)
    #x_scale

    # x_scale = norm_mean_std(x_in_memory)
    # x_scale = norm_minmax_1(x_in_memory)
    # x_scale = norm_minmax_2(x_in_memory)
    x_scale = norm_tanh(x_in_memory)
    #y_scale
    # y_scale=y_in_memory.T
    # minmax:
    yup = torch.sub(y_in_memory.T, torch.min(y_in_memory)) # 分子
    ydown = (torch.max(y_in_memory)-torch.min(y_in_memory)).T # 分母
    y_scale = torch.div(yup, ydown)
    # y_scale = torch.exp(y_in_memory)
    # y_scale = y_scale.T

    print("x0_scale shape: ", x0_scale.shape)
    print("x_scale shape: ", x_scale.shape)
    print("y_scale shape: ", y_scale.shape)

    return x0_scale, x_scale, y_scale

x0_in_memory, x_in_memory, y_in_memory = load_input_data(root)
x0_scale, x_scale, y_scale = data_normalization(x0_in_memory, x_in_memory, y_in_memory)

# Adjacent Matrix Building
## part 1: spatial adjacent matrix building using GCN's method
def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    return adj

def node_deg(adj):
  adj = preprocess_adj(adj)
  deg = adj.sum(axis=0)
  return deg

def aggregating_gcn(adj, method = "both"):
  if method == "both":
    adj = preprocess_adj(adj)
    degs = node_deg(adj)
    # print(degs.shape)
    adj_c = np.zeros_like(adj)
    for i in range(adj.shape[0]):

      for j in range(adj.shape[1]):
        norm = pow(degs[0,i] * degs[0,j], -0.5)
        # print(norm.dtype)
        adj_c[i, j] = norm
    adj_c = np.multiply(adj_c, adj)
  return adj_c

#part 2: add event information to the above adjacnt matrix
def ES_adjacnt_matrix(A):
    adj_c = aggregating_gcn(A) 
    adj_c = adj_c.astype(np.float32)
    x_f = x_scale.numpy()
    adj = []
    for i in range(max(x_f.shape)):
        adj_f = x_f[:,i,:].T
        adj.append(np.hstack((adj_c, adj_f)))
        Adj = np.array(adj)

    print("Adj_shape: ", Adj.shape)

    return Adj

adj_path = "/home/hanyu/TITS/link/link_49_hk_diag.pt"
A = torch.load(adj_path)
Adj = ES_adjacnt_matrix(A)

# rank label building——matrix
def rank_label_matrix(A, y_scale, Fsoftmax=0):
    A = torch.from_numpy(preprocess_adj(A)).float()
    Y = torch.zeros((max(y_scale.shape), A.shape[0], A.shape[1]))
    for i in tqdm(range(max(y_scale.shape))):
        Y[i,:,:] = torch.mul(y_scale[i,:], A) # y_sacle is a line, and it multiply each line of A
    # softmax:
    if Fsoftmax:
        softmax = torch.nn.Softmax(2)
        Y = softmax(Y)
    print('\n')
    print("label size: ", Y.shape)
    return Y

# rank label building——line
def rank_label_line(A, y_scale):
    Y = torch.zeros((max(y_scale.shape), max(A.shape), 1))
    for i in tqdm(range(max(y_scale.shape))):
        for j in range(max(A.shape)):
            Y[i, j, 0] = y_scale[i, A[1, j]]
    print('\n')
    print("label size: ", Y.shape)
    return Y

Y = rank_label_matrix(A, y_scale)