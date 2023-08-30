import torch
import numpy as np
from sklearn.metrics import ndcg_score, average_precision_score
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
torch.set_printoptions(threshold=torch.inf)
import scipy.sparse as sp

## part 1: spatial adjacent matrix building using GCN's method

def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    return adj

def ToMatrix(link, preds, target):
    Mpreds = torch.zeros((max(preds.shape), link.shape[0], link.shape[1]))
    Mtarget = torch.zeros((max(preds.shape), link.shape[0], link.shape[1]))
    for i in tqdm(range(max(Mpreds.shape))):
        # Y[i,:,:] = torch.mul(torch.from_numpy(A), y_scale[i,:]) 
        Mpreds[i,:,:] = torch.mul(preds[i,:], link) # y_sacle is a line, and it multiply each line of A
        Mtarget[i,:,:] = torch.mul(target[i,:], link)
    print('\n')
    print("label size: ", Mtarget.shape)
    print("preds size: ", Mpreds.shape)
    return Mpreds, Mtarget

def ToMatrix2(link, preds, target):   # only self, no recommendation
    Mpreds = torch.zeros((max(preds.shape), link.shape[0], link.shape[1]))
    Mtarget = torch.zeros((max(preds.shape), link.shape[0], link.shape[1]))
    one = torch.eye(link.shape[0])
    for i in tqdm(range(max(Mpreds.shape))):
        # Y[i,:,:] = torch.mul(torch.from_numpy(A), y_scale[i,:]) 
        Mpreds[i,:,:] = torch.mul(preds[i,:], one) # y_sacle is a line, and it multiply each line of A
        Mtarget[i,:,:] = torch.mul(target[i,:], one)
    print('\n')
    print("label size: ", Mtarget.shape)
    print("preds size: ", Mpreds.shape)
    return Mpreds, Mtarget

def get_rec_metrics(method, preds, target, num_test, topk):
    print("pred_shape:", preds.shape)
    print("targ_shape:", target.shape)
    # ndcg = tfr.keras.metrics.NDCGMetric(topn=topk)
    # mrr = tfr.keras.metrics.MRRMetric(topn=topk)
    # map = tfr.keras.metrics.MeanAveragePrecisionMetric(topn=topk)
    ndcg_list = []
    mrr_list = []
    map_list = []
    for i in tqdm(range(num_test)):
        ndcgi = ndcg_score(target[i,:,:], preds[i,:,:], k=topk)
        # mrri = mrr(preds[i,:,:], target[i,:,:])
        if topk == 1:
            mapj = 0
            for j in range(target.shape[1]):
                mapj += average_precision_score(target[i,j,:]==target[i,j,:].max(), preds[i,j,:])      
            mapi = mapj / target.shape[1]
            # mapi = map(np.argsort(target[i,:,:],1)>0.5, preds[i,:,:])
            ndcg_list.append(ndcgi)
            # mrr_list.append(mrri)
            map_list.append(mapi)
        else:
            mapj = 0
            for j in range(target.shape[1]):
                mapj += average_precision_score(target[i,j,:]>=target[i,j,np.argsort(target[i,j,:])[-1*topk]], preds[i,j,:])      
            mapi = mapj / target.shape[1]
            # mapi = map(np.argsort(target[i,:,:],1)>0.5, preds[i,:,:])
            ndcg_list.append(ndcgi)
            # mrr_list.append(mrri)
            map_list.append(mapi)
    ndcg_all = np.array(ndcg_list)
    ndcg_all_mean = ndcg_all.mean()
    ndcg_all_std = ndcg_all.std()
    mrr_all = np.array(mrr_list)
    mrr_all_mean = mrr_all.mean()
    mrr_all_std = mrr_all.std()
    map_all = np.array(map_list)
    map_all_mean = map_all.mean()
    map_all_std = map_all.std()   
    print("\n%s_ndcg_mean(std): %f(%f)"%(method, ndcg_all_mean, ndcg_all_std))
    print("%s_mrr_mean(std): %f(%f)"%(method, mrr_all_mean, mrr_all_std))
    print("%s_map_mean(std): %f(%f)"%(method, map_all_mean, map_all_std))
    return ndcg_all
  
# worst waiting time computing
def WWT(targ,n=288):
    targ = targ[:n]
    WT = np.zeros_like(targ)
    for i in tqdm(range(targ.shape[0])):
        for j in range(targ.shape[1]):
            if targ[i,j]>0.5:
                WT[i,j] = 0
        else:
            WT[i,j] = 0.5-targ[i,j]
    return WT

def wt(pred,targ,topn=1):
  for i in range(max(pred.shape)):
    Spred = np.argsort(pred)
    # print(Spred)
    WT = -10
    for j in range(topn):
      j = j+1
      if WT < targ[Spred[-1*j]]:
        WT = targ[Spred[-1*j]]
  WT_best = targ.max()
  if WT>0.5:
    WT = 0
  else:
    WT = 0.5 - WT
  if WT_best>0.5:
    WT_best = 0
  else:
    WT_best = 0.5 - WT_best
  return WT, WT_best

def WTall(pred, targ, n=288, topn=1):
  pred = pred[:n]
  targ = targ[:n]
  WT = np.zeros_like(pred[:,:,0])
  WT_best = np.zeros_like(pred[:,:,0])
  print("WT_shape: ", WT.shape)
  for i in tqdm(range(pred.shape[0])):
    for j in range(pred.shape[1]):
      WT[i,j],WT_best[i,j] = wt(pred[i,j],targ[i,j],topn)
  return WT, WT_best

A = torch.load('/home/hanyu/TITS/link/link_49_hk_diag.pt')
A = torch.from_numpy(preprocess_adj(A)).float()



HCGtest = torch.load('/home/hanyu/TITS/models/checkpoint/test.h5', map_location=torch.device('cpu'))
preds_HCGtest = HCGtest[2:,:,:,0]

target_HCGtest = HCGtest[2:,:,:,1]
pndcg = get_rec_metrics("Proposed", preds_HCGtest, target_HCGtest, 2878, 1)