import torch
from tqdm import tqdm
import os
import os.path as osp

def processed_file_names(root):

    # root: the path that saves the original dataset

    return os.listdir(root)

def load_STdata(root, savepath):

    # savepath: the path that saves the final dataset

    len_file = len(processed_file_names(root))

    print('loading ST dataset to memory...')

    xs = []

    for i in tqdm(range(len_file)):
        x = torch.load(osp.join(root, '%d.pt'%i))
        xs.append(x)

    x_in_memory = torch.stack(xs, axis=1).float()
    torch.save(x_in_memory, os.path.join(savepath, 'hkall.pt'))
    print("shape of ST dataset:", x_in_memory.shape)

    return x_in_memory

def make_ESdata(data, index, num_b, t_a, max_a):

    # This function create a 'ESGraph' at index time from the 'data'
    # data: all data in one graph [num_node, num_time]
    # index: the current time index
    # num_b: the number of turn-over events before current time
    # t_a: the interval between current time and a futute predicting time
    # max_a: the largest duration of predicting

    n, l = data.shape
    datai = data[:,index] # one slice, [num_node, 1]
    graph = torch.zeros(num_b+1+1,n) 
    Y = data[:, (torch.arange(max_a) + t_a + index)]

    for i in range(n):
        count_b = 1
        count_a = 1        
        index_b = index-1
        data_before = data[i,index_b]
        ilist = [datai[i]]

        for j in range(int(num_b)):
            if j % 2 == 0:
                data_now = datai[i]  
            else:     
                data_now = int(not bool(datai[i])) 
            while data_now == data_before:
                count_b = count_b + 1
                index_b = index_b - 1
                data_before = data[i,index_b]          
            if data_before == 1:
                count_b = count_b * (-1)
            ilist.insert(0, count_b)
            count_b = 0
        state = Y[i,0]
        state_next = Y[i,count_a]

        while state == state_next and count_a < max_a-1:   ## the longest during time is set to be 25min
            count_a = count_a + 1
            state_next = Y[i, count_a]

        if state == 0:
            count_a = count_a * (-1)        

        ilist.append(count_a)
        itensor = torch.tensor(ilist)
        graph[:,i] = itensor

    return graph

root = '/home/hanyu/TITS/data'
save_path_ST = "/home/hanyu/TITS/ST"
save_path_ES = "/home/hanyu/TITS/ES"


#### create real ESGraph ####

num_ESGraph = 20000  # num_ESGraph < len_file - len_time_y - len_y - 1
num_b = 5
len_time_y = 4 
len_y=6
count = 0

STdata = load_STdata(root, save_path_ST)

for i in tqdm(range(num_ESGraph)):

    one_graph = make_ESdata(STdata, i, num_b, len_time_y, len_y)
    torch.save(one_graph, os.path.join(save_path_ES, '%d.pt'%count))
    count += 1
