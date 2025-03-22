import sys
sys.path.append("../")
import numpy as np
import pandas as pd
from itertools import *
import torch

from method.diffnaps import *
import numpy as np
from utils.measures import *
import method.my_layers as myla


from method.network import *


import matplotlib.pyplot as plt

def tuple_to_name(string):
    a,b,c,_ = string.split(",")
    string = a[1:] + "_"+b[2]+"_"+c[0]
    return string


def gen_label_dict(base, length):
    label_list = list(map(lambda x: "".join(map(str,x)),product(base, repeat=length)))
    str_to_label = dict(zip(label_list,range(0,len(label_list))))
    return str_to_label, {v:k for k,v in str_to_label.items()}

def tranform_labels(labels, label_dict):
    num_labels = []
    for row in labels:
        l = "".join(list(map(str, list(row))))
        num_labels.append(label_dict[l])
    return num_labels


def sort_str(string):
    pattern = list(map(int,string.split(" ")))
    for p in sorted(pattern):
        print(p, end=" ")
        
def to_list(string):
    pattern = list(map(int,string.split(" ")))
    return pattern


def get_positional_patterns(weights, classifier_weights, t1=0.3, t2=0.3, t_mean=0.25, general=False):
    l = []
    num_l = []
    all_patterns = []
    # extract all patterns present in the data using thresholding on encoder
    hidden = torch.zeros(weights.shape[0], dtype=torch.int32) 
    for i,hn in enumerate(myla.BinarizeTensorThresh(weights, t1)):
        pat = torch.squeeze(hn.nonzero())
        pat = pat.reshape(-1)
        if hn.sum() >= 1 and list(pat.cpu().numpy()) not in l and weights[i].cpu().numpy().mean()<t_mean:
            all_patterns.append(list(pat.cpu().numpy()))
            l.append((i,list(pat.cpu().numpy())))
            num_l.append((i,list(weights[i].cpu().numpy())))
            hidden[i] = 1
    all_patterns = set(map(tuple, all_patterns)) 

    # assign patterns using thresholding on the classifer              
    patterns = dict(l)
    num_patterns = dict(num_l)
    bin_class = myla.BinarizeTensorThresh(classifier_weights, t2)
    assignment_tensor = (hidden * bin_class)
    labels = [str(i) for i in range(classifier_weights.shape[0])]
    assignment = {k:{} for k in labels }
    num_assignment = {k:{} for k in labels }
    assigned_patterns = []
    for key,hn in zip(labels, assignment_tensor):
        temp = sorted(list(map(list,set( [tuple(patterns.get(int(i), -1)) for i in torch.squeeze(hn.nonzero()).reshape(-1)]))))
        sorted(temp,key=lambda x:x[0])
        assignment[key] = temp
        num_assignment[key] = sorted(list(map(list,set( [tuple(num_patterns.get(int(i), -1)) for i in torch.squeeze(hn.nonzero()).reshape(-1)]))))
        assigned_patterns.extend(temp)
    assigned_patterns = set(map(tuple,assigned_patterns))
    general_patterns = list(map(list,all_patterns-assigned_patterns))
    if general:
        return l,num_l, hidden, num_assignment, assignment, sorted(general_patterns)
    else:
        return l,num_l, hidden, num_assignment, assignment
    


class TrainConfig():
    def __init__(self, train_set_size = 0.9, batch_size = 64, test_batch_size = 64, epochs = 100, lr = 0.01,
                       gamma = 0.1, seed = 1, log_interval = 10, hidden_dim = 500, thread_num = 12,
                       weight_decay = 0.05, wd_class=0.00, binaps=False, lambda_c=1.0,
                       spike_weight=0.0, vertical_decay=0.0, sparse_regu=0.0, elb_lamb=0.0, elb_k=0.0,
                       class_elb_k = 0.0, class_elb_lamb = 0.0, regu_rate = 1.0, class_regu_rate = 1.0, model=DiffnapsNet, test=False, init_enc="",
                        input_dim=100, z = 64, s = 256, bias = False, k=5, hyper_lr=0.0001, wd=0.0001, ngen=2, label_dim=2, beta=0.01, hyper_hidden_layers=[512,512], device="cuda", mode='replace'):
        self.train_set_size = train_set_size
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.gamma = gamma
        self.seed = seed
        self.log_interval = log_interval
        self.hidden_dim = hidden_dim
        self.thread_num = thread_num
        self.weight_decay = weight_decay
        self.wd_class = wd_class
        self.binaps = binaps
        self.lambda_c = lambda_c
        self.model = model
        self.spike_weight = spike_weight
        self.vertical_decay = vertical_decay
        self.sparse_regu = sparse_regu
        self.aggregator = torch.mean
        self.test = test
        self.elb_k = elb_k
        self.elb_lamb = elb_lamb
        self.class_elb_k = class_elb_k
        self.class_elb_lamb = class_elb_lamb
        self.regu_rate = regu_rate
        self.class_regu_rate = class_regu_rate
        self.init_enc = init_enc
        self.input_dim = input_dim
        self.z = z
        self.s = s
        self.bias = bias
        self.k = k
        self.hyper_lr = hyper_lr
        self.wd = wd
        self.ngen = ngen
        self.label_dim = label_dim
        self.beta = beta
        self.hyper_hidden_layers = hyper_hidden_layers
        self.device = device
        self.mode = mode

    def items(self):
        return self.__dict__.items()
        

def check_frequencies(data, patterns_gt, slack=0):
    ret = {}
    for k, patterns in patterns_gt.items():
        l = []
        for pattern in patterns:
            support = np.sum(np.sum(data[:,pattern],axis=1)>=(len(pattern)-slack))
            l.append(support)
        ret[k] = l
    return ret

def leave_x_out_frequencies(data, patterns_gt, x):
    ret = {}
    for k, patterns in patterns_gt.items():
        l = []
        for pattern in patterns:
            support = np.sum(np.sum(data[:,pattern],axis=1)==(len(pattern)-x))
            l.append(support)
        ret[k] = l
    return ret


