#from utils.data_loader import *
from itertools import *
import functools
import random
import operator
import numpy as np

def collaps(patterns):
    return functools.reduce(operator.iconcat, patterns, [])

def gen_patterns(items, n, ran=[5,10]):
    return [sorted(random.sample(set(items), random.choice(range(*ran)))) for i in range(n)]

def gen_rows_for_one_p(pattern, n_rows, n_cols, n_random):
    mat = np.zeros((n_rows, n_cols))
    mat[:,pattern] = 1
    noise = np.random.choice(n_cols,(n_rows,n_random),replace=True)
    ind = [[i] for i in range(n_rows)]
    mat[ind,noise] = 1
    return mat

def gen_rows_for_n_p(patterns, n_rows, n_cols, k_of_n, n_random):
    pattern_combinations = list(map( lambda x: np.array(collaps(x)),combinations(patterns,k_of_n)))
    patterns = np.array(pattern_combinations, dtype=object)
    templates = np.zeros((len(patterns), n_cols))
    for i, p in enumerate(patterns):
        p = p.astype(np.int32)
        templates[i,p] = 1
    assignment = np.random.choice(len(patterns),n_rows)
    mat = templates[assignment]
    noise = np.random.choice(n_cols,(n_rows,n_random),replace=True)
    ind = [[i] for i in range(n_rows)] 
    mat[ind,noise] = 1
    return mat

def gen_one_pattern_per_class(n_classes, n_rows_per_class, n_cols, n_random, pattern_len = [5,10]):
    gt_dict = {}
    data = []

    if isinstance(n_rows_per_class, list):
        assert len(n_rows_per_class) == n_classes
    else:
        n_rows_per_class = n_classes * [n_rows_per_class]

    for i, n_rows in enumerate(n_rows_per_class):
        gt_dict[i] = gen_patterns(range(0,n_cols),1, pattern_len)
        data.append(gen_rows_for_one_p(gt_dict[i][0], n_rows, n_cols, n_random))
    data = np.concatenate(data,axis=0)
    labels = np.array(collaps([n_rows*[i] for i,n_rows in enumerate(n_rows_per_class)]))

    return data, labels, gt_dict
    
def gen_n_patterns_per_class(n_classes, n_rows_per_class, n_cols, n_random, n_patterns, k_of_n, pattern_len = [5,10], overlap=False):
    gt_dict = {}
    data = []
    assert n_patterns >= k_of_n
    if isinstance(n_rows_per_class, list):
        assert len(n_rows_per_class) == n_classes
    else:
        n_rows_per_class = n_classes * [n_rows_per_class]
    
    if overlap:
        overlapping = gen_patterns(range(0,n_cols),n_classes, [5,6])
    for i, n_rows in enumerate(n_rows_per_class):
        gt_dict[i] = gen_patterns(range(0,n_cols),n_patterns, pattern_len)
        rows = gen_rows_for_n_p(gt_dict[i], n_rows, n_cols, k_of_n, n_random)
        if overlap:
            rows[:,overlapping[i]]=1
        data.append(rows)
    data = np.concatenate(data,axis=0)
    labels = np.array(collaps([n_rows*[i] for i,n_rows in enumerate(n_rows_per_class)]))
    if overlap:
        return data, labels, gt_dict, overlapping
    else:
        return data, labels, gt_dict

def gen_with_mat(mat, n_rows_per_class, n_cols, n_random, pattern_len = [5,10]):
    np.random.seed(0)
    n_classes = len(mat)
    pattern_dict = {}
    templates = np.zeros((n_classes, n_cols))
    for i in range(n_classes):
        pat = gen_patterns(range(0,n_cols),1, pattern_len)[0]
        pattern_dict[i] = [pat]
        templates[i,pat] = 1
    assignment = []
    labels = [] 
    for i in range(n_classes):
        ass = np.random.choice(n_classes,n_rows_per_class, p=mat[i])
        assignment.append(ass)
        labels.extend(n_rows_per_class*[i])
    assignment = np.concatenate(assignment)
    data = templates[assignment]
    n_rows = n_rows_per_class*n_classes
    noise = np.random.choice(n_cols,(n_rows,n_random),replace=True)
    ind = [[i] for i in range(n_rows)] 
    data[ind,noise] = 1
    return data, np.array(labels), pattern_dict
    
def apply_label_noise(data, labels, p=0.1):
    unique = np.unique(labels)
    new_labels = labels.copy()
    for l in unique:
        ind = labels==l
        n_flip = int(p*ind.sum())
        vals = list(set(unique) - {l})
        replace_vals = np.random.choice(vals,n_flip)
        ind = np.random.choice(ind.nonzero()[0],size=n_flip)
        new_labels[ind] = replace_vals
    return data, new_labels

def flip_rows(data, labels, p_flip):
    mask = np.random.choice(2,data.shape,p=[1-p_flip,p_flip])
    data_mod = np.mod(data+mask,2)
    return data_mod, labels
    
def resample_labels(data, labels, mat):
    new_labels = labels.copy()
    unique = np.unique(labels)
    for l in unique:
        ind = labels==l
        replace_vals = np.random.choice(unique,np.sum(ind),p=mat[l])
        new_labels[ind] = replace_vals
    return data, new_labels



def experiment1(cols=None, seed=0,overlap=False):
    mat = [[0.9,0.1],
              [0.1,0.9]]
    print(cols)
    if cols is None:
        cols = [1000, 5000, 10000, 25000,50000]
    exp_dict = {}
    for n_cols in cols:
        n_rows = 10000
        np.random.seed(seed)
        random.seed(seed)
        if n_cols < 1000:
            n_patterns = 5
            pattern_len = [5,15]
            n_random = 10
        else:
            n_patterns = 10
            pattern_len = [5,15]
            n_random = 10

        data, labels, gt_dict = gen_n_patterns_per_class(2,n_rows, n_cols,n_random,n_patterns,3,pattern_len=pattern_len)
        data,labels = resample_labels(data,labels,mat)
        if n_cols >= 1000:
            data = apply_pattern_noise(data)
        data = apply_destructive(data, gt_dict, 0.025)
        exp_dict[n_cols] = [data.astype(int), labels, gt_dict]
    return exp_dict

def apply_gen_destructive(data,p):
    noise =  np.random.choice(2,size=data.shape,p=[1-p,p])
    data = np.logical_xor(data,noise)
    return data

def apply_destructive(data, gt_dict, p):
    for label, patterns in gt_dict.items():
        for pat in patterns:
            mask = np.sum(data[:,pat],axis=1)==len(pat)
            sub_data = data[mask][:,pat]
            noise =  np.random.choice(2,size=sub_data.shape,p=[1-p,p])
            #data[mask,np.array(pat)] = 
            data[np.ix_(mask,pat)]=np.logical_xor(sub_data,noise)
    return data

def apply_pattern_noise(data):
    rows, cols = data.shape
    pattern_len = [int(0.01*cols),int(0.025*cols)]
    mask,labels,gt_dict = gen_n_patterns_per_class(1, rows, cols, 0, 20,2,pattern_len=pattern_len)
    data = np.logical_or(data,mask)
    return data


def class_mat(k):
    mat = np.diag(v=[0.9 for i in range(k)])
    mat[mat==0] = 0.1/(k-1)
    return mat

def experiment2(K=None,seed=0):
    if K is None:
        K = [2,5,10,25,50]
    exp_dict = {}
    n_cols = 5000
    n_rows = 1000
    n_random = 10
    for k in K:
        #n_rows = k * n_cols
        mat = class_mat(k)
        np.random.seed(seed)
        random.seed(seed)
        data, labels, gt_dict = gen_n_patterns_per_class(k, n_rows, n_cols, n_random, 10,3, pattern_len = [5,15])
        data = apply_destructive(data, gt_dict, 0.025)
        data,labels = resample_labels(data,labels,mat)
        data = apply_pattern_noise(data)
        exp_dict[k] = [data.astype(int), labels, gt_dict]
    return exp_dict

def experiment2_test(K=None):
    if K is None:
        K = [2,5,10,25,50]
    exp_dict = {}
    n_cols = 500
    n_rows = 500
    n_random = 15
    for k in K:
        #n_rows = k * n_cols
        mat = class_mat(k)
        np.random.seed(0)
        random.seed(0)
        data, labels, gt_dict = gen_n_patterns_per_class(k, n_rows, n_cols, n_random, 5,1, pattern_len = [5,10])
        data = apply_destructive(data, gt_dict, 0.025)
        data,labels = resample_labels(data,labels,mat)
        data = apply_pattern_noise(data)
        exp_dict[k] = [data.astype(int), labels, gt_dict]
    return exp_dict

def experiment3(snrs,seed=0):
    mat = [[0.9,0.1],
              [0.1,0.9]]
    exp_dict = {}
    pattern_len = [5,15]
    n_patterns = 10
    n_cols = 5000
    n_rows = 1000

    avg_signal = 10
    for snr in snrs:
        #int(0.001*n_cols)
        np.random.seed(seed)
        random.seed(seed)
        n_random = snr
        data, labels, gt_dict = gen_n_patterns_per_class(2,n_rows, n_cols,n_random,n_patterns,3,pattern_len=pattern_len)
        data,labels = resample_labels(data,labels,mat)
        data = apply_destructive(data, gt_dict, 0.025)
        data = apply_pattern_noise(data)
        exp_dict[snr] = [data.astype(int), labels, gt_dict]
    return exp_dict

def experiment4(destructive,seed=0):
    mat = [[0.9,0.1],
              [0.1,0.9]]
    exp_dict = {}
    pattern_len = [5,15]
    n_patterns = 10
    n_cols = 5000
    n_rows = 1000
    n_random = 10
    for des in destructive:
        np.random.seed(seed)
        random.seed(seed)
        data, labels, gt_dict = gen_n_patterns_per_class(2,n_rows, n_cols,n_random,n_patterns,3,pattern_len=pattern_len)
        data,labels = resample_labels(data,labels,mat)
        data = apply_pattern_noise(data)
        data = apply_destructive(data, gt_dict, des)
        exp_dict[des] = [data.astype(int), labels, gt_dict]
    return exp_dict




def get_gt_dict(exp):
    gt_dict = {}
    for k,v in exp.items():
        gt_dict[k] = v[2]
    return gt_dict