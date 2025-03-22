
import os
import torch
import pandas as pd
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader

def readDatFile(dat_file):

  ncol = -1
  nrow = 0
  with open(dat_file) as datF:
    # read .dat format line by line
    l = datF.readline()
    while l:
      # drop newline
      l = l[:-1]
      if l == "":
        continue
      if l[-1] == " ":
        l = l[:-1]
      # get indices as array
      sl = l.split(" ")
      sl = [int(i) for i in sl]
      maxi = max(sl)
      if (ncol < maxi):
        ncol = maxi
      nrow += 1
      l = datF.readline()
  data = np.zeros((nrow, ncol), dtype=np.single)
  with open(dat_file) as datF:
    # read .dat format line by line
    l = datF.readline()
    rIdx = 0
    while l:
      # drop newline
      l = l[:-1]
      if l == "":
        continue
      if l[-1] == " ":
        l = l[:-1]
      # get indices as array
      sl = l.split(" ")
      idxs = np.array(sl, dtype=int)
      idxs -= 1
      # assign active cells
      data[rIdx, idxs] = np.repeat(1, idxs.shape[0])
      rIdx += 1
      l = datF.readline()

  return data



## Construct a dataset from a regular csv file
class RegDataset(Dataset):

  def __init__(self, csv_file, train_proportion, is_training):
    data = pd.read_csv(csv_file, sep=",", index_col=0)
    self.data = np.asarray(data)
    ran = np.arange(0,math.ceil(train_proportion*self.data.shape[0]))
    trainmin = self.data[ran,1].min()
    trainmax = self.data[ran,1].max()
    if not(is_training):
      ran = np.arange(math.ceil(train_proportion*self.data.shape[0]),self.data.shape[0])
    self.data = self.data[ran,:]
    self.data[:,1] = self.data[:,1] - trainmin /  (trainmax - trainmin)

  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, index):
    return self.data[index,0], self.data[index,1]



## Construct a dataset from a .dat file
class DatDataset(Dataset):

  def __init__(self, dat_file, train_proportion, is_training, device_cpum, file=True, data=None):
    if file:
        data = readDatFile(dat_file)
        self.data = np.asarray(data)
    else:
        self.data = np.asarray(data)
    print(self.data.dtype)
    self.sparsity = np.count_nonzero(self.data)/np.prod(self.data.shape)
    if is_training:
      ran = np.arange(0,math.ceil(train_proportion*self.data.shape[0]))
    else:
      ran = np.arange(math.ceil(train_proportion*self.data.shape[0]),self.data.shape[0])
    self.data = torch.from_numpy(self.data[ran,:])#, device=device_cpu)


  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, index):
    return self.data[index,:], self.data[index,:]

  def matmul(self, other):
    return self.data.matmul(other)

  def nrow(self):
    return self.data.shape[0]
  def ncol(self):
    return self.data.shape[1]

  def getSparsity(self):
    return self.sparsity



## Construct a dataset from a .dat file
class DiffnapsDatDataset(DatDataset):

  def __init__(self, dat_file, train_proportion, is_training, device_cpum,  data, labels):
    self.data = np.asarray(data)
    self.labels = np.asarray(labels)
    
    if is_training:
      ran = np.arange(0,math.ceil(train_proportion*self.data.shape[0]))
    else:
      ran = np.arange(math.ceil(train_proportion*self.data.shape[0]),self.data.shape[0])
    self.sparsity = np.count_nonzero(self.data)/np.prod(self.data.shape)
    self.data = torch.from_numpy(self.data[ran,:])
    self.labels = torch.from_numpy(self.labels[ran])

  def __len__(self):
    return self.data.shape[0]

  def ncol(self):
    return self.data.shape[1]

  def __getitem__(self, index):
    return self.data[index,:], self.labels[index]
