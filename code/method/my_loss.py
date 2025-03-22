import torch
import torch.nn as nn

class VanillaWeightedXor(nn.Module):

    def __init__(self, weight, weight_decay, device_gpu):
        super(VanillaWeightedXor, self).__init__()
        ## sparsity of data
        self.weight = weight
        ## decay rate
        self.weight_decay = weight_decay
        print("Data Sparsity:")
        print(self.weight)

    def forward(self, output, target, w):

        relu = nn.ReLU()
        diff = relu((output - target)).sum(1).mul(self.weight).mean() + relu((target - output)).sum(1).mul(1-self.weight).mean()
        diff += self.weight_decay*(((w - 1/target.size()[1])).sum(1).clamp(min=1).pow(2).sum())

        return diff



## weigh the different cases (positive vs negative) differently
## based on the data sparsity
class weightedXor(nn.Module):

    def __init__(self, weight, weight_decay, device_gpu, aggregator=torch.sum, label_decay=0.05, labels=0):
        super(weightedXor, self).__init__()
        ## sparsity of data
        self.weight = weight
        ## decay rate
        self.weight_decay = weight_decay
        self.labels = labels
        self.split_dim = -1
        print("Data Sparsity:")
        print(self.weight)

    def forward(self, output, target, w, hidden=None):
        relu = nn.ReLU()
        diff = relu((output - target)).sum(1).mul(self.weight).mean() + relu((target - output)).sum(1).mul(1-self.weight).mean()
        diff += horizontal_L2(self.config.weight_decay, w, target)
        #print(target.size())
        #print(horizontal_L2(self.config.weight_decay, w, target))
        if self.config.vertical_decay>0:
            diff += vertical_L2(self.config.vertical_decay, w, target)
        #diff += spike_regu(self.config.spike_weight, w, target)
        if self.config.elb_k >0.0 or self.config.elb_lamb>0.0:
            diff += elb_regu(self.config.elb_k, self.config.elb_lamb, w, target)
        if not hidden is None:
            diff += self.config.sparse_regu * torch.mean(torch.abs(hidden))
            #print(torch.mean(torch.norm(hidden,1,dim=1)))
        return diff
    
def vertical_L2(lambd, w, target):
    return torch.mean(lambd*(((w - 1/target.size()[1])).sum(1).clamp(min=1).pow(2)))

def spike_regu(lambd, w, target):
    return lambd * torch.mean(w*(1-w))

def horizontal_L2(lambd,w, target):
    #print(((w - 1/target.size()[1])).sum(1).clamp(min=1))
    #print(((w - 1/target.size()[1])).sum(1))
    return torch.mean(lambd*(((w - 1/target.size()[1])).sum(1).clamp(min=1).pow(2)))

def horizontal_L2_class(lambd,w, target):
    return torch.mean(lambd*(((w - 0)).sum(1).clamp(min=1).pow(2)))

def horizontal_L1(lambd, w, target):
    return torch.mean(torch.abs(lambd*(((w - 1/target.size()[1])).sum(1).clamp(min=1))))

def elb_regu(k, lambd, w, target):
    offset = 1/target.shape[1]
    w = w - offset
    elastic = lambda w: k*torch.abs(w) + lambd * torch.square(w)
    #elastic = lambda w: horizontal_L1(k,w,target) + horizontal_L2(lambd,w,target)# lambd * torch.square(w)
    return torch.mean( torch.minimum(elastic(w), elastic(w-1)))

def elb_regu_class(k, lambd, w, target):
    elastic = lambda w: k*torch.abs(w) + lambd * torch.square(w)
    return torch.mean( torch.minimum(elastic(w), elastic(w-1)))

class xor(nn.Module):

    def __init__(self, weight_decay, device_gpu):
        super(xor, self).__init__()
        self.weight_decay = weight_decay

    def forward(self, output, target, w):
        diff = (output - target).pow(2).sum(1).mean()

        # set minimum of weight to 0, to avoid penalizing too harshly for large matrices
        diff += (w - 1/target.size()[1]).pow(2).sum()*self.weight_decay

        return diff
