
import torch
import torch.nn as nn
import torch.distributions as torchdist
from torch.autograd import Variable
from torch.autograd import Function

from torch.nn import init



def BinarizeTensorThresh(tens, thresh):
    t = (tens > thresh).float()
    return t

def BinarizeTensorStoch(tens, device_gpu):
    t = tens.bernoulli()
    return t

def SteppyBias(tensor, is_dec):
    if is_dec:
        t = tensor.clamp(min=0,max=0)
    else:
        t = tensor.clamp(max=-1)
    t = t.int().float()
    return t

class BinarizeFunction(Function):
    @staticmethod
    def forward(ctx, input, bias, biasNI):
        ctx.save_for_backward(input, bias, biasNI)
        res = (input+biasNI).clamp(0,1).round()
        return res

    @staticmethod
    def backward(ctx, grad_output):
        input, bias, biasNI = ctx.saved_tensors
        if (bias[0] < 0):
            grad_input = (input+biasNI).clamp(0,1).round()*grad_output
        else:
            grad_input = grad_output
        # Throw out negative gradient to bias if node was not active
        # (Do not punish for things it did not do)
        grad_bias = (1-(input+biasNI).clamp(0,1).round())*grad_output.clamp(max=0).sum(0) + (input+biasNI).clamp(0,1).round()*grad_output.sum(0)
        #print(grad_output.shape)
        return grad_input, grad_bias, None


class BinaryActivation(nn.Module):

    def __init__(self, size, device_gpu):
        super(BinaryActivation, self).__init__()
        self.bias = nn.Parameter(-torch.ones(size, device=device_gpu), requires_grad=True)
        self.biasNI = self.bias.clone().detach().to(device_gpu)

    def forward(self, input, is_dec):
        with torch.no_grad():
            self.biasNI = SteppyBias(self.bias, is_dec)
        return BinarizeFunction.apply(input, self.bias, self.biasNI)

    def clipBias(self):
        with torch.no_grad():
            self.bias.clamp_(max=-1)

    def noBias(self):
        with torch.no_grad():
            self.bias.clamp_(min=0, max=0)



class BinaryLinearFunction(Function):

    @staticmethod
    def forward(ctx, input, weight, weightB):
        ctx.save_for_backward(input, weight, weightB)
        #print(input.shape)
        #print(weightB.t().shape)
        out = input.matmul(weightB.t())
        return out


    @staticmethod
    def backward(ctx, grad_output):

        input, weight, weightB = ctx.saved_tensors

        grad_input = grad_output.matmul(weight)

        grad_weight = grad_output.t().matmul(input)

        return grad_input, grad_weight, None



class BinarizedLinearModule(nn.Module):

    def __init__(self, inum, onum, threshold, data_sparsity, is_dec, enc_weights, enc_weightsB, bInit, device_cpu, device_gpu, mode='None',alpha_weight=None):
        super(BinarizedLinearModule, self).__init__()
        self.inum = inum
        self.onum = onum
        self.devGPU = device_gpu
        self.devCPU = device_cpu
        # threshold to use for binarization
        self.threshold = threshold
        self.mode = mode

        self.is_dec = is_dec
        if is_dec:
            self.alpha = nn.Parameter(torch.tensor(0.01))
            self.alpha.data = alpha_weight
            self.weight = nn.Parameter(torch.zeros(inum, onum, device = device_gpu))
            self.weightB = torch.zeros(inum, onum, device = device_gpu)
            self.weight.data = enc_weights.transpose(0,1)
            self.weightB.data = enc_weightsB.transpose(0,1)
        else:
            self.alpha = nn.Parameter(torch.tensor(0.01))
            self.base_weight = nn.Parameter(torch.log(enc_weights / (1 - enc_weights)))

            self.weight = nn.Parameter(torch.log(enc_weights / (1 - enc_weights)))
            # binarized weight matrix
            self.weightB = torch.zeros(onum, inum, device = device_gpu)

    def get_weight(self):
        # if tmp_external_weight is not None:
        #     if self.is_dec:
        #         external_weight = tmp_external_weight.T
        #     else:
        #         external_weight = tmp_external_weight
        #     if self.mode == 'replace':
        #         weight = external_weight
        #     elif self.mode == 'add1':
        #         weight = self.weight + external_weight
        #     elif self.mode == 'add2':
        #         weight = (1 - self.alpha) * self.weight + self.alpha * external_weight
        #     elif self.mode == 'add3':
        #         weight = self.weight + self.alpha * external_weight
        #     else:
        #         weight = self.weight
        # else:
        #     weight = self.weight
        final_weight = self.weight
        return final_weight

    def forward(self, input):
        # these tensors are not tracked for gradient computation
        final_weight = self.get_weight()
        if not (self.is_dec):
            #self.weight = nn.Parameter(final_weight)
            #self.weight.data.copy_(final_weight)
            with torch.no_grad():
                #self.weight.data.copy_(final_weight)
                self.weightB.data.copy_(BinarizeTensorStoch(final_weight, self.devGPU))
                #self.weightB.data.copy_(BinarizeTensorThresh(self.weight,self.threshold))

        out = BinaryLinearFunction.apply(input, final_weight, self.weightB)
        return out
    
    def forward_test(self, input,t):
        # these tensors are not tracked for gradient computation
        with torch.no_grad():
            if not(self.is_dec):
                #self.weightB.data.copy_(BinarizeTensorStoch(self.weight, self.devGPU))
                self.weightB.data.copy_(BinarizeTensorThresh(self.weight,t))

        out = BinaryLinearFunction.apply(input, self.weight, self.weightB)
        return out

    def extra_repr(self):
        return 'input_features={}, output_features={}, threshold={}'.format(
            self.inum, self.onum, self.threshold
        )

    def clipWeights(self, mini=-1, maxi=1):
        with torch.no_grad():
            # weights are clamped at 1/input_dim corresponding to - on expectation - one incoming weight per node being active. Counteracting dying neurons

            self.weight.clamp_(1/(self.inum), maxi)
