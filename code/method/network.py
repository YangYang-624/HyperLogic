import sys 
sys.path.append("../")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
from torch.optim.lr_scheduler import MultiStepLR

import numpy as np

import method.dataLoader as mydl
import method.my_layers as myla
import method.my_loss as mylo

def initWeights(w, data):
    init.constant_(w, 0)


class Net(nn.Module):
    def __init__(self, init_weights, init_bias, data_sparsity, device_cpu, device_gpu):
        super(Net, self).__init__()
        input_dim = init_weights.size()[1]
        hidden_dim = init_weights.size()[0]
        self.fc0_enc = myla.BinarizedLinearModule(input_dim, hidden_dim, .5, data_sparsity, False, init_weights, None, init_bias, device_cpu, device_gpu)
        self.fc3_dec = myla.BinarizedLinearModule(hidden_dim, input_dim, .5, data_sparsity, True, self.fc0_enc.weight.data, self.fc0_enc.weightB.data, None, device_cpu, device_gpu)
        self.act0 = myla.BinaryActivation(hidden_dim, device_gpu)
        self.act3 = myla.BinaryActivation(input_dim, device_gpu)
        self.clipWeights()


    def forward(self, x):
        x = self.fc0_enc(x)
        x = self.act0(x, False)
        x = self.fc3_dec(x)
        output = self.act3(x, True)
        return output

    def clipWeights(self, mini=-1, maxi=1):
        self.fc0_enc.clipWeights(mini, maxi)
        self.act0.clipBias()
        self.act3.noBias()


def train(model, device_cpu, device_gpu, train_loader, optimizer, lossFun, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device_gpu)
        optimizer.zero_grad()
        output = model(data)
        itEW = [par for name, par in model.named_parameters() if name.endswith("enc.weight")]
        loss = lossFun(output, data, next(iter(itEW)))
        loss.backward()
        optimizer.step()
        model.clipWeights()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    return


def test(model, device_cpu, device_gpu, test_loader, lossFun):
    model.eval()
    test_loss = 0
    correct = 0
    numel = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device_gpu)
            output = model(data)
            itEW = [par for name, par in model.named_parameters() if name.endswith("enc.weight")]
            test_loss += lossFun(output, data, next(iter(itEW)))
            numel +=  output.numel()
            correct += torch.sum(output==data)

    _, target = next(iter(test_loader))
    print('\nTest set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, numel,
        100. * correct / numel))


def learn(input, lr, gamma, weight_decay, epochs, hidden_dim, train_set_size, batch_size, test_batch_size, log_interval, device_cpu, device_gpu):


    kwargs = {}
    trainDS = mydl.DatDataset(input, train_set_size, True, device_cpu)
    train_loader = torch.utils.data.DataLoader(trainDS,
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(mydl.DatDataset(input, train_set_size, False, device_cpu),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    if hidden_dim == -1:
        hidden_dim = trainDS.ncol()

    new_weights = torch.zeros(hidden_dim, trainDS.ncol(), device=device_gpu)
    initWeights(new_weights, trainDS.data)
    new_weights.clamp_(1/(trainDS.ncol()), 1)
    bInit = torch.zeros(hidden_dim, device=device_gpu)
    init.constant_(bInit, -1)

    model = Net(new_weights, bInit, trainDS.getSparsity(), device_cpu, device_gpu).to(device_gpu)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    lossFun = mylo.weightedXor(trainDS.getSparsity(), weight_decay, device_gpu)

    scheduler = MultiStepLR(optimizer, [5,7], gamma=gamma)
    for epoch in range(1, epochs + 1):
        train(model, device_cpu, device_gpu, train_loader, optimizer, lossFun, epoch, log_interval)

        test(model, device_cpu, device_gpu, test_loader, lossFun)
        scheduler.step()

    return model, new_weights, trainDS
