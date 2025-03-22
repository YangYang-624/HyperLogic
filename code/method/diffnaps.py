import time

import sys 
sys.path.append("./")
sys.path.append("../")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from method import HyperNet
from method import ops
import method.dataLoader as mydl
import method.my_layers as myla
import method.my_loss as mylo




## not used
def initWeights(w, data):
    init.constant_(w, 0.01) 
    

class DiffnapsNet(nn.Module):
    def __init__(self, init_weights, label_dim, init_bias, data_sparsity, device_cpu, device_gpu, config=None):
        super(DiffnapsNet, self).__init__()
        self.config = config
        self.mode = config.mode
        self.beta = nn.Parameter(torch.tensor(0.01))

        input_dim = init_weights.size()[1]
        hidden_dim = init_weights.size()[0]
        # Initialization of the architecture fc0_enc is W^e and fc3_dec is W^d
        self.fc0_enc = myla.BinarizedLinearModule(input_dim, hidden_dim, .5, data_sparsity, False, init_weights, None, init_bias, device_cpu, device_gpu, self.mode)
        self.fc3_dec = myla.BinarizedLinearModule(hidden_dim, input_dim, .5, data_sparsity, True, self.fc0_enc.weight.data, self.fc0_enc.weightB.data, None, device_cpu, device_gpu, self.mode, self.fc0_enc.alpha.data)
        self.act0 = myla.BinaryActivation(hidden_dim, device_gpu)
        self.act3 = myla.BinaryActivation(input_dim, device_gpu)
        torch.nn.init.xavier_normal_(self.fc0_enc.weight)

        mu, sigma = 0.1, 0.1
        y = torch.clamp(torch.normal(mu, sigma, size=(label_dim, hidden_dim)), 0.01, 0.99)
        x = torch.log(y / (1 - y))
        self.classifier_weight = nn.Parameter(x)
        #torch.nn.init.xavier_normal_(self.classifier_weight)
        #self.classifier_weight = nn.Parameter(torch.Tensor(label_dim, hidden_dim))
        #self.classifier_weight = nn.Parameter(torch.log(torch.clamp(torch.normal(mu, sigma, size=(label_dim, hidden_dim)), 0.01, 0.99) / (1 - torch.clamp(torch.normal(0.1, 0.1, size=(label_dim, hidden_dim)), 0.01, 0.99))))
        #self.classifier = nn.Linear(hidden_dim, label_dim,bias=False) #  corresponds to W^c

        # if config.init_enc=="bimodal":
        #         #     print("BiModal")
        #         #     init_bi_modal(self.classifier.weight,0.25,0.75,0.1, device_cpu)
        #         # else:
        #         #     torch.nn.init.xavier_normal_(self.classifier.weight)
        self.bin_classifier = nn.Linear(hidden_dim, label_dim,bias=False)
        print(self.fc0_enc.weight.mean())



    def forward(self, x, external_classifier=None):
        x = self.fc0_enc(x)
        z = self.act0(x, False)
        classification = self.classify(z, external_classifier)
        x = self.fc3_dec(z)
        output = self.act3(x, True)
        return output, classification, z

    def get_classify_weight(self, external_classifier=None):
        if external_classifier is not None and self.mode != 'None':
            if self.mode == 'replace':
                weight = external_classifier
            elif self.mode == 'add1':
                weight = self.classifier_weight + external_classifier
            elif self.mode == 'add2':
                weight = (1 - self.beta) * self.classifier_weight + self.beta * external_classifier
            elif self.mode == 'add3':
                weight = self.classifier_weight + self.beta * external_classifier
            else:
                weight = self.classifier_weight
        else:
            weight = self.classifier_weight
        final_weight = torch.sigmoid(weight/0.5)
        return final_weight

    def classify(self, z, external_classifier=None):
        final_weight = self.get_classify_weight(external_classifier)
        classification = F.linear(z, final_weight)
        return classification

    def clipWeights(self, mini=-1, maxi=1):
        self.fc0_enc.clipWeights(mini, maxi)
        if self.mode == 'add2':
            with torch.no_grad():
                self.fc0_enc.alpha.data.clamp_(0, 1)
                self.beta.data.clamp_(0.1, 1)
        #self.classifier.weight.data = self.classifier.weight.data.clamp(0,1)
        self.act0.clipBias()
        self.act3.noBias()
    
    def forward_test(self, x, t_enc, t_class): # Forwarding with binarized network
        w_bin = myla.BinarizeTensorThresh(self.classifier.weight,t_class)
        self.bin_classifier.weight.data = w_bin
        w_bin = myla.BinarizeTensorThresh(self.fc0_enc.weight,t_enc)
        x = self.fc0_enc.forward_test(x, t_enc)
        z = self.act0(x, False)
        classification = self.bin_classifier(z)
        x = self.fc3_dec(z)
        output = self.act3(x, True)
        return output, classification, z

    def forward_c_bin(self, x, t_class):
        w_bin = myla.BinarizeTensorThresh(self.classifier.weight,t_class)
        self.bin_classifier.weight.data = w_bin
        x = self.fc0_enc.forward(x)
        z = self.act0(x, False)
        classification = self.bin_classifier(z)
        x = self.fc3_dec(z)
        output = self.act3(x, True)
        return output, classification, z


    def learn(self, device_cpu, device_gpu, train_loader, optimizer, lossFun, epoch, log_interval, config, verbose=True):
        self.clipWeights()
        self.train()
        classification_loss = nn.CrossEntropyLoss()
        lossFun.config = config
        print_gpu(1)
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device_gpu)
            target = target.to(device_gpu)
            output, classification, z = self(data)
            itEW = [par for name, par in self.named_parameters() if name.endswith("enc.weight")]
            recon_loss = lossFun(output, data, next(iter(itEW)),hidden=z)
            #print(recon_loss)
            c_loss = classification_loss(classification,target.long())
            c_w = self.classifier.weight
            #print(config.lambda_c * c_loss )
            loss = recon_loss + config.lambda_c * c_loss + mylo.elb_regu_class(config.class_elb_k, config.class_elb_lamb, c_w, None) + mylo.horizontal_L2_class(config.wd_class, c_w, None)
            #print(mylo.elb_regu(config.class_elb_k, config.class_elb_lamb, c_w, None))
            loss.backward()
            optimizer.step()
            self.clipWeights()
            if batch_idx % log_interval == 0 and verbose:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                # print(self.fc0_enc.weight)
                # print(self.fc3_dec.weight)
            optimizer.zero_grad()
        #print("End")
        return

    def get_best_weights(self, mixer, generator, device_cpu, device_gpu, train_loader, lossFun, verbose):
        s = torch.randn(self.config.k, self.config.s).to(self.config.device)
        codes = mixer(s)
        """ generate weights ~ G(Q(s)) """
        external_classifier = generator(codes)[0]
        external_classifier = torch.tanh(external_classifier)
        external_classifier = external_classifier.reshape(self.config.k, self.config.hidden_dim, self.config.label_dim).transpose(1, 2)
        loss_list = []
        for i in range(self.config.k):
            loss = self.test(device_cpu, device_gpu, train_loader, lossFun, False, external_classifier[i])
            loss_list.append(loss)
        min_idx = loss_list.index(min(loss_list))
        best_external_classifier = external_classifier[min_idx]

        best_classifier = self.get_classify_weight(best_external_classifier).data
        return best_external_classifier, best_classifier

    def test(self, device_cpu, device_gpu, test_loader, lossFun, verbose=True, external_classifier=None):

        self.eval()
        test_loss = 0
        correct = 0
        correct_class = 0
        numel = 0
        rows = 0
        classification_loss = nn.CrossEntropyLoss()
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device_gpu)
                target = target.to(device_gpu)
                output, classification, z = self(data, external_classifier)
                itEW = [par for name, par in self.named_parameters() if name.endswith("enc.weight")]
                test_loss += lossFun(output, data, next(iter(itEW))) + classification_loss(classification,
                                                                                           target.long())
                numel += output.numel()
                correct += torch.sum(output == data)
                correct_class += torch.sum(torch.argmax(classification.softmax(dim=1), dim=1) == target)
                rows += target.numel()

        _, target = next(iter(test_loader))
        if verbose:
            print(
                '\nTest set: Average loss: {:.6f}, Recon Accuracy: {}/{} ({:.0f}%), Classification Accuracy: {}/{} ({:.0f}%)\n'.format(
                    test_loss, correct, numel, 100. * correct / numel, correct_class, rows,
                    100. * correct_class / rows))
        return test_loss.item()

def init_bi_modal(weight,m1,m2,std, device):
        left = torch.normal(mean=m1,std=std, size=weight.data.shape)
        right = torch.normal(mean=m2,std=std, size=weight.data.shape)
        mask = torch.randint(0,2,size=weight.data.shape)
        weight.data = (left*mask + right*(1-mask)).to(device)






def test_bin(model, device_cpu, device_gpu, test_loader, t_enc=0.3, t_class=0.9):
    model.eval()
    test_loss = 0
    correct = 0
    correct_class = 0
    numel = 0
    rows = 0
    classification_loss = nn.CrossEntropyLoss()
    incorret = []
    gt = []
    incorret_pred = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device_gpu)
            target = target.to(device_gpu)
            output, classification, z = model.forward_test(data,t_enc, t_class)
            itEW = [par for name, par in model.named_parameters() if name.endswith("enc.weight")]
            #test_loss += lossFun(output, data, next(iter(itEW))) + classification_loss(classification,target) 
            numel +=  output.numel()
            correct += torch.sum(output==data)
            correct_class += torch.sum(torch.argmax(classification.softmax(dim=1),dim=1)==target)
            rows += target.numel()
            classi = torch.argmax(classification.softmax(dim=1),dim=1)
            #print(classi.shape)
            #print(target[ind].shape)
            ind = torch.argmax(classification.softmax(dim=1),dim=1)!=target
            incorret.append(data[ind].cpu().numpy())
            gt.append(target[ind].cpu().numpy())
            incorret_pred.append(classi[ind].cpu().numpy())
    _, target = next(iter(test_loader))
    print('\nTest set: Average loss: {:.6f}, Recon Accuracy: {}/{} ({:.0f}%), Classification Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, numel, 100. * correct / numel, correct_class, rows, 100. * correct_class / rows))
    return incorret, incorret_pred, gt

def test_c_bin(model, device_cpu, device_gpu, test_loader, t_enc=0.3, t_class=0.9):
    model.eval()
    test_loss = 0
    correct = 0
    correct_class = 0
    numel = 0
    rows = 0
    classification_loss = nn.CrossEntropyLoss()
    incorret = []
    gt = []
    incorret_pred = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device_gpu)
            target = target.to(device_gpu)
            output, classification, z = model.forward_c_bin(data,t_class)
            itEW = [par for name, par in model.named_parameters() if name.endswith("enc.weight")]
            #test_loss += lossFun(output, data, next(iter(itEW))) + classification_loss(classification,target) 
            numel +=  output.numel()
            correct += torch.sum(output==data)
            correct_class += torch.sum(torch.argmax(classification.softmax(dim=1),dim=1)==target)
            rows += target.numel()
            classi = torch.argmax(classification.softmax(dim=1),dim=1)
            #print(classi.shape)
            #print(target[ind].shape)
            ind = torch.argmax(classification.softmax(dim=1),dim=1)!=target
            incorret.append(data[ind].cpu().numpy())
            gt.append(target[ind].cpu().numpy())
            incorret_pred.append(classi[ind].cpu().numpy())
    _, target = next(iter(test_loader))
    print('\nTest set: Average loss: {:.6f}, Recon Accuracy: {}/{} ({:.0f}%), Classification Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, numel, 100. * correct / numel, correct_class, rows, 100. * correct_class / rows))
    return incorret, incorret_pred, gt

def test_normal(model, device_cpu, device_gpu, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    correct_class = 0
    numel = 0
    rows = 0
    classification_loss = nn.CrossEntropyLoss()
    incorret = []
    gt = []
    incorret_pred = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device_gpu)
            target = target.to(device_gpu)
            output, classification, z = model.forward(data)
            itEW = [par for name, par in model.named_parameters() if name.endswith("enc.weight")]
            #test_loss += lossFun(output, data, next(iter(itEW))) + classification_loss(classification,target) 
            numel +=  output.numel()
            correct += torch.sum(output==data)
            correct_class += torch.sum(torch.argmax(classification.softmax(dim=1),dim=1)==target)
            rows += target.numel()
            classi = torch.argmax(classification.softmax(dim=1),dim=1)
            #print(classi.shape)
            #print(target[ind].shape)
            ind = torch.argmax(classification.softmax(dim=1),dim=1)!=target
            incorret.append(data[ind].cpu().numpy())
            gt.append(target[ind].cpu().numpy())
            incorret_pred.append(classi[ind].cpu().numpy())
    _, target = next(iter(test_loader))
    print('\nTest set: Average loss: {:.6f}, Recon Accuracy: {}/{} ({:.0f}%), Classification Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, numel, 100. * correct / numel, correct_class, rows, 100. * correct_class / rows))
    return incorret, incorret_pred, gt

def update_elb(config):
    config.elb_lamb =  config.elb_lamb * config.regu_rate
    config.elb_k =  config.elb_k * config.regu_rate
    config.class_elb_lamb =  config.class_elb_lamb * config.class_regu_rate
    config.class_elb_k =  config.class_elb_k * config.class_regu_rate

def print_gpu(debug=0):
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    f = (t-r)/1024/1024
    #print("Position %d"%debug)
    #print(round(f,2))

def learn_diffnaps_net(data, config, labels = None, ret_test=False, verbose=True):
    start_time = time.time()

    torch.manual_seed(config.seed)
    torch.set_num_threads(config.thread_num)
    device_cpu = torch.device("cpu")

    if not torch.cuda.is_available():
        device_gpu = device_cpu
        print("WARNING: Running purely on CPU. Slow.")
    else:
        device_gpu = torch.device("cuda")
    if labels is None:
        data_copy = np.copy(data)[:,:-2]
        labels_copy = (data[:,-2] + 2*data[:,-1]).astype(int)
    else:
        data_copy = data
        labels_copy = labels
    
    trainDS = mydl.DiffnapsDatDataset("file", config.train_set_size, True, device_cpu, data=data_copy, labels = labels_copy)
    train_loader = torch.utils.data.DataLoader(trainDS, batch_size=config.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(mydl.DiffnapsDatDataset("file", config.train_set_size, False, device_cpu, data=data_copy, labels = labels_copy), batch_size=config.test_batch_size, shuffle=True)
    
    if config.hidden_dim == -1:
        hidden_dim = trainDS.ncol()
        
    new_weights = torch.zeros(config.hidden_dim, trainDS.ncol(), device=device_gpu)
    initWeights(new_weights, trainDS.data)
    new_weights.clamp_(1/(trainDS.ncol()), 1)
    bInit = torch.zeros(config.hidden_dim, device=device_gpu)
    init.constant_(bInit, -1)

    """HyperNet"""
    hypergan = HyperNet.HyperGAN(config)
    mixer = hypergan.mixer
    generator = hypergan.generator
    Dz = hypergan.discriminator

    """Main Net"""
    model = config.model(new_weights, np.max(labels_copy)+1, bInit, trainDS.getSparsity(), device_cpu, device_gpu, config=config).to(device_gpu)

    """Optimizers"""
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    optimQ = torch.optim.Adam(mixer.parameters(), lr=config.hyper_lr, weight_decay=config.wd)
    optimW = []
    for m in range(config.ngen):
        s = getattr(generator, 'W{}'.format(m + 1))
        optimW.append(torch.optim.Adam(s.parameters(), lr=config.hyper_lr, weight_decay=config.wd))
    optimD = torch.optim.Adam(Dz.parameters(), lr=config.hyper_lr, weight_decay=config.wd)

    lossFun = mylo.weightedXor(trainDS.getSparsity(), config.weight_decay, device_gpu, label_decay = 0, labels=2)

    scheduler = MultiStepLR(optimizer, [5,7], gamma=config.gamma)

    print_gpu()
    for epoch in range(1, config.epochs + 1):
        #print(model.fc0_enc.weight.data.mean())
        #model.learn(device_cpu, device_gpu, train_loader, optimizer, lossFun, epoch, config.log_interval, config, verbose=verbose)

        model.clipWeights()
        model.train()
        classification_loss = nn.CrossEntropyLoss()
        lossFun.config = config
        print_gpu(1)

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device_gpu)
            target = target.to(device_gpu)

            # Q_loss = 0
            s = torch.randn(config.k, config.s).to(config.device)
            z = torch.randn(config.k, config.z).to(config.device)
            codes = mixer(s)
            """ generate weights ~ G(Q(s)) """
            external_classifier = generator(codes)[0]
            external_classifier = torch.tanh(external_classifier[0])
            external_classifier = external_classifier.reshape(config.hidden_dim, config.label_dim).transpose(0, 1)
            """ calculate d_loss, and total loss on Q and G """
            d_loss, d_q = ops.calc_d_loss(config, Dz, z, codes, config.ngen)
            d_loss = d_loss * config.beta
            d_loss.backward(retain_graph=True)

            one_qz = torch.ones((config.k * config.ngen, 1), requires_grad=True).to(config.device)
            log_qz = ops.log_density(torch.ones(config.k * config.ngen, 1), 2).reshape(-1, 1).to(config.device)
            Q_loss = F.binary_cross_entropy_with_logits(d_q + log_qz, one_qz)

            output, classification, z = model(data, external_classifier)
            #output, classification, z = model(data, None, None)
            itEW = [par for name, par in model.named_parameters() if name.endswith("enc.weight")]
            recon_loss = lossFun(output, data, next(iter(itEW)), hidden=z)
            # #print(recon_loss)
            c_loss = classification_loss(classification, target.long())
            c_w = model.get_classify_weight(external_classifier)
            # #print(config.lambda_c * c_loss )
            G_loss = recon_loss + config.lambda_c * c_loss + mylo.elb_regu_class(config.class_elb_k, config.class_elb_lamb, c_w, None) + mylo.horizontal_L2_class(config.wd_class, c_w, None)
            # #QG_loss = Q_loss
            QG_loss = Q_loss + G_loss

            optimizer.zero_grad()
            optimQ.zero_grad()
            optimD.zero_grad()
            for optim in optimW:
                optim.zero_grad()

            # print(mylo.elb_regu(config.class_elb_k, config.class_elb_lamb, c_w, None))
            QG_loss.backward()

            optimizer.step()
            optimQ.step()
            optimD.step()
            for optim in optimW:
                optim.step()

            model.clipWeights()
            if batch_idx % config.log_interval == 0 and verbose:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\talpha(enc):{:.3f}\tbeta(cls):{:.3f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), G_loss.item(),
                        model.fc0_enc.alpha.item(), model.beta.item()))

                # test(model, device_cpu, device_gpu, train_loader, lossFun, verbose=verbose)
                # print("weight gradients:", model.fc0_enc.weight.grad)
                # print("Classifier gradients:", model.classifier.weight.grad)
                # print("Base classifier:", model.classifier.weight)
        # best_external_encoder, best_external_classifier, best_encoder, best_classifier = model.get_best_weights(mixer, generator, device_cpu, device_gpu, train_loader, lossFun, verbose)
        #_ = model.test(device_cpu, device_gpu, test_loader, lossFun, verbose, best_external_encoder, best_external_classifier)
        scheduler.step()
        update_elb(config)

    time_taken = time.time() - start_time
    time_taken = time_taken / 60

    best_external_classifier, best_classifier = model.get_best_weights(mixer, generator, device_cpu, device_gpu, train_loader, lossFun, verbose)
    c_w = best_classifier
    del hypergan
    if ret_test:
        return model, model.fc0_enc.weight.data, trainDS, c_w, time_taken, test_loader
    else:
        return model, model.fc0_enc.weight.data, trainDS, c_w, time_taken