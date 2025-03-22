import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import asdict
from method.hypergan_base import HyperGAN_Base

# """ class model of target network for testing """
#
#
# class Small(nn.Module):
#     def __init__(self):
#         super(Small, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 32, 5, stride=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, 2),
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(32, 32, 5, stride=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, 2),
#         )
#         self.linear = nn.Linear(512, 10)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = x.reshape(-1, 512)
#         x = self.linear(x)
#         return x


class Mixer(nn.Module):
    def __init__(self, args):
        super(Mixer, self).__init__()
        for k, v in args.items():
            setattr(self, k, v)
        self.linear1 = nn.Linear(self.s, 512, bias=self.bias)
        self.linear2 = nn.Linear(512, 512, bias=self.bias)
        self.linear3 = nn.Linear(512, self.z * self.ngen, bias=self.bias)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)

    def forward(self, x):
        x = x.reshape(-1, self.s)  # flatten filter size
        x = torch.zeros_like(x).normal_(0, 0.01) + x
        x = F.relu(self.bn1(self.linear1(x)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        x = x.reshape(-1, self.ngen, self.z)
        w = torch.stack([x[:, i] for i in range(self.ngen)])
        return w


class GeneratorW(nn.Module):
    def __init__(self, params, layer):
        super(GeneratorW, self).__init__()
        self.input_dim = params.z
        self.hidden_layers = params.hyper_hidden_layers
        if layer == 'encoder':
            self.output_dim = params.input_dim * params.hidden_dim
        elif layer == 'classification':
            self.output_dim = params.hidden_dim * params.label_dim
        self.bias = params.bias
        layers = []
        batch_norm_layers = []

        # 构建隐藏层和BatchNorm层
        for i, size in enumerate(self.hidden_layers):
            in_features = self.input_dim if i == 0 else self.hidden_layers[i - 1]
            layer = nn.Linear(in_features, size, bias=self.bias)
            # 使用He初始化，适配ELU
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu')
            layers.append(layer)
            # 创建并添加BatchNorm层
            bn_layer = nn.BatchNorm1d(size)
            layers.append(bn_layer)
            batch_norm_layers.append(bn_layer)
            # 添加ELU激活函数
            layers.append(nn.ELU())

        # 输出层
        output_layer = nn.Linear(self.hidden_layers[-1], self.output_dim, bias=self.bias)
        nn.init.xavier_uniform_(output_layer.weight)  # 使用Xavier初始化
        layers.append(output_layer)

        self.layers = nn.Sequential(*layers)
        self.batch_norm_layers = batch_norm_layers

    def forward(self, x):
        # 如果bias为False，将所有BatchNorm层的偏置置零
        if not self.bias:
            for bn_layer in self.batch_norm_layers:
                bn_layer.bias.data.zero_()

        x = torch.zeros_like(x).normal_(0, 0.01) + x
        return self.layers(x)


class DiscriminatorZ(nn.Module):
    def __init__(self, args):
        super(DiscriminatorZ, self).__init__()
        for k, v in args.items():
            setattr(self, k, v)
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 1)

    def forward(self, x):
        x = x.reshape(-1, self.z)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        x= torch.sigmoid(x)
        return x


class HyperGAN(HyperGAN_Base):

    def __init__(self, args):
        super(HyperGAN, self).__init__(args)
        self.mixer = Mixer(args).to(args.device)
        self.generator = self.Generator(args)
        self.discriminator = DiscriminatorZ(args).to(args.device)

    class Generator(object):
        def __init__(self, args):
            self.W1 = GeneratorW(args, layer='classification').to(args.device)
            self.weights = [self.W1]

        def __call__(self, x):
            layers = [weight(x[i]) for i, weight in enumerate(self.weights)]
            return layers

        def as_list(self):
            return self.weights

    def restore_models(self, args):
        d = torch.load(args.resume)
        self.mixer.load_state_dict(d['mixer']['state_dict'])
        self.discriminator.load_state_dict(d['Dz']['state_dict'])
        generators = self.generator.as_list()
        for i, gen in enumerate(generators):
            gen.load_state_dict(d['W{}'.format(i + 1)]['state_dict'])

    def save_models(self, args, metrics=None):
        save_dict = {
            'mixer': {'state_dict': self.mixer.state_dict()},
            'netD': {'state_dict': self.discriminator.state_dict()}
        }
        for i, gen in enumerate(self.generator.as_list()):
            save_dict['W{}'.format(i + 1)] = {'state_dict': gen.state_dict()}

        path = 'saved_models/{}/{}-{}.pt'.format(args.dataset, args.exp, metrics)
        torch.save(save_dict, path)
