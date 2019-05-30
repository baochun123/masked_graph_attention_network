"""
created by xingxiangrui on 2019.5.30
    Resnet model and score level GATLayer
    GALayer with filter as masked attention
"""

import torchvision.models as models
import torch
from torch import nn
import torch.nn.functional as F
import pickle
import numpy as np

class BGATLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha):
        """
        :param in_features: input's features
        :param out_features: output's features
        :param dropout: attention dropout.
        :param alpha:
        :param is_activate:
        """
        super(BGATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(in_features, out_features))
        nn.init.xavier_uniform(self.W.data, gain=1.414)  # fixme
        self.a = nn.Parameter(torch.zeros(2 * out_features, 1))
        nn.init.xavier_uniform(self.a.data, gain=1.414)  # fixme
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        # self.beta = nn.Parameter(data=torch.ones(1))
        self.beta = nn.Parameter(data=torch.ones(1))  # fixme!!!!!!

        with open('coco_correlations.pkl', 'rb') as f:
            print("loading coco_correlations.pkl from local...")
            correlations = pickle.load(f)
        # with open('coco_names.pkl', 'rb') as f:
        #     print("loading coco_names.pkl")
        #     self.names = pickle.load(f)
        self.coco_correlation_A_B = correlations['pp']
        self.probability_filter_threshold=0.1
        self.mask=torch.FloatTensor(np.where(self.coco_correlation_A_B>self.probability_filter_threshold,1,0))
        # self.register_parameter('beta', self.beta)

    def forward(self, x):
        # [B:batch ,N:Number_of_classes ,C:channels=1]
        B, N, C = x.size()
        # h = torch.bmm(x, self.W.expand(B, self.in_features, self.out_features))  # [B,N,C]
        h = torch.matmul(x, self.W)  # [B,N,C]
        a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, C), h.repeat(1, N, 1)], dim=2).view(B, N, N,
                                                                                                  2 * self.out_features)  # [B,N,N,2C]
        # temp = self.a.expand(B, self.out_features * 2, 1)
        # temp2 = torch.matmul(a_input, self.a)
        attention = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))  # [Batch,N_clsses,N_classes]

        # fixme add masked attention
        mask=self.mask.cuda()
        attention= attention.mul(mask)

        attention = F.softmax(attention, dim=2)  # [B,N,N]
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.bmm(attention, h)  # [B,N,N]*[B,N,C]-> [B,N,C]
        out = F.elu(h_prime + self.beta * h)
        return out


class Head(nn.Module):
    def __init__(self, nclasses):
        super(Head, self).__init__()
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(2048, nclasses)
        self.gat = BGATLayer(in_features=1, out_features=1, dropout=0, alpha=0.2)


    def forward(self, x):
        # size [batch, channels=2048, W=14 , H=14 ]
        B, C, _, _ = x.size()
        # output x [ Batch, channels , W]
        x = self.gmp(x).view(B, C)
        x = self.fc(x)
        x=x.view(x.size(0),x.size(1),1)
        residual=x
        output=residual+self.gat(x)
        output=output.squeeze(2)
        return output


class Resnet(nn.Module):
    def __init__(self, nclasses, backbone):
        super(Resnet, self).__init__()
        if backbone == 'resnet101':
            model = models.resnet101(pretrained=False)
            print('loading pretrained resnet101 from local...')
            model.load_state_dict(torch.load('./resnet101-5d3b4d8f.pth'))
        else:
            raise Exception()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4, )
        self.heads = Head(nclasses)

    def forward(self, x, embedding=None):
        x = self.features(x)  # [B,2048,H,W]
        output = self.heads(x)
        return output

    def get_config_optim(self, lr, lrp):
        return [
            {'params': self.features.parameters(), 'lr': lrp},
            {'params': self.heads.parameters(), 'lr': lr},
        ]


if __name__ == '__main__':
    model = Resnet(backbone='resnet101', nclasses=80)

    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    x = torch.zeros(2, 3, 448, 448).random_(0, 10)
    x.requires_grad = True
    out = model(x)
    print('program end...')
    # model=models.(pretrained=False)
    # print('Number of model parameters: {}'.format(
    #     sum([p.data.nelement() for p in model.parameters()])))
