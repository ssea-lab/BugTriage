# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import scaled_Laplacian, cheb_polynomial


class ConvNet(nn.Module):
    def __init__(self, input_c,k_1,k_2,a):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(input_c, k_1, (1, 3, a),nn.ReLU()))
        self.layer2 = nn.Sequential(
            nn.Conv3d(k_1, k_2, (1, 3, k_1)),
            nn.ReLU())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out


class LSTMNet(nn.Module):
    def __init__(self, embedding_dim, num_hiddens, num_layers):
        super(LSTMNet, self).__init__()
        self.encoder = nn.LSTM(input_size=embedding_dim,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               batch_first=True)

        self.decoder = nn.Linear(2 * num_hiddens, 2)

    def forward(self, inputs):
        outputs, _ = self.encoder(inputs)  # output, (h, c)
        outs = self.decoder(outputs)
        return outs



class GRCNN_component(nn.Module):

    def __init__(self, batch_size, N, input_c, k_1, k_2, a, embedding_dim, num_hiddens, num_layers,fc_num1,fc_num2,out_dim):
        super(GRCNN_component, self).__init__()
        self.Conv = ConvNet(input_c,k_1,k_2,a)
        self.LSTM = LSTMNet(embedding_dim, num_hiddens, num_layers)

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, nb_time_filter, T)
        '''
        # cnn
        spatial_cnn = self.Conv(x)
        # lstm
        time_lstm = self.LSTM(spatial_cnn)  # (b,F,N,T)

        return time_lstm


class GRCNN(nn.Module):

    def __init__(self, DEVICE, input_c_hour, input_c_day, input_c_week, k_1, k_2, a, embedding_dim, num_hiddens, num_layers,fc_num1,fc_num2,out_dim):
        super(GRCNN, self).__init__()
        self.ComponentList = nn.ModuleList([GRCNN_component(input_c_hour, k_1, k_2, a, embedding_dim, num_hiddens, num_layers,fc_num1,fc_num2,out_dim)])
        self.ComponentList.extend([GRCNN_component(input_c_day, k_1, k_2, a, embedding_dim, num_hiddens, num_layers,fc_num1,fc_num2,out_dim)])
        self.ComponentList.extend([GRCNN_component(input_c_week, k_1, k_2, a, embedding_dim, num_hiddens, num_layers, fc_num1, fc_num2, out_dim)])
        self.DEVICE = DEVICE
        self.to(DEVICE)

        self.w_omega1 = nn.Parameter(torch.Tensor(
            num_hiddens, num_hiddens))
        self.w_omega2 = nn.Parameter(torch.Tensor(
            num_hiddens, num_hiddens))
        self.a = nn.Parameter(torch.Tensor(num_hiddens, 1))
        self.w_omega3 = nn.Parameter(torch.Tensor(
            num_hiddens, 1))


    def forward(self, x):
        for cl in self.ComponentList:
            h_a = h_a + cl(x)
        e = torch.nn.LeakyReLU(torch.matmul(self.a,torch.cat(torch.matmul(self.ComponentList, self.w_omega1),torch.cat(torch.matmul(cl(x), self.w_omega2)))))
        # att
        att_score = F.softmax(e, dim=1)
        # att_score
        output = torch.tanh(torch.matmul(self.ComponentList,att_score,self.w_omega3))
        return output


def make_model(DEVICE, input_c_hour, input_c_day, input_c_week, k_1, k_2, a, embedding_dim, num_hiddens, num_layers,fc_num1,fc_num2,out_dim):
    model = GRCNN(DEVICE, input_c_hour, input_c_day, input_c_week, k_1, k_2, a, embedding_dim, num_hiddens, num_layers,fc_num1,fc_num2,out_dim)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model