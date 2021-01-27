# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import scaled_Laplacian, cheb_polynomial


class ConvNet(nn.Module):
    def __init__(self, input_c,k_1,k_2,a):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(input_c,k_1,(1, 3, a),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv3d(k_1, k_2, (1, 3, k_1)),
            nn.ReLU())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out


class LSTM_Attention(nn.Module):
    def __init__(self, embedding_dim, num_hiddens, num_layers):
        super(LSTM_Attention, self).__init__()
        # embedding之后的shape: torch.Size([200, 8, 300])
        self.encoder = nn.LSTM(input_size=embedding_dim,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               batch_first=True)
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.w_omega = nn.Parameter(torch.Tensor(
            num_hiddens * 2, num_hiddens * 2))
        self.u_omega = nn.Parameter(torch.Tensor(num_hiddens * 2, 1))
        self.decoder = nn.Linear(2 * num_hiddens, 2)

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, inputs):
        #inputs形状为(seq_len,batch_size,embedding_dim)
        # rnn.LSTM只返回最后一层的隐藏层在各时间步的隐藏状态。
        outputs, _ = self.encoder(inputs)  # output, (h, c)
        # outputs形状是(seq_len,batch_size, 2 * num_hiddens)
        x = outputs.permute(1, 0, 2)
        # x形状是(batch_size, seq_len, 2 * num_hiddens)

        # Attention过程
        u = torch.tanh(torch.matmul(x, self.w_omega))
        # u形状是(batch_size, seq_len, 2 * num_hiddens)
        att = torch.matmul(u, self.u_omega)
        # att形状是(batch_size, seq_len, 1)
        att_score = F.softmax(att, dim=1)
        # att_score形状仍为(batch_size, seq_len, 1)
        scored_x = x * att_score
        # scored_x形状是(batch_size, seq_len, 2 * num_hiddens)
        # Attention过程结束

        feat = torch.sum(scored_x, dim=1)
        # feat形状是(batch_size, 2 * num_hiddens)
        outs = self.decoder(feat)
        # out形状是(batch_size, 2)
        return outs



class GRCNN_component(nn.Module):

    def __init__(self, batch_size, N, input_c, k_1, k_2, a, embedding_dim, num_hiddens, num_layers,fc_num1,fc_num2,out_dim):
        super(GRCNN_component, self).__init__()
        self.Conv = ConvNet(input_c,k_1,k_2,a)
        self.ALSTM = LSTM_Attention(embedding_dim, num_hiddens, num_layers)
        self.fc1 = torch.nn.Linear(fc_num1, fc_num2)
        self.fc2 = torch.nn.Linear(fc_num2, out_dim)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        self.w_p = nn.Parameter(torch.Tensor(out_dim, N))
        nn.init.uniform_(self.w_p, -1, 1)

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, nb_time_filter, T)
        '''
        # cnn
        spatial_cnn = self.Conv(x)
        # Alstm
        time_Alstm = self.ALSTM(spatial_cnn)  # (b,F,N,T)

        out = self.fc1(time_Alstm)
        out = self.fc2(x)
        out = self.softmax(x)

        return out*self.w_p


class GRCNN(nn.Module):

    def __init__(self, DEVICE, input_c_hour, input_c_day, input_c_week, k_1, k_2, a, embedding_dim, num_hiddens, num_layers,fc_num1,fc_num2,out_dim):
        super(GRCNN, self).__init__()
        self.ComponentList = nn.ModuleList([GRCNN_component(input_c_hour, k_1, k_2, a, embedding_dim, num_hiddens, num_layers,fc_num1,fc_num2,out_dim)])
        self.ComponentList.extend([GRCNN_component(input_c_day, k_1, k_2, a, embedding_dim, num_hiddens, num_layers,fc_num1,fc_num2,out_dim)])
        self.ComponentList.extend([GRCNN_component(input_c_week, k_1, k_2, a, embedding_dim, num_hiddens, num_layers, fc_num1, fc_num2, out_dim)])
        self.DEVICE = DEVICE
        self.to(DEVICE)

    def forward(self, x):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        for cl in self.ComponentList:
            sum_out = sum_out + cl(x)
        output = nn.Softmax(sum_out)

        return output


def make_model(DEVICE, input_c_hour, input_c_day, input_c_week, k_1, k_2, a, embedding_dim, num_hiddens, num_layers,fc_num1,fc_num2,out_dim):
    L_tilde = scaled_Laplacian(adj_mx)
    cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE) for i in cheb_polynomial(L_tilde, K)]
    model = GRCNN(DEVICE, nb_block, input_c_hour, input_c_day, input_c_week, k_1, k_2, a, embedding_dim, num_hiddens, num_layers,fc_num1,fc_num2,out_dim)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model