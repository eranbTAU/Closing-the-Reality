import torch
import torch.nn as nn
from torch.nn import init

def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)

def init_weights(net, init_type='xavier', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)

# Position Invariant
class ConFNet1(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConFNet1, self).__init__()
        self.cnn1 = nn.Conv1d(1, 8, kernel_size=1, stride=1, padding=1)
        self.cnn2 = nn.Conv1d(8, 64, kernel_size=1, stride=1, padding=1)
        self.rnn = nn.LSTM(input_size=7, hidden_size=5, num_layers=1, batch_first=True)
        self.linear_1 = nn.Linear(320, 256)
        self.linear_2 = nn.Linear(256, 8)
        self.linear_3 = nn.Linear(8, 128)
        self.linear_4 = nn.Linear(128, 32)
        self.linear_5 = nn.Linear(32, 8)
        self.linear_6 = nn.Linear(8, 16)
        self.linear_7 = nn.Linear(16, 4)
        self.linear_8 = nn.Linear(4, output_dim)
        self.activation = nn.Tanh()
        self.bn1 = nn.BatchNorm1d(num_features=256)
        self.bn2 = nn.BatchNorm1d(num_features=8)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.bn4 = nn.BatchNorm1d(num_features=32)
        self.bn5 = nn.BatchNorm1d(num_features=8)
        self.bnc = nn.BatchNorm1d(num_features=8)
        self.bnc2 = nn.BatchNorm1d(num_features=64)
        self.Dropout = nn.Dropout(0.1)
        self.LKR = nn.LeakyReLU(negative_slope=0.01)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_c = self.LKR(self.bnc(self.cnn1(x)))
        x_c2 = self.LKR(self.bnc2(self.cnn2(x_c)))
        outRNN, h_h = self.rnn(x_c2)
        x_flatten = outRNN.reshape(-1, 64*5)
        x1 = self.activation(self.bn1(self.linear_1(x_flatten)))
        x1_d = self.Dropout(x1)
        x2 = self.activation(self.bn2(self.linear_2(x1_d)))
        x2_d = self.Dropout(x2)
        # x2_d = x2_d + x1_d
        x3 = self.activation(self.bn3(self.linear_3(x2_d)))
        x3_d = self.Dropout(x3)
        # x3_d = x3_d + x2_d
        x4 = self.activation(self.bn4(self.linear_4(x3_d)))
        x4_4 = self.Dropout(x4)
        # x4_4 = x4_4 + x1_d
        x5 = self.activation(self.bn5(self.linear_5(x4_4)))
        x5 = x2_d + x5
        x6 = self.activation(self.linear_6(x5))
        # x6 = x6 + x4_4
        x7 = self.activation(self.linear_7(x6))
        # x7 = x7 + x3_d
        x8 = self.Sigmoid(self.linear_8(x7))
        return x8

# Position conditioned
class ConFNet2(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConFNet2, self).__init__()
        self.cnn1 = nn.Conv1d(1, 8, kernel_size=1, stride=1, padding=1)
        self.cnn2 = nn.Conv1d(8, 64, kernel_size=1, stride=1, padding=1)
        self.rnn = nn.LSTM(input_size=10, hidden_size=5, num_layers=1, batch_first=True)
        self.linear_1 = nn.Linear(320, 256)
        self.linear_2 = nn.Linear(256, 8)
        self.linear_3 = nn.Linear(8, 128)
        self.linear_4 = nn.Linear(128, 32)
        self.linear_5 = nn.Linear(32, 8)
        self.linear_6 = nn.Linear(8, 16)
        self.linear_7 = nn.Linear(16, 4)
        self.linear_8 = nn.Linear(4, output_dim)
        self.activation = nn.Tanh()
        self.bn1 = nn.BatchNorm1d(num_features=256)
        self.bn2 = nn.BatchNorm1d(num_features=8)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.bn4 = nn.BatchNorm1d(num_features=32)
        self.bn5 = nn.BatchNorm1d(num_features=8)
        self.bnc = nn.BatchNorm1d(num_features=8)
        self.bnc2 = nn.BatchNorm1d(num_features=64)
        self.Dropout = nn.Dropout(0.1)
        self.LKR = nn.LeakyReLU(negative_slope=0.01)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_c = self.LKR(self.bnc(self.cnn1(x)))
        x_c2 = self.LKR(self.bnc2(self.cnn2(x_c)))
        outRNN, h_h = self.rnn(x_c2)
        x_flatten = outRNN.reshape(-1, 64*5)
        x1 = self.activation(self.bn1(self.linear_1(x_flatten)))
        x1_d = self.Dropout(x1)
        x2 = self.activation(self.bn2(self.linear_2(x1_d)))
        x2_d = self.Dropout(x2)
        # x2_d = x2_d + x1_d
        x3 = self.activation(self.bn3(self.linear_3(x2_d)))
        x3_d = self.Dropout(x3)
        # x3_d = x3_d + x2_d
        x4 = self.activation(self.bn4(self.linear_4(x3_d)))
        x4_4 = self.Dropout(x4)
        # x4_4 = x4_4 + x1_d
        x5 = self.activation(self.bn5(self.linear_5(x4_4)))
        x5 = x2_d + x5
        x6 = self.activation(self.linear_6(x5))
        # x6 = x6 + x4_4
        x7 = self.activation(self.linear_7(x6))
        # x7 = x7 + x3_d
        x8 = self.Sigmoid(self.linear_8(x7))
        return x8

class ConFNet3(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConFNet3, self).__init__()
        self.cnn1 = nn.Conv1d(1, 8, kernel_size=1, stride=1, padding=1)
        self.cnn2 = nn.Conv1d(8, 64, kernel_size=1, stride=1, padding=1)
        self.rnn = nn.LSTM(input_size=10, hidden_size=5, num_layers=1, batch_first=True)
        self.linear_1 = nn.Linear(320, 256)
        self.linear_2 = nn.Linear(256, 8)
        self.linear_3 = nn.Linear(8, 128)
        self.linear_4 = nn.Linear(128, 32)
        self.linear_5 = nn.Linear(32, 8)
        self.linear_6 = nn.Linear(8, 16)
        self.linear_7 = nn.Linear(16, 4)
        self.linear_8 = nn.Linear(4, output_dim)
        self.activation = nn.Tanh()
        self.bn1 = nn.BatchNorm1d(num_features=256)
        self.bn2 = nn.BatchNorm1d(num_features=8)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.bn4 = nn.BatchNorm1d(num_features=32)
        self.bn5 = nn.BatchNorm1d(num_features=8)
        self.bnc = nn.BatchNorm1d(num_features=8)
        self.bnc2 = nn.BatchNorm1d(num_features=64)
        self.Dropout = nn.Dropout(0.1)
        self.LKR = nn.LeakyReLU(negative_slope=0.01)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_c = self.LKR(self.bnc(self.cnn1(x)))
        x_c2 = self.LKR(self.bnc2(self.cnn2(x_c)))
        outRNN, h_h = self.rnn(x_c2)
        x_flatten = outRNN.reshape(-1, 64*5)
        x1 = self.activation(self.bn1(self.linear_1(x_flatten)))
        x1_d = self.Dropout(x1)
        x2 = self.activation(self.bn2(self.linear_2(x1_d)))
        x2_d = self.Dropout(x2)
        # x2_d = x2_d + x1_d
        x3 = self.activation(self.bn3(self.linear_3(x2_d)))
        x3_d = self.Dropout(x3)
        # x3_d = x3_d + x2_d
        x4 = self.activation(self.bn4(self.linear_4(x3_d)))
        x4_4 = self.Dropout(x4)
        # x4_4 = x4_4 + x1_d
        x5 = self.activation(self.bn5(self.linear_5(x4_4)))
        x5 = x2_d + x5
        x6 = self.activation(self.linear_6(x5))
        # x6 = x6 + x4_4
        x7 = self.activation(self.linear_7(x6))
        # x7 = x7 + x3_d
        x8 = self.Sigmoid(self.linear_8(x7))
        return x8

class ConFNet4(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConFNet4, self).__init__()
        self.cnn1 = nn.Conv1d(1, 8, kernel_size=1, stride=1, padding=1)
        self.cnn2 = nn.Conv1d(8, 64, kernel_size=1, stride=1, padding=1)
        self.rnn = nn.LSTM(input_size=10, hidden_size=5, num_layers=1, batch_first=True)
        self.linear_1 = nn.Linear(320, 256)
        self.linear_2 = nn.Linear(256, 8)
        self.linear_3 = nn.Linear(8, 128)
        self.linear_4 = nn.Linear(128, 32)
        self.linear_5 = nn.Linear(32, 8)
        self.linear_6 = nn.Linear(8, 16)
        self.linear_7 = nn.Linear(16, 4)
        self.linear_8 = nn.Linear(4, output_dim)
        self.activation = nn.Tanh()
        self.bn1 = nn.BatchNorm1d(num_features=256)
        self.bn2 = nn.BatchNorm1d(num_features=8)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.bn4 = nn.BatchNorm1d(num_features=32)
        self.bn5 = nn.BatchNorm1d(num_features=8)
        self.bnc = nn.BatchNorm1d(num_features=8)
        self.bnc2 = nn.BatchNorm1d(num_features=64)
        self.Dropout = nn.Dropout(0.1)
        self.LKR = nn.LeakyReLU(negative_slope=0.01)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_c = self.LKR(self.bnc(self.cnn1(x)))
        x_c2 = self.LKR(self.bnc2(self.cnn2(x_c)))
        outRNN, h_h = self.rnn(x_c2)
        x_flatten = outRNN.reshape(-1, 64*5)
        x1 = self.activation(self.bn1(self.linear_1(x_flatten)))
        x1_d = self.Dropout(x1)
        x2 = self.activation(self.bn2(self.linear_2(x1_d)))
        x2_d = self.Dropout(x2)
        # x2_d = x2_d + x1_d
        x3 = self.activation(self.bn3(self.linear_3(x2_d)))
        x3_d = self.Dropout(x3)
        # x3_d = x3_d + x2_d
        x4 = self.activation(self.bn4(self.linear_4(x3_d)))
        x4_4 = self.Dropout(x4)
        # x4_4 = x4_4 + x1_d
        x5 = self.activation(self.bn5(self.linear_5(x4_4)))
        x5 = x2_d + x5
        x6 = self.activation(self.linear_6(x5))
        # x6 = x6 + x4_4
        x7 = self.activation(self.linear_7(x6))
        # x7 = x7 + x3_d
        x8 = self.Sigmoid(self.linear_8(x7))
        return x8

dispatch = {'model_Wed_Oct_27_15_46_42.pt': {(ConFNet1, [-79, 79, 0, 60, 0, 8, -78.7, 70.39, -74.28, 99.92, -1.59, 1.87])},
            'model_Thu_Oct_28_15_05_50.pt': {(ConFNet2, [-79, 79, 0, 60, 0, 8, -78.7, 70.39, -74.28, 99.92, -1.59, 1.87])},
            'model_Mon_Nov_15_14_46_49.pt': {(ConFNet3, [-79, 79, 0, 60, 0, 8, -78.7, 70.39, -74.28, 99.92, -1.59, 1.87])},
            'model_Sun_Nov_21_16_17_39.pt': {(ConFNet4, [-79, 79, 0, 60, 0, 8, -78.7, 70.39, -74.28, 99.92, -1.59, 1.87])}}

def get_fd_model(name):
    return dispatch[name]