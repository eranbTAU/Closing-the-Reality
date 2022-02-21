import torch
import torch.nn as nn
from torch.nn import init

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

class FNetSkip(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FNetSkip, self).__init__()
        self.linear_1 = nn.Linear(input_dim, 256)
        self.linear_2 = nn.Linear(256, 4)
        self.linear_3 = nn.Linear(4, 64)
        self.linear_4 = nn.Linear(64, 256)
        self.linear_5 = nn.Linear(256, 64)
        self.linear_6 = nn.Linear(64, 4)
        self.linear_7 = nn.Linear(4, 64)
        self.linear_8 = nn.Linear(64, output_dim)
        self.activation = nn.Sigmoid()
        self.bn1 = nn.BatchNorm1d(num_features=256)
        self.bn2 = nn.BatchNorm1d(num_features=4)
        self.bn3 = nn.BatchNorm1d(num_features=64)
        self.bn4 = nn.BatchNorm1d(num_features=256)
        self.bn5 = nn.BatchNorm1d(num_features=64)

        self.Dropout = nn.Dropout(0.2)

    def forward(self, x):
        x1 = self.activation(self.bn1(self.linear_1(x)))
        x1_d = self.Dropout(x1)
        x2 = self.activation(self.bn2(self.linear_2(x1_d)))
        x2_d = self.Dropout(x2)
        x3 = self.activation(self.bn3(self.linear_3(x2_d)))
        x3_d = self.Dropout(x3)
        x4 = self.activation(self.bn4(self.linear_4(x3_d)))
        x4_4 = self.Dropout(x4)
        x5 = self.activation(self.bn5(self.linear_5(x4_4)))
        x5 = x5 + x5
        x6 = self.activation(self.linear_6(x5))
        x6 = x6 + x6
        x7 = self.activation(self.linear_7(x6))
        x8 = self.activation(self.linear_8(x7))
        return x8

class FNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FNet, self).__init__()
        self.linear_1 = nn.Linear(input_dim, 64)
        self.linear_2 = nn.Linear(64, 8)
        self.linear_3 = nn.Linear(8, 256)
        self.linear_4 = nn.Linear(256, 128)
        self.linear_5 = nn.Linear(128, output_dim)
        self.activation = nn.Sigmoid()
        self.bn1 = nn.BatchNorm1d(num_features=1)
        self.bn2 = nn.BatchNorm1d(num_features=64)
        self.bn3 = nn.BatchNorm1d(num_features=8)
        self.bn4 = nn.BatchNorm1d(num_features=256)

    def forward(self, x):
        x = self.activation(self.linear_1(x))
        x = self.activation(self.linear_2(x))
        x = self.activation(self.linear_3(x))
        x = self.activation(self.linear_4(x))
        x = self.activation(self.linear_5(x))
        return x

class ConFNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConFNet, self).__init__()
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
        self.rnn = nn.LSTM(input_size=10, hidden_size=5, num_layers=2, batch_first=True)
        self.linear_1 = nn.Linear(320, 1024)
        self.linear_2 = nn.Linear(1024, 2048)
        self.linear_3 = nn.Linear(2048, 512)
        self.linear_4 = nn.Linear(512, 2048)
        self.linear_5 = nn.Linear(2048, 2048)
        self.linear_6 = nn.Linear(2048, 512)
        self.linear_7 = nn.Linear(512, 32)
        self.linear_8 = nn.Linear(32, output_dim)
        self.activation = nn.Tanh()#nn.ReLU()
        self.bn1 = nn.BatchNorm1d(num_features=1024)
        self.bn2 = nn.BatchNorm1d(num_features=2048)
        self.bn3 = nn.BatchNorm1d(num_features=512)
        self.bn4 = nn.BatchNorm1d(num_features=2048)
        self.bn5 = nn.BatchNorm1d(num_features=2048)
        self.bn6 = nn.BatchNorm1d(num_features=512)
        self.bn7 = nn.BatchNorm1d(num_features=32)
        self.bnc = nn.BatchNorm1d(num_features=8)
        self.bnc2 = nn.BatchNorm1d(num_features=64)
        self.Dropout = nn.Dropout(0.1)
        self.LKR = nn.LeakyReLU(negative_slope=0.01)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_c = self.LKR(self.bnc(self.cnn1(x)))
        # x_c = self.Dropout(x_c)
        x_c2 = self.LKR(self.bnc2(self.cnn2(x_c)))
        # x_c2 = self.Dropout(x_c2)
        outRNN, h_h = self.rnn(x_c2)
        x_flatten = outRNN.reshape(-1, 64*5)
        x1 = self.activation(self.bn1(self.linear_1(x_flatten)))
        # x1 = self.activation(self.linear_1(x_flatten))
        # x1 = self.Dropout(x1)
        x2 = self.activation(self.bn2(self.linear_2(x1)))
        # x2 = self.activation(self.linear_2(x1))
        # x2 = self.Dropout(x2)
        # x2_d = x2_d + x1_d
        x3 = self.activation(self.bn3(self.linear_3(x2)))
        # x3 = self.activation(self.linear_3(x2))
        # x3 = self.Dropout(x3)
        # x3_d = x3_d + x2_d
        x4 = self.activation(self.bn4(self.linear_4(x3)))
        # x4 = self.activation(self.linear_4(x3))
        # x4 = self.Dropout(x4)
        x4 = x4 + x2
        x5 = self.activation(self.bn5(self.linear_5(x4)))
        # x5 = self.activation(self.linear_5(x4))
        # x5 = self.Dropout(x5)
        x5 = x4 + x5
        x6 = self.activation(self.bn6(self.linear_6(x5)))
        # x6 = self.activation(self.linear_6(x5))
        # x6 = self.Dropout(x6)
        x6 = x6 + x3
        x7 = self.activation(self.bn7(self.linear_7(x6)))
        # x7 = self.activation(self.linear_7(x6))
        # x7 = self.Dropout(x7)
        # x7_d = x7_d + x3_d
        x8 = self.linear_8(x7)
        return x8

class CNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNN, self).__init__()
        self.cnn1 = nn.Conv1d(1, 8, kernel_size=1, stride=1, padding=1)
        self.cnn2 = nn.Conv1d(8, 64, kernel_size=1, stride=1, padding=1)
        self.cnn3 = nn.Conv1d(64, 96, kernel_size=1, stride=1, padding=1)
        self.cnn4 = nn.Conv1d(96, 32, kernel_size=1, stride=1, padding=1)
        self.fc1 = nn.Linear(448, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_dim)

        self.bn1 = nn.BatchNorm1d(num_features=8)
        self.bn2 = nn.BatchNorm1d(num_features=64)
        self.bn3 = nn.BatchNorm1d(num_features=96)
        self.bn4 = nn.BatchNorm1d(num_features=32)
        self.LKR = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x_c = self.LKR(self.bn1(self.cnn1(x)))
        x_c2 = self.LKR(self.bn2(self.cnn2(x_c)))
        x_c3 = self.LKR(self.bn3(self.cnn3(x_c2)))
        x_c4 = self.LKR(self.bn4(self.cnn4(x_c3)))
        x_flatten = x_c4.view(-1, 32 * 14)
        x_c5 = self.LKR(self.fc1(x_flatten))
        x_c6 = self.LKR(self.fc2(x_c5))
        x_c7 = self.LKR(self.fc3(x_c6))
        return x_c7

class LSTM(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LSTM, self).__init__()
        self.num_layers = 1  # number of layers
        self.input_size = input_dim  # input size
        self.hidden_size = 2  # hidden state

        self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                           num_layers=self.num_layers, batch_first=True)

        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 80)
        self.fc3 = nn.Linear(80, 16)
        self.fc4 = nn.Linear(16, output_dim)

        self.bn1 = nn.BatchNorm1d(num_features=32)
        self.bn2 = nn.BatchNorm1d(num_features=80)
        self.bn3 = nn.BatchNorm1d(num_features=16)
        self.LKR = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        outRNN, h_h = self.rnn(x)
        x_flatten = outRNN.reshape(-1, 1*2)
        x_f1 = self.LKR(self.bn1(self.fc1(x_flatten)))
        x_f2 = self.LKR(self.bn2(self.fc2(x_f1)))
        x_f3 = self.LKR(self.bn3(self.fc3(x_f2)))
        x_f4 = self.LKR(self.fc4(x_f3))
        return x_f4


class GRU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GRU, self).__init__()
        self.num_layers = 1  # number of layers
        self.input_size = input_dim  # input size
        self.hidden_size = 1  # hidden state

        self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size,
                           num_layers=self.num_layers, batch_first=True)

        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, output_dim)

        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.bn2 = nn.BatchNorm1d(num_features=128)
        self.bn3 = nn.BatchNorm1d(num_features=32)
        self.LKR = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        outRNN, h_h = self.rnn(x)
        x_flatten = outRNN.reshape(-1, 1*1)
        x_f1 = self.LKR(self.bn1(self.fc1(x_flatten)))
        x_f2 = self.LKR(self.bn2(self.fc2(x_f1)))
        x_f3 = self.LKR(self.bn3(self.fc3(x_f2)))
        x_f4 = self.LKR(self.fc4(x_f3))
        return x_f4