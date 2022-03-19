import torch
import torch.nn as nn

class FNetSkip(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FNetSkip, self).__init__()
        self.linear_1 = nn.Linear(input_dim, 320)
        self.linear_2 = nn.Linear(320, 8)
        self.linear_3 = nn.Linear(8, 512)
        self.linear_4 = nn.Linear(512, 256)
        self.linear_5 = nn.Linear(256, 16)
        self.linear_6 = nn.Linear(16, 128)
        self.linear_7 = nn.Linear(128, 64)
        self.linear_8 = nn.Linear(64, output_dim)
        self.activation = nn.Tanh()
        self.bn1 = nn.BatchNorm1d(num_features=320)
        self.bn2 = nn.BatchNorm1d(num_features=8)
        self.bn3 = nn.BatchNorm1d(num_features=512)
        self.bn4 = nn.BatchNorm1d(num_features=256)
        self.bn5 = nn.BatchNorm1d(num_features=16)
        self.Sigmoid = nn.Sigmoid()
        self.Dropout = nn.Dropout(0.2)

    def forward(self, x):
        x1 = self.activation(self.bn1(self.linear_1(x)))
        # x1_d = self.Dropout(x1)
        x2 = self.activation(self.bn2(self.linear_2(x1)))
        # x2_d = self.Dropout(x2)
        x3 = self.activation(self.bn3(self.linear_3(x2)))
        # x3_d = self.Dropout(x3)
        x4 = self.activation(self.bn4(self.linear_4(x3)))
        # x4_4 = self.Dropout(x4)
        x5 = self.activation(self.bn5(self.linear_5(x4)))
        # x5 = x5 + x5
        x6 = self.activation(self.linear_6(x5))
        # x6 = x6 + x6
        x7 = self.activation(self.linear_7(x6))
        # x7 = x7 + x6
        x8 = self.Sigmoid(self.linear_8(x7))
        return x8


class FNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FNet, self).__init__()
        self.linear_1 = nn.Linear(input_dim, 320)
        self.linear_2 = nn.Linear(320, 256)
        self.linear_3 = nn.Linear(256, 128)
        self.linear_4 = nn.Linear(128, 64)
        self.linear_5 = nn.Linear(64, output_dim)
        self.activation = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()
        self.bn1 = nn.BatchNorm1d(num_features=320)
        self.bn2 = nn.BatchNorm1d(num_features=256)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.bn4 = nn.BatchNorm1d(num_features=64)

    def forward(self, x):
        x = self.Sigmoid(self.bn1(self.linear_1(x)))
        x = self.Sigmoid(self.bn2(self.linear_2(x)))
        x = self.Sigmoid(self.bn3(self.linear_3(x)))
        x = self.Sigmoid(self.bn4(self.linear_4(x)))
        x = self.activation(self.linear_5(x))
        return x


# state_42_Sun_Jul_18_21_51_26_ConFNetRNN.pt
class ConFNet1(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConFNet1, self).__init__()
        self.cnn1 = nn.Conv1d(1, 8, kernel_size=1, stride=1, padding=1)
        self.cnn2 = nn.Conv1d(8, 64, kernel_size=1, stride=1, padding=1)
        self.rnn = nn.RNN(input_size=6, hidden_size=5, num_layers=3, batch_first=True)
        self.linear_1 = nn.Linear(320, 256)
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
        self.bnc = nn.BatchNorm1d(num_features=8)
        self.bnc2 = nn.BatchNorm1d(num_features=64)
        self.Dropout = nn.Dropout(0.2)
        self.LRelu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x = x.unsqueeze(1)
        x_c = self.LRelu(self.bnc(self.cnn1(x)))
        x_c2 = self.LRelu(self.bnc2(self.cnn2(x_c)))
        outRNN, h_h = self.rnn(x_c2)
        x_flatten = outRNN.reshape(-1, 64*5)
        # -----
        # outRNN, h_h = self.rnn(x_c2, self.h0)
        # x_flatten = x_c2.view(-1, 64 * 6)
        # -----
        x1 = self.activation(self.bn1(self.linear_1(x_flatten)))
        x1_d = self.Dropout(x1)
        x2 = self.activation(self.bn2(self.linear_2(x1_d)))
        x2_d = self.Dropout(x2)
        x3 = self.activation(self.bn3(self.linear_3(x2_d)))
        x3_d = self.Dropout(x3)
        x4 = self.activation(self.bn4(self.linear_4(x3_d)))
        x4_4 = self.Dropout(x4)
        x5 = self.activation(self.bn5(self.linear_5(x4_4)))
        x5 = x3 + x5
        x6 = self.activation(self.linear_6(x5))
        x6 = x2 + x6
        x7 = self.activation(self.linear_7(x6))
        x8 = self.activation(self.linear_8(x7))
        return x8

# state_99_Mon_Jul_26_00_56_28.pt
class ConFNet2(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConFNet2, self).__init__()
        self.cnn1 = nn.Conv1d(1, 8, kernel_size=1, stride=1, padding=1)
        self.cnn2 = nn.Conv1d(8, 64, kernel_size=1, stride=1, padding=1)
        self.rnn = nn.RNN(input_size=6, hidden_size=5, num_layers=3, batch_first=True)
        self.linear_1 = nn.Linear(320, 256)
        self.linear_2 = nn.Linear(256, 32)
        self.linear_3 = nn.Linear(32, 128)
        self.linear_4 = nn.Linear(128, 16)
        self.linear_5 = nn.Linear(16, 128)
        self.linear_6 = nn.Linear(128, 64)
        self.linear_7 = nn.Linear(64, 512)
        self.linear_8 = nn.Linear(512, output_dim)
        self.activation = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(num_features=256)
        self.bn2 = nn.BatchNorm1d(num_features=32)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.bn4 = nn.BatchNorm1d(num_features=16)
        self.bn5 = nn.BatchNorm1d(num_features=128)
        self.bnc = nn.BatchNorm1d(num_features=8)
        self.bnc2 = nn.BatchNorm1d(num_features=64)
        self.Dropout = nn.Dropout(0.1)
        self.Tanh = nn.Tanh()

    def forward(self, x):
        x = x.unsqueeze(1)
        x_c = self.activation(self.bnc(self.cnn1(x)))
        x_c2 = self.activation(self.bnc2(self.cnn2(x_c)))
        outRNN, h_h = self.rnn(x_c2)
        x_flatten = outRNN.reshape(-1, 64*5)
        x1 = self.activation(self.bn1(self.linear_1(x_flatten)))
        x1_d = self.Dropout(x1)
        x2 = self.activation(self.bn2(self.linear_2(x1_d)))
        x2_d = self.Dropout(x2)
        x3 = self.activation(self.bn3(self.linear_3(x2_d)))
        x3_d = self.Dropout(x3)
        x4 = self.activation(self.bn4(self.linear_4(x3_d)))
        x4_4 = self.Dropout(x4)
        x5 = self.activation(self.bn5(self.linear_5(x4_4)))
        x5 = x3 + x5
        x6 = self.activation(self.linear_6(x5))
        x7 = self.activation(self.linear_7(x6))
        x8 = self.Tanh(self.linear_8(x7))
        return x8

# state_87_Mon_Jul_26_15_54_36.pt
class ConFNet3(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConFNet3, self).__init__()
        self.cnn1 = nn.Conv1d(1, 8, kernel_size=1, stride=1, padding=1)
        self.cnn2 = nn.Conv1d(8, 64, kernel_size=1, stride=1, padding=1)
        self.rnn = nn.RNN(input_size=6, hidden_size=5, num_layers=4, batch_first=True)
        self.linear_1 = nn.Linear(320, 512)
        self.linear_2 = nn.Linear(512, 512)
        self.linear_3 = nn.Linear(512, 512)
        self.linear_4 = nn.Linear(512, 256)
        self.linear_5 = nn.Linear(256, 64)
        self.linear_6 = nn.Linear(64, 8)
        self.linear_7 = nn.Linear(8, 128)
        self.linear_8 = nn.Linear(128, output_dim)
        self.activation = nn.Tanh()
        self.bn1 = nn.BatchNorm1d(num_features=512)
        self.bn2 = nn.BatchNorm1d(num_features=512)
        self.bn3 = nn.BatchNorm1d(num_features=512)
        self.bn4 = nn.BatchNorm1d(num_features=256)
        self.bn5 = nn.BatchNorm1d(num_features=64)
        self.bnc = nn.BatchNorm1d(num_features=8)
        self.bnc2 = nn.BatchNorm1d(num_features=64)
        self.Dropout = nn.Dropout(0.1)
        self.Tanh = nn.Tanh()

    def forward(self, x):
        x = x.unsqueeze(1)
        x_c = self.activation(self.bnc(self.cnn1(x)))
        x_c2 = self.activation(self.bnc2(self.cnn2(x_c)))
        outRNN, h_h = self.rnn(x_c2)
        x_flatten = outRNN.reshape(-1, 64*5)
        x1 = self.activation(self.bn1(self.linear_1(x_flatten)))
        x1_d = self.Dropout(x1)
        x2 = self.activation(self.bn2(self.linear_2(x1_d)))
        x2_d = self.Dropout(x2)
        x2_d = x2_d + x1_d
        x3 = self.activation(self.bn3(self.linear_3(x2_d)))
        x3_d = self.Dropout(x3)
        x3_d = x3_d + x2_d
        x4 = self.activation(self.bn4(self.linear_4(x3_d)))
        x4_4 = self.Dropout(x4)
        x5 = self.activation(self.bn5(self.linear_5(x4_4)))
        x6 = self.activation(self.linear_6(x5))
        x7 = self.activation(self.linear_7(x6))
        x8 = self.Tanh(self.linear_8(x7))
        return x8

# state_68_Tue_Jul_27_02_21_28.pt
class ConFNet4(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConFNet4, self).__init__()
        self.cnn1 = nn.Conv1d(1, 8, kernel_size=1, stride=1, padding=1)
        self.cnn2 = nn.Conv1d(8, 64, kernel_size=1, stride=1, padding=1)
        self.rnn = nn.LSTM(input_size=6, hidden_size=5, num_layers=4, batch_first=True)
        self.linear_1 = nn.Linear(320, 256)
        self.linear_2 = nn.Linear(256, 128)
        self.linear_3 = nn.Linear(128, 64)
        self.linear_4 = nn.Linear(64, 32)
        self.linear_5 = nn.Linear(32, 16)
        self.linear_6 = nn.Linear(16, 8)
        self.linear_7 = nn.Linear(8, 6)
        self.linear_8 = nn.Linear(6, output_dim)
        self.activation = nn.Sigmoid()
        self.bn1 = nn.BatchNorm1d(num_features=256)
        self.bn2 = nn.BatchNorm1d(num_features=128)
        self.bn3 = nn.BatchNorm1d(num_features=64)
        self.bn4 = nn.BatchNorm1d(num_features=32)
        self.bn5 = nn.BatchNorm1d(num_features=16)
        self.bnc = nn.BatchNorm1d(num_features=8)
        self.bnc2 = nn.BatchNorm1d(num_features=64)
        self.Dropout = nn.Dropout(0.1)
        self.LKR = nn.LeakyReLU(negative_slope=0.01)
        self.Tanh = nn.Tanh()

    def forward(self, x):
        x = x.unsqueeze(1)
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
        x5 = self.activation(self.bn5(self.linear_5(x4_4)))
        x6 = self.activation(self.linear_6(x5))
        x7 = self.activation(self.linear_7(x6))
        x8 = self.Tanh(self.linear_8(x7))
        return x8

# model_Wed_Jul_28_08_33_09.pt
class ConFNet5(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConFNet5, self).__init__()
        self.cnn1 = nn.Conv1d(1, 8, kernel_size=1, stride=1, padding=1)
        self.cnn2 = nn.Conv1d(8, 64, kernel_size=1, stride=1, padding=1)
        self.rnn = nn.LSTM(input_size=6, hidden_size=5, num_layers=3, batch_first=True)
        self.linear_1 = nn.Linear(320, 256)
        self.linear_2 = nn.Linear(256, 256)
        self.linear_3 = nn.Linear(256, 32)
        self.linear_4 = nn.Linear(32, 32)
        self.linear_5 = nn.Linear(32, 128)
        self.linear_6 = nn.Linear(128, 32)
        self.linear_7 = nn.Linear(32, 64)
        self.linear_8 = nn.Linear(64, output_dim)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.bn1 = nn.BatchNorm1d(num_features=256)
        self.bn2 = nn.BatchNorm1d(num_features=256)
        self.bn3 = nn.BatchNorm1d(num_features=32)
        self.bn4 = nn.BatchNorm1d(num_features=32)
        self.bn5 = nn.BatchNorm1d(num_features=128)
        self.bnc = nn.BatchNorm1d(num_features=8)
        self.bnc2 = nn.BatchNorm1d(num_features=64)
        self.Dropout = nn.Dropout(0.3)
        self.LKR = nn.LeakyReLU(negative_slope=0.01)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)
        x_c = self.LKR(self.bnc(self.cnn1(x)))
        x_c2 = self.LKR(self.bnc2(self.cnn2(x_c)))
        outRNN, h_h = self.rnn(x_c2)
        x_flatten = outRNN.reshape(-1, 64*5)
        x1 = self.activation(self.bn1(self.linear_1(x_flatten)))
        x1_d = self.Dropout(x1)
        x2 = self.activation(self.bn2(self.linear_2(x1_d)))
        x2_d = self.Dropout(x2)
        x2_d = x2_d + x1_d
        x3 = self.activation(self.bn3(self.linear_3(x2_d)))
        x3_d = self.Dropout(x3)
        # x3_d = x3_d + x2_d
        x4 = self.activation(self.bn4(self.linear_4(x3_d)))
        x4_4 = self.Dropout(x4)
        x4_4 = x4_4 + x3_d
        x5 = self.activation(self.bn5(self.linear_5(x4_4)))
        x6 = self.activation(self.linear_6(x5))
        x6 = x6 + x4_4
        x7 = self.activation(self.linear_7(x6))
        # x7 = x7 + x6
        x8 = self.Sigmoid(self.linear_8(x7))
        return x8

class ConFNet6(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConFNet6, self).__init__()
        self.cnn1 = nn.Conv1d(1, 8, kernel_size=1, stride=1, padding=1)
        self.cnn2 = nn.Conv1d(8, 64, kernel_size=1, stride=1, padding=1)
        self.rnn = nn.LSTM(input_size=6, hidden_size=5, num_layers=3, batch_first=True)
        self.linear_1 = nn.Linear(320, 64)
        self.linear_2 = nn.Linear(64, 512)
        self.linear_3 = nn.Linear(512, 16)
        self.linear_4 = nn.Linear(16, 512)
        self.linear_5 = nn.Linear(512, 128)
        self.linear_6 = nn.Linear(128, 32)
        self.linear_7 = nn.Linear(32, 64)
        self.linear_8 = nn.Linear(64, output_dim)
        self.activation = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.bn2 = nn.BatchNorm1d(num_features=512)
        self.bn3 = nn.BatchNorm1d(num_features=16)
        self.bn4 = nn.BatchNorm1d(num_features=512)
        self.bn5 = nn.BatchNorm1d(num_features=128)
        self.bnc = nn.BatchNorm1d(num_features=8)
        self.bnc2 = nn.BatchNorm1d(num_features=64)
        self.Dropout = nn.Dropout(0.1)
        self.LKR = nn.LeakyReLU(negative_slope=0.01)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)
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
        x4_4 = x4_4 + x2_d
        x5 = self.activation(self.bn5(self.linear_5(x4_4)))
        x6 = self.activation(self.linear_6(x5))
        # x6 = x6 + x4_4
        x7 = self.activation(self.linear_7(x6))
        x7 = x7 + x1_d
        x8 = self.Sigmoid(self.linear_8(x7))
        return x8

class ConFNet7(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConFNet7, self).__init__()
        self.cnn1 = nn.Conv1d(1, 8, kernel_size=1, stride=1, padding=1)
        self.cnn2 = nn.Conv1d(8, 64, kernel_size=1, stride=1, padding=1)
        self.rnn = nn.LSTM(input_size=6, hidden_size=5, num_layers=5, batch_first=True)
        self.linear_1 = nn.Linear(320, 128)
        self.linear_2 = nn.Linear(128, 64)
        self.linear_3 = nn.Linear(64, 4)
        self.linear_4 = nn.Linear(4, 32)
        self.linear_5 = nn.Linear(32, 32)
        self.linear_6 = nn.Linear(32, 4)
        self.linear_7 = nn.Linear(4, 512)
        self.linear_8 = nn.Linear(512, output_dim)
        self.activation = nn.Tanh()
        self.bn1 = nn.BatchNorm1d(num_features=128)
        self.bn2 = nn.BatchNorm1d(num_features=64)
        self.bn3 = nn.BatchNorm1d(num_features=4)
        self.bn4 = nn.BatchNorm1d(num_features=32)
        self.bn5 = nn.BatchNorm1d(num_features=32)
        self.bnc = nn.BatchNorm1d(num_features=8)
        self.bnc2 = nn.BatchNorm1d(num_features=64)
        self.Dropout = nn.Dropout(0.1)
        self.LKR = nn.LeakyReLU(negative_slope=0.01)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)
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
        # x4_4 = x4_4 + x2_d
        x5 = self.activation(self.bn5(self.linear_5(x4_4)))
        x5 = x4_4 + x5
        x6 = self.activation(self.linear_6(x5))
        x6 = x6 + x3_d
        x7 = self.activation(self.linear_7(x6))
        # x7 = x7 + x1_d
        x8 = self.Sigmoid(self.linear_8(x7))
        return x8

class ConFNet8(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConFNet8, self).__init__()
        self.cnn1 = nn.Conv1d(1, 8, kernel_size=1, stride=1, padding=1)
        self.cnn2 = nn.Conv1d(8, 64, kernel_size=1, stride=1, padding=1)
        self.rnn = nn.LSTM(input_size=6, hidden_size=5, num_layers=4, batch_first=True)
        self.linear_1 = nn.Linear(320, 256)
        self.linear_2 = nn.Linear(256, 16)
        self.linear_3 = nn.Linear(16, 32)
        self.linear_4 = nn.Linear(32, 256)
        self.linear_5 = nn.Linear(256, 128)
        self.linear_6 = nn.Linear(128, 256)
        self.linear_7 = nn.Linear(256, 32)
        self.linear_8 = nn.Linear(32, output_dim)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.bn1 = nn.BatchNorm1d(num_features=256)
        self.bn2 = nn.BatchNorm1d(num_features=16)
        self.bn3 = nn.BatchNorm1d(num_features=32)
        self.bn4 = nn.BatchNorm1d(num_features=256)
        self.bn5 = nn.BatchNorm1d(num_features=128)
        self.bnc = nn.BatchNorm1d(num_features=8)
        self.bnc2 = nn.BatchNorm1d(num_features=64)
        self.Dropout = nn.Dropout(0.2)
        self.LKR = nn.LeakyReLU(negative_slope=0.01)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)
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
        x4_4 = x4_4 + x1_d
        x5 = self.activation(self.bn5(self.linear_5(x4_4)))
        # x5 = x3_d + x5
        x6 = self.activation(self.linear_6(x5))
        x6 = x6 + x4_4
        x7 = self.activation(self.linear_7(x6))
        x7 = x7 + x3_d
        x8 = self.Sigmoid(self.linear_8(x7))
        return x8
# 'state_39_Sun_Aug__1_13_54_05.pt'
# 'model_Sun_Aug__1_13_54_05.pt'
class ConFNet9(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConFNet9, self).__init__()
        self.cnn1 = nn.Conv1d(1, 8, kernel_size=1, stride=1, padding=1)
        self.cnn2 = nn.Conv1d(8, 64, kernel_size=1, stride=1, padding=1)
        self.rnn = nn.LSTM(input_size=6, hidden_size=5, num_layers=1, batch_first=True)
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
        x = x.unsqueeze(1)
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

class ConFNet10(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConFNet10, self).__init__()
        self.cnn1 = nn.Conv1d(1, 8, kernel_size=1, stride=1, padding=1)
        self.cnn2 = nn.Conv1d(8, 64, kernel_size=1, stride=1, padding=1)
        self.rnn = nn.LSTM(input_size=6, hidden_size=5, num_layers=1, batch_first=True)
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
        x = x.unsqueeze(1)
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

class ConFNet11(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConFNet11, self).__init__()
        self.cnn1 = nn.Conv1d(1, 8, kernel_size=1, stride=1, padding=1)
        self.cnn2 = nn.Conv1d(8, 64, kernel_size=1, stride=1, padding=1)
        self.rnn = nn.LSTM(input_size=6, hidden_size=5, num_layers=1, batch_first=True)
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
        x = x.unsqueeze(1)
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

class ConFNet_s1(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConFNet_s1, self).__init__()
        self.cnn1 = nn.Conv1d(1, 8, kernel_size=2, stride=1, padding=2)
        self.cnn2 = nn.Conv1d(8, 32, kernel_size=2, stride=1, padding=2)
        self.cnn3 = nn.Conv1d(32, 64, kernel_size=2, stride=1, padding=2)
        self.cnn4 = nn.Conv1d(64, 128, kernel_size=2, stride=1, padding=2)
        self.rnn = nn.LSTM(input_size=6, hidden_size=5, num_layers=1, batch_first=True)
        self.linear_0 = nn.Linear(1792, 320)
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
        self.bnc2 = nn.BatchNorm1d(num_features=32)
        self.bnc3 = nn.BatchNorm1d(num_features=64)
        self.bnc4 = nn.BatchNorm1d(num_features=128)
        self.Dropout = nn.Dropout(0.1)
        self.LKR = nn.LeakyReLU(negative_slope=0.01)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)
        x_c = self.Dropout(self.LKR(self.bnc(self.cnn1(x))))
        x_c2 = self.Dropout(self.LKR(self.bnc2(self.cnn2(x_c))))
        x_c3 = self.Dropout(self.LKR(self.bnc3(self.cnn3(x_c2))))
        x_c4 = self.Dropout(self.LKR(self.bnc4(self.cnn4(x_c3))))
        x_flatten = x_c4.view(-1, 128*14)
        x_0 = self.linear_0(x_flatten)
        x1 = self.activation(self.bn1(self.linear_1(x_0)))
        x2 = self.activation(self.bn2(self.linear_2(x1)))
        x3 = self.activation(self.bn3(self.linear_3(x2)))
        x4 = self.activation(self.bn4(self.linear_4(x3)))
        x5 = self.activation(self.bn5(self.linear_5(x4)))
        x5 = x2 + x5
        x6 = self.activation(self.linear_6(x5))
        x7 = self.activation(self.linear_7(x6))
        x8 = self.Sigmoid(self.linear_8(x7))
        return x8

dispatch = {'state_42_Sun_Jul_18_21_51_26_ConFNetRNN.pt': (ConFNet1, [-120,120,-25,25,-50,50,-1.4,1.4]),
            'state_99_Mon_Jul_26_00_56_28.pt': (ConFNet2, [-120,120,-25,25,-50,50,-1.4,1.4]),
            'state_87_Mon_Jul_26_15_54_36.pt': (ConFNet3, [-120,120,-25,25,-50,50,-1.4,1.4]),
            'state_68_Tue_Jul_27_02_21_28.pt': (ConFNet4, [-120,120,-25,25,-50,50,-1.4,1.4]),
            'model_Wed_Jul_28_08_33_09.pt': (ConFNet5, [-120,120,-25,25,-50,50,-1.4,1.4]),
            'state_19_Wed_Jul_28_14_25_53.pt': (ConFNet6, [-120,120,-25,25,-50,50,-1.4,1.4]),
            'model_Wed_Jul_28_14_25_53.pt': (ConFNet6, [-120,120,-25,25,-50,50,-1.4,1.4]),
            'state_19_Wed_Jul_28_15_39_58.pt': (ConFNet7, [-120,120,-25,25,-50,50,-1.4,1.4]),
            'model_Wed_Jul_28_16_30_10.pt': (ConFNet8, [-120,120,-24.6090,24.6524,-48.9289,49.9884,-1.3976,1.3989]),
            'state_30_Wed_Jul_28_16_30_10.pt': (ConFNet8, [-120,120,-24.6090,24.6524,-48.9289,49.9884,-1.3976,1.3989]),
            'state_39_Sun_Aug__1_13_54_05.pt': (ConFNet9, [-120,120,-21.9717,21.9717,-44.9300,44.9875,-1.3000,1.2928]),
            'model_Sun_Aug__1_13_54_05.pt': (ConFNet9, [-120,120,-21.9717,21.9717,-44.9300,44.9875,-1.3000,1.2928]),
            'model_Mon_Aug__2_09_31_08.pt': (ConFNet10, [-120,120,-21.9717,21.9717,-44.9300,44.9875,-1.3000,1.2928]),
            'model_Mon_Aug__2_14_08_47.pt': (ConFNet11, [-120,120,-21.9717,21.9717,-44.9300,44.9875,-1.3000,1.2928]),
            'model_Tue_Sep__26_3_21_12.pt': (ConFNet_s1, [-120,120,-21.9717,21.9717,-44.9300,44.9875,-1.3000,1.2928]),
            'model_Wed_Nov_24_15_25_05.pt': (FNetSkip, [-120,120,-21.9717,21.9717,-44.9300,44.9875,-1.3000,1.2928]),
            'model_Thu_Nov_25_11_29_30.pt': (FNet, [-120,120,-24.9079,24.6524,-49.7925,49.9884,-1.3976,1.3989])}

def get_fd_model(name):
    return dispatch[name]