import torch
import torch.nn as nn

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

dispatch = {'model_Tue_Sep__26_3_21_12.pt': (ConFNet_s1, [-120,120,-21.9717,21.9717,-44.9300,44.9875,-1.3000,1.2928])}

def get_fd_model(name):
    return dispatch[name]
