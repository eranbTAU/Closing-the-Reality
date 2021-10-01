
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
import os.path
import torch.utils.data as data_utils
import time
import re
import seaborn as sns
import copy
from torch.autograd import Variable
import re
import socket
s = socket.socket()

PATH = r"E:\CarProject\NewCode_Project\gan\results"
spath = r"E:\CarProject\NewCode_Project\gan\gans"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
t = time.asctime()


class GeneratorR(nn.Module):
    def __init__(self):
        super(GeneratorR, self).__init__()

        self.fc1 = nn.Linear(6, 32)
        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=(2/6)**0.5)
        self.fc2 = nn.Linear(32, 16)
        torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=(2/32)**0.5)
        self.fc3 = nn.Linear(16, 8)
        torch.nn.init.normal_(self.fc3.weight, mean=0.0, std=(2/16)**0.5)
        self.fc4 = nn.Linear(8, 5)
        torch.nn.init.xavier_uniform(self.fc4.weight)

    def forward(self, x, training=False):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x

class DiscriminatorR(nn.Module):
    def __init__(self):
        super(DiscriminatorR, self).__init__()
        self.fc1 = nn.Linear(5, 32)
        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=(2/5)**0.5)
        self.fc2 = nn.Linear(32, 32)
        torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=(2/32)**0.5)
        self.fc3 = nn.Linear(64, 32)
        torch.nn.init.normal_(self.fc3.weight, mean=0.0, std=(2/32)**0.5)
        self.fc4 = nn.Linear(32, 32)
        torch.nn.init.normal_(self.fc4.weight, mean=0.0, std=(2/32)**0.5)
        self.fc5 = nn.Linear(64, 32)
        torch.nn.init.normal_(self.fc5.weight, mean=0.0, std=(2/32)**0.5)
        self.fc6 = nn.Linear(32, 1)
        torch.nn.init.xavier_uniform(self.fc6.weight)

    def forward(self, x, training=False):
        output = F.leaky_relu(self.fc1(x))
        output2 = F.leaky_relu(self.fc2(output))
        output3 = torch.cat((output, output2), 1)
        output4 = F.leaky_relu(self.fc3(output3))
        output5 = F.leaky_relu(self.fc4(output4))
        output6 = torch.cat((output4, output5), 1)
        output7 = F.leaky_relu(self.fc5(output6))
        x = torch.sigmoid(self.fc6(output7))
        return x

class GanModelR(nn.Module):
    def __init__(self):
        super(GanModelR, self).__init__()
        self.generator = GeneratorR()
        self.discriminator = DiscriminatorR()

class ConFNet(torch.nn.Module):
    def __init__(self):
        super(ConFNet, self).__init__()
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
        self.linear_8 = nn.Linear(512, 3)
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

    def forward(self, x): # b, c, ta, tb
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


    def fit(self, trainloader, validloader, epochs, criterion, optimizer, scheduler, name_option):
        try:
            running_loss = 0.0
            loss_train=[]
            loss_valid=[]
            best_loss= 100
            best_model_wts = copy.deepcopy(c_net.state_dict())
            for epoch in range(epochs):
                for i, (data, labels) in enumerate(trainloader):
                    data = data.to(device).type(torch.FloatTensor)
                    labels = labels.to(device)
                    # forward
                    data=torch.unsqueeze(data, 1)
                    output = c_net(data)
                    #loss = criterion(output, labels)
                    loss = torch.sum((labels - output) ** 2) / len(labels)
                    # backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    running_loss += float(loss)
                    #running_loss += loss.item()  # return loss of one batch as float
                    if i % 100 == 99:
                        total_loss= running_loss / (i + 1)
                        print(f"[{epoch},{i}]loss: {round(total_loss,7)}")
                print(f"Training loss {epoch}: {round(running_loss / len(trainloader),7)}")
                loss_train.append(running_loss/len(trainloader))
                valid_loss_now= self.evaluate(validloader,criterion)[1]
                loss_valid.append(valid_loss_now)
                if valid_loss_now<=best_loss:
                    best_loss=valid_loss_now
                    best_model_wts = copy.deepcopy(c_net.state_dict())
                running_loss = 0.0
                scheduler.step()
        except:
            save_network(epoch, self, optimizer, name_option)
        else:
            print("Training is complete")
            save_network(epoch, self, optimizer, name_option)
        return loss_train, loss_valid, best_model_wts

    def evaluate(self, loader,criterion):
        pre = torch.tensor([])
        y = torch.tensor([])
        self.eval()
        with torch.no_grad():
            running_loss = 0
            for i, (data, labels) in enumerate(loader):
                data = data.to(device).type(torch.FloatTensor)
                data = torch.unsqueeze(data, 1)
                labels = labels.to(device)
                # forward
                output = self(data)
                #loss = criterion(output, labels)
                loss = torch.sum((labels - output) ** 2) / len(labels)
                #running_loss += loss.item()
                running_loss += float(loss)
                pre = torch.cat((pre, output), 0)
                y = torch.cat((y, labels), 0)
        self.train()
        return pre, running_loss/ len(loader), y

    def loss_graph(self, train_loss, valid_loss, name_option):
        plt.plot(range(len(train_loss)),train_loss, label="train_loss")
        plt.plot(range(len(valid_loss)), valid_loss, label="test_loss")
        plt.legend()
        plt.savefig("loss train&validation "+str(name_option)+".png")
        plt.show()


class Net_fc(nn.Module):

    def __init__(self):
        super(Net_fc, self).__init__()
        self.fc1 = nn.Linear(2*1, 128)
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        self.bn1 = nn.BatchNorm1d(num_features=128)
        self.fc2 = nn.Linear(128, 64)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        self.bn2 = nn.BatchNorm1d(num_features=64)
        self.fc3 = nn.Linear(64, 32)
        torch.nn.init.kaiming_normal_(self.fc3.weight)
        self.bn3 = nn.BatchNorm1d(num_features=32)
        self.fc4 = nn.Linear(32, 16)
        torch.nn.init.kaiming_normal_(self.fc4.weight)
        self.bn4 = nn.BatchNorm1d(num_features=16)
        self.fc5 = nn.Linear(16, 8)
        torch.nn.init.kaiming_normal_(self.fc5.weight)
        self.bn5 = nn.BatchNorm1d(num_features=8)
        self.fc6 = nn.Linear(8, 3)
        torch.nn.init.kaiming_normal_(self.fc6.weight)
        #self.bn6 = nn.BatchNorm1d(num_features=8)


    def forward(self, x, training=False):
        p=0.2
        x = torch.tanh(self.bn1(self.fc1(x)))
        x = F.dropout(x, p, training)
        x = torch.tanh(self.bn2(self.fc2(x)))
        x = F.dropout(x, p, training)
        x = torch.tanh(self.bn3(self.fc3(x)))
        x = F.dropout(x, p, training)
        x = torch.tanh(self.bn4(self.fc4(x)))
        #x = F.dropout(x, p, training)
        x = torch.tanh(self.bn5(self.fc5(x)))
        #x = F.dropout(x, p, training)
        x = torch.tanh(self.fc6(x))
        return x

    def fit(self, trainloader, validloader, epochs, criterion, optimizer, scheduler, name_option):
        try:
            running_loss = 0.0
            loss_train=[]
            loss_valid=[]
            best_loss= 100
            best_model_wts = copy.deepcopy(net.state_dict())
            for epoch in range(epochs):
                for i, (data, labels) in enumerate(trainloader):
                    data = data.to(device).type(torch.FloatTensor)
                    labels = labels.to(device)
                    # forward
                    output = net(data, training=True)
                    #loss = criterion(output, labels)
                    loss = torch.sum((labels - output) ** 2) / len(labels)
                    # backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    #running_loss += loss.item()  # return loss of one batch as float
                    running_loss += float(loss)
                    if i % 100 == 99:  # print every 20 mini-batches
                        total_loss= running_loss / (i + 1)
                        print((f"[{epoch},{i}]loss: {round(total_loss,7)}"))
                print(f"Training loss {epoch}: {round(running_loss / len(trainloader),7)}")
                loss_train.append(running_loss/len(trainloader))
                valid_loss_now= self.evaluate(validloader,criterion)[1]
                loss_valid.append(valid_loss_now)
                if valid_loss_now<=best_loss:
                    best_loss=valid_loss_now
                    best_model_wts = copy.deepcopy(net.state_dict())
                running_loss = 0.0
                scheduler.step()
        except:
            save_network(epoch, self, optimizer, name_option)
        else:
            print("Training is complete")
            save_network(epoch, self, optimizer, name_option)
        return loss_train, loss_valid, best_model_wts

    def evaluate(self, loader,criterion):
        pre = torch.tensor([]).type(torch.cuda.FloatTensor)
        y = torch.tensor([]).type(torch.cuda.FloatTensor)
        self.eval()
        with torch.no_grad():
            running_loss = 0
            for i, (data, labels) in enumerate(loader):
                data = data.to(device).type(torch.cuda.FloatTensor)
                labels = labels.to(device)
                # forward
                output = self(data, training=False)
                #loss = criterion(output, labels)
                loss = torch.sum((labels - output) ** 2) / len(labels)
                #running_loss += loss.item()
                running_loss += float(loss)
                pre = torch.cat((pre, output), 0)
                y = torch.cat((y, labels), 0)
        self.train()
        return pre, running_loss/ len(loader), y

    def loss_graph(self, train_loss, valid_loss, name_option):
        plt.plot(range(len(train_loss)),train_loss, label="train_loss")
        plt.plot(range(len(valid_loss)), valid_loss, label="test_loss")
        plt.legend()
        plt.savefig("loss train&validation "+str(name_option)+".png")
        plt.show()

class ConFNet2(torch.nn.Module):
    def __init__(self):
        super(ConFNet2, self).__init__()
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
        self.linear_8 = nn.Linear(4, 3)
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
        x8 = F.tanh(self.linear_8(x7))
        return x8

    def fit(self, trainloader, validloader, epochs, optimizer, name_option):
        try:
            running_loss = 0.0
            loss_train=[]
            loss_valid=[]
            best_loss= 100
            best_model_wts = copy.deepcopy(c_net2.state_dict())
            for epoch in range(epochs):
                for i, (data, labels) in enumerate(trainloader):
                    data = data.to(device).type(torch.FloatTensor)
                    labels = labels.to(device)
                    # forward
                    data = torch.unsqueeze(data, 1)
                    output = self(data)
                    #loss = criterion(output, labels)
                    #loss = torch.sum((labels - output) ** 2) / len(labels)
                    loss = self.weighted_mse_loss(output, labels)
                    loss += self.huber_loss(labels, output)
                    loss += self.logcosh(labels, output)
                    loss += torch.sum((output- labels) ** 2) / len(output)
                    # backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    #running_loss += loss.item()  # return loss of one batch as float
                    running_loss += float(loss)
                    if i % 100 == 99:  # print every 20 mini-batches
                        total_loss= running_loss / (i + 1)
                        print((f"[{epoch},{i}]loss: {round(total_loss,7)}"))
                print(f"Training loss {epoch}: {round(running_loss / len(trainloader),7)}")
                loss_train.append(running_loss/len(trainloader))
                valid_loss_now = self.evaluate(validloader)[1]
                loss_valid.append(valid_loss_now)
                if valid_loss_now <= best_loss:
                    best_loss = valid_loss_now
                    best_model_wts = copy.deepcopy(c_net2.state_dict())
                running_loss = 0.0
        except:
            save_network(epoch, self, optimizer, name_option)
        else:
            print("Training is complete")
            save_network(epoch, self, optimizer, name_option)
        return loss_train, loss_valid, best_model_wts

    def evaluate(self, loader):
        pre = torch.tensor([]).type(torch.cuda.FloatTensor)
        y = torch.tensor([]).type(torch.cuda.FloatTensor)
        self.eval()
        with torch.no_grad():
            running_loss = 0
            for i, (data, labels) in enumerate(loader):
                data = data.to(device).type(torch.cuda.FloatTensor)
                data = torch.unsqueeze(data, 1)
                labels = labels.to(device)
                # forward
                output = self(data)
                loss = self.weighted_mse_loss(output, labels)
                loss += self.huber_loss(labels, output)
                loss += self.logcosh(labels, output)
                loss += torch.sum((output - labels) ** 2) / len(output)
                #loss = criterion(output, labels)
                #loss = torch.sum((labels - output) ** 2) / len(labels)
                #running_loss += loss.item()
                running_loss += float(loss)
                pre = torch.cat((pre, output), 0)
                y = torch.cat((y, labels), 0)
        self.train()
        z = running_loss / len(loader)
        return pre, z, y

    def weighted_mse_loss(self, input, target):
        # alpha of 0.5 means half weight goes to first, remaining half split by remaining 15
        weights = Variable(torch.Tensor([1, 2, 2.5]))
        weights = weights.type(torch.cuda.FloatTensor)# 1, 1.5, 2
        pct_var = (input - target) ** 2
        out = pct_var * weights.expand_as(target)
        loss = out.mean()
        return loss

    def huber_loss(self, y, y_pred, sigma=0.1):
        r = (y - y_pred).abs()
        loss1 = (r[r <= sigma]).pow(2).mean()
        loss2 = (r[r > sigma]).mean() * sigma - sigma ** 2 / 2
        if np.isnan(float(loss1)) == True:
            return loss2
        if np.isnan(float(loss2)) == True:
            return loss1
        return loss1+loss2

    # log cosh loss
    def logcosh(self,true, pred):
        loss = torch.log(torch.cosh(pred - true))
        return torch.sum(loss) / len(loss)

    def loss_graph(self, train_loss, valid_loss, name_option):
        plt.plot(range(len(train_loss)),train_loss, label="train_loss")
        plt.plot(range(len(valid_loss)), valid_loss, label="test_loss")
        plt.legend()
        plt.savefig("loss train&validation "+str(name_option)+".png")
        plt.show()

def loss_val(loss_list ,start, till, step, name_option):
    plt.plot(range(start,till,step),loss_list, label="generated data size")
    #plt.set_ylabel('valid loss')
    plt.savefig("validation loss "+str(name_option)+".png")
    plt.show()

def plot_all(prediction, actual, title, name_option):  # input: array[[dx,dy.d_theta],...] shape:(num_of_samples,3)
    pred1, pred2, pred3 = prediction[:, 0], prediction[:, 1], prediction[:, 2]
    actual1, actual2, actual3 = actual[:, 0], actual[:, 1], actual[:, 2]
    x = np.arange(1, len(pred1) + 1)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 10))
    fig.suptitle(title)
    ax1.plot(x, pred1, label='predicted')
    ax1.plot(x, actual1, label='actual')
    ax1.set_ylabel('dx')
    ax2.plot(x, pred2, label='predicted')
    ax2.plot(x, actual2, label='actual')
    ax2.set_ylabel('dy')
    ax3.plot(x, pred3, label='predicted')
    ax3.plot(x, actual3, label='actual')
    ax3.set_ylabel('d_theta')
    ax3.set_xlabel('time step')
    ax1.legend()
    plt.savefig(str(name_option) + ".png")
    plt.show()


def save_network(epoch, net, optimizer,model_name):
    state = {
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, os.path.join(PATH,str(model_name)+".pth"))

def plot_pre_vs_gan(prediction, gan, title, model_name):  # input: array[[dx,dy.d_theta],...] shape:(num_of_samples,3)
    pred1, pred2, pred3 = prediction[:, 0], prediction[:, 1], prediction[:, 2]
    gan1, gan2, gan3 = gan[:, 0], gan[:, 1], gan[:, 2]
    x = np.arange(1, len(pred1) + 1)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 10))
    fig.suptitle(title)
    ax1.plot(x, pred1, label='predicted')
    ax1.plot(x, gan1, label='actual')
    ax1.set_ylabel('dx')
    ax2.plot(x, pred2, label='predicted')
    ax2.plot(x, gan2, label='actual')
    ax2.set_ylabel('dy')
    ax3.plot(x, pred3, label='predicted')
    ax3.plot(x, gan3, label='actual')
    ax3.set_ylabel('d_theta')
    ax3.set_xlabel('time step')
    ax1.legend()
    plt.savefig(str(model_name) + ".png")
    plt.show()


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(6, 32)
        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=(2/6)**0.5)
        self.fc2 = nn.Linear(32, 16)
        torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=(2/32)**0.5)
        self.fc3 = nn.Linear(16, 8)
        torch.nn.init.normal_(self.fc3.weight, mean=0.0, std=(2/16)**0.5)
        self.fc4 = nn.Linear(8, 5)
        torch.nn.init.xavier_normal(self.fc4.weight)

        self.bn1 = nn.BatchNorm1d(num_features=32)
        self.bn2 = nn.BatchNorm1d(num_features=16)
        self.bn3 = nn.BatchNorm1d(num_features=8)
        #self.bn4 = nn.BatchNorm1d(num_features=5)

    def forward(self, x, training=False):
        p = 0.2

        x = F.leaky_relu(self.bn1(self.fc1(x)))
        #x = F.dropout(x, p, training)
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        #x = F.dropout(x, p, training)
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = torch.tanh(self.fc4(x))
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.fc1 = nn.Linear(5, 32)
        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=(2/5)**0.5)
        self.fc2 = nn.Linear(32, 16)
        torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=(2/32)**0.5)
        self.fc3 = nn.Linear(16, 8)
        torch.nn.init.normal_(self.fc3.weight, mean=0.0, std=(2/16)**0.5)
        self.fc4 = nn.Linear(8, 6)
        torch.nn.init.normal_(self.fc4.weight, mean=0.0, std=(2/8)**0.5)
        self.fc5 = nn.Linear(6, 1)
        torch.nn.init.xavier_normal(self.fc5.weight)

    def forward(self, x, training=False):
        p = 0.2

        x = F.leaky_relu(self.fc1(x))
        #x = F.dropout(x, p, training)
        x = F.leaky_relu(self.fc2(x))
        #x = F.dropout(x, p, training)
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x

class GanModel(nn.Module):
    def __init__(self):
        super(GanModel, self).__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()

class GeneratorB(nn.Module):
    def __init__(self):
        super(GeneratorB, self).__init__()

        self.fc1 = nn.Linear(6, 32)
        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=(2/6)**0.5)
        self.fc2 = nn.Linear(32, 16)
        torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=(2/32)**0.5)
        self.fc3 = nn.Linear(16, 8)
        torch.nn.init.normal_(self.fc3.weight, mean=0.0, std=(2/16)**0.5)
        self.fc4 = nn.Linear(8, 5)
        torch.nn.init.xavier_uniform(self.fc4.weight)

    def forward(self, x, training=False):
        p = 0.2
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x

class DiscriminatorB(nn.Module):
    def __init__(self):
        super(DiscriminatorB, self).__init__()
        self.fc1 = nn.Linear(5, 32)
        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=(2/5)**0.5)
        self.fc2 = nn.Linear(32, 16)
        torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=(2/32)**0.5)
        self.fc3 = nn.Linear(16, 8)
        torch.nn.init.normal_(self.fc3.weight, mean=0.0, std=(2/16)**0.5)
        self.fc4 = nn.Linear(8, 6)
        torch.nn.init.normal_(self.fc4.weight, mean=0.0, std=(2/8)**0.5)
        self.fc5 = nn.Linear(6, 1)
        torch.nn.init.xavier_uniform(self.fc5.weight)

    def forward(self, x, training=False):
        p = 0.2
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x

class GanModelB(nn.Module):
    def __init__(self):
        super(GanModelB, self).__init__()
        self.generator = GeneratorB()
        self.discriminator = DiscriminatorB()

def process_data(data, val, test, batch_size):
    dataset = pd.DataFrame({'engine1': data[0][:,0], 'engine2': data[0][:,1]})
    labels = pd.DataFrame({'label1': data[1][:, 0], 'label2': data[1][:, 1], 'label3': data[1][:, 2]})
    print(dataset.isna().sum())
    print(labels.isna().sum())

    # Scale features
    max_engine = np.amax(data[0], axis=0)
    min_engine = np.amin(data[0], axis=0)
    max_dx_dy_dtheta = np.amax(data[1], axis=0)
    min_dx_dy_dtheta = np.amin(data[1], axis=0)
    scaler1 = MinMaxScaler()
    scaler1.fit(data[0])
    #scaler1.data_max_ = scaler1.data_max_+0.1#לא לשכוח לעדכן גם את הנירמול של הגאן
    #scaler1.data_min_ = scaler1.data_min_ - 0.1
    data[0] = scaler1.transform(data[0])
    data[0] = data[0]-0.5
    val[0] = scaler1.transform(val[0])
    val[0] = val[0]-0.5
    test[0] = scaler1.transform(test[0])
    test[0] = test[0]-0.5


    scaler2 = MinMaxScaler()
    scaler2.fit(data[1])
    #scaler2.data_max_ = scaler2.data_max_+0.1#לא לשכוח לעדכן גם את הנירמול של הגאן
    #scaler2.data_min_ = scaler2.data_min_ - 0.1
    data[1] = scaler2.transform(data[1])
    data[1] = data[1]-0.5
    val[1] = scaler2.transform(val[1])
    val[1] = val[1]-0.5
    test[1] = scaler2.transform(test[1])
    test[1] = test[1]-0.5

    x_train_tensor = torch.tensor(data[0]).type(torch.FloatTensor)
    y_train_tensor = torch.tensor(data[1]).type(torch.FloatTensor)

    x_valid_tensor = torch.tensor(val[0]).type(torch.FloatTensor)
    y_valid_tensor = torch.tensor(val[1]).type(torch.FloatTensor)

    x_test_tensor = torch.tensor(test[0]).type(torch.FloatTensor)
    y_test_tensor = torch.tensor(test[1]).type(torch.FloatTensor)

    train = data_utils.TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = data_utils.DataLoader(train, batch_size = batch_size, num_workers=0,  shuffle = False)
    valid = data_utils.TensorDataset(x_valid_tensor, y_valid_tensor)
    valid_loader = data_utils.DataLoader(valid, batch_size = batch_size, num_workers=0, shuffle = False)
    test = data_utils.TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = data_utils.DataLoader(test, batch_size = batch_size, num_workers=0, shuffle = False)
    #test_loader = test
    return train_loader, valid_loader, test_loader, scaler1, scaler2

def Normalization(data):
    scalers= [-120, 120,-120, 120,-21.9717,21.9717 ,-44.9300, 44.9875, -1.3000,  1.2928]
    for i in range(0):
        data[:,i]=(data[:,i] - scalers[i]) / (scalers[i+1] - scalers[i])
    return data

def gans_from_file(spath,size_train_data):
    gans_list = []
    for root, dirs, files in os.walk(spath, topdown=True, followlinks=False):
        for file in files:
            if file.startswith(size_train_data):
                if file.endswith('.pth'):
                    PATH_gan = os.path.join(root, file)
                    gans_list.append(PATH_gan)
    return gans_list

if __name__=='__main__':
    # hyper-parameters: gan
    num_gan_data = 55068
    start=14000
    till = 55068
    step = 2000
    batch_size=64
    t = time.asctime()
    file_name = str(re.sub(r":" ,"_" ,t))
    file_name = file_name+ "gan_all_data28.7_MODULE5_40epoch"
    opt_name = file_name

    #PATH_gan = r"C:\Users\נטע\Documents\נטע\פרימרוז\שיעורי בית\project\data 28.7\5\28.7_dis1_b256_250epoch_basic Thu Jul 29 09_32_29 2021.pth"
    #gan_model = torch.load(PATH_gan)
    #gan_model.eval()

    #true data for test
    t= open(r"E:\CarProject\NewCode_Project\EXTRC\LARGE_DATA_ANTON\train.pkl", 'rb')
    true_data=pickle.load(t)
    v= open(r"E:\CarProject\NewCode_Project\EXTRC\LARGE_DATA_ANTON\val.pkl", 'rb')
    val=pickle.load(v)
    te= open(r"E:\CarProject\NewCode_Project\EXTRC\LARGE_DATA_ANTON\DataExtract_Thu_Jul_29_14_25_16.pkl", 'rb')
    test=pickle.load(te)

    train_loader, valid_loader, test_loader ,scaler1, scaler2= process_data(true_data, val, test, batch_size)




    EREN_loss_l=[]
    EREN_best_loss_l=[]
    NETA_loss_l=[]
    NETA_best_loss_l=[]
    for i in range(start, num_gan_data, 2000):
        EREN_loss_list = []
        EREN_best_loss_list = []
        NETA_loss_list = []
        NETA_best_loss_list = []
        gans_list = gans_from_file(spath, str(i)+"_")
        for j in range(10):
            #noise = torch.from_numpy(np.random.normal(0, 0.2, [num_gan_data, 6])).type(torch.FloatTensor)
            noise = torch.from_numpy(np.random.uniform(-0.5, 0.5, [num_gan_data, 6])).type(torch.FloatTensor)
            # PATH_gan = r"C:\Users\נטע\Documents\נטע\פרימרוז\שיעורי בית\project\data 28.7\5\28.7_dis1_b256_250epoch_basic Thu Jul 29 09_32_29 2021.pth"
            gan_model = torch.load(gans_list[j])
            gan_model.eval()
            generated_data = gan_model.generator(noise, training=False)
            generated_data = generated_data.data.type(torch.FloatTensor)
            #data for training
            generated_data_dataset = data_utils.TensorDataset(generated_data[:,0:2].type(torch.cuda.FloatTensor), generated_data[:,2:].type(torch.cuda.FloatTensor))
            generated_data_loader = data_utils.DataLoader(generated_data_dataset, batch_size = batch_size, num_workers=0,  shuffle = False)

            c_net2 = ConFNet2()
            c_net2 = c_net2.to(device)
            optimizer = optim.AdamW(c_net2.parameters(), lr=0.037244507313868365,
                                    betas=(0.935695835862957, 0.9813179715548666), weight_decay=0.006156535534942842)
            e = 40
            loss_train, loss_valid, best_model_wts = c_net2.fit(generated_data_loader, valid_loader, e, optimizer,"ERAN_" + opt_name+str(i)+str(j))
            #c_net2.loss_graph(loss_train, loss_valid, "ERAN_NEW" + "generatedData_vs_validation " + str(opt_name)+str(i)+str(j))
            prediction_test, l, y_test = c_net2.evaluate(valid_loader)
            loss = c_net2.weighted_mse_loss(prediction_test, y_test)
            EREN_loss_list.append(float(loss))
            #plot_all(prediction_test[0:50, :], y_test[0:50, :], "predictions vs labels",
            #         "ERAN_NEW" + opt_name)  # input: array[[dx,dy.d_theta],...] shape:(num_of_samples,3)
            c_net_best = c_net2.to(device)
            c_net_best.load_state_dict(best_model_wts)
            best_prediction_test, l, y_test = c_net_best.evaluate(valid_loader)
            loss = c_net_best.weighted_mse_loss(best_prediction_test, y_test)
            EREN_best_loss_list.append(float(loss))
            #plot_all(best_prediction_test[0:50, :], y_test[0:50, :], "best module: predictions vs labels", "ERAN_NEW" + opt_name)
            #torch.save(c_net_best, os.path.join(PATH, "ERAN_NEW_BEST"+str(opt_name) + ".pth"))
            c_net=ConFNet()

            net=Net_fc()
            net = net.to(device)
            criterion = nn.MSELoss()
            optimizer = optim.SGD(net.parameters(), lr=0.015, momentum=0.9)
            #optimizer = optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999))
            e=120
            steps = e/3
            scheduler = optim.lr_scheduler.StepLR(optimizer, steps, gamma=0.9, verbose=True)
            loss_train, loss_valid, best_model_wts = net.fit(generated_data_loader, valid_loader, e, criterion, optimizer, scheduler,"NETA"+ opt_name+str(i)+str(j))
            #net.loss_graph(loss_train, loss_valid, "NETA_generatedData_vs_validation "+str(opt_name)+str(i)+str(j))
            prediction_test,l , y_test= net.evaluate(valid_loader,criterion)
            NETA_loss_list.append(l)
            #plot_all(prediction_test[0:50,:], y_test[0:50,:], "predictions vs labels", "NETA_"+opt_name) # input: array[[dx,dy.d_theta],...] shape:(num_of_samples,3)
            net_best = Net_fc()
            net_best = net_best.to(device)
            net_best.load_state_dict(best_model_wts)
            best_prediction_test, l, y_test = net_best.evaluate(valid_loader, criterion)
            NETA_best_loss_list.append(l)
            #plot_all(best_prediction_test[0:50, :], y_test[0:50, :], "best module: predictions vs labels", "NETA" + opt_name)
            #torch.save(net_best, os.path.join(PATH, "NETA_BEST"+str(opt_name) + ".pth"))
        EREN_loss_l.append(sum(EREN_loss_list) / len(EREN_loss_list))
        EREN_best_loss_l.append(sum(EREN_best_loss_list) / len(EREN_best_loss_list))
        NETA_loss_l.append(sum(NETA_loss_list) / len(NETA_loss_list))
        NETA_best_loss_l.append(sum(NETA_best_loss_list) / len(NETA_best_loss_list))
        pd.DataFrame(EREN_loss_list).to_pickle(PATH + "./"+"EREN_loss_" + str(i)  + ".pkl")
        pd.DataFrame(EREN_best_loss_list).to_pickle(PATH + "./"+"EREN_best_loss_" + str(i)  + ".pkl")
        pd.DataFrame(NETA_loss_list).to_pickle(PATH + "./"+"NETA_loss_" + str(i)  + ".pkl")
        pd.DataFrame(NETA_best_loss_list).to_pickle(PATH + "./"+"NETA_best_loss_" + str(i)  + ".pkl")

    EREN_loss_df = pd.DataFrame(EREN_loss_l)
    EREN_best_loss_df = pd.DataFrame(EREN_best_loss_l)
    NETA_loss_df = pd.DataFrame(NETA_loss_l)
    NETA_best_loss_df = pd.DataFrame(NETA_best_loss_l)

    EREN_loss_df.to_pickle(PATH + "./"+"EREN_loss_" + opt_name  + ".pkl")
    EREN_best_loss_df.to_pickle(PATH + "./" + "EREN_best_loss_" + opt_name + ".pkl")
    NETA_loss_df.to_pickle(PATH + "./" + "NETA_loss_" + opt_name + ".pkl")
    NETA_best_loss_df.to_pickle(PATH + "./" + "NETA_best_loss_" + opt_name + ".pkl")

    # loss_val(EREN_loss_list, start, till, step, "EREN_MODULE_loss_list")
    # loss_val(EREN_best_loss_list, start, till, step, "EREN_BEST_MODULE_loss_list")
    # loss_val(NETA_loss_list, start, till, step, "NETA_MODULE_loss_list")
    # loss_val(NETA_best_loss_list, start, till, step, "NETA_BEST_MODULE_loss_list")

    g=10



    #PATH_pre = r"C:\Users\נטע\Documents\נטע\פרימרוז\שיעורי בית\project\fc_pre_2featues\fc_model.pth"
    #fc_model = torch.load(PATH_pre)
    #fc_model.eval()
    ##pre, fc_model_loss, y_gan = fc_model.evaluate(generated_data_loader)
    #plot_pre_vs_gan(pre, y_gan, "GanOutcome_vs_prediction", fc_model)
    #print("loss gan outcome vs prediction: "+ str(fc_model_loss))
    m=10