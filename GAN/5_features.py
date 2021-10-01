
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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
PATH_gan = r"E:\CarProject\NewCode_Project\gan"

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        #self.fc1 = nn.Linear(6, 32)
        #torch.nn.init.kaiming_normal_(self.fc1.weight)
        #self.fc2 = nn.Linear(32, 16)
        #torch.nn.init.kaiming_normal_(self.fc2.weight)
        #self.fc3 = nn.Linear(16, 8)
        #torch.nn.init.kaiming_normal_(self.fc3.weight)
        #self.fc4 = nn.Linear(8, 5)
        #torch.nn.init.kaiming_normal_(self.fc4.weight)

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
        #x = F.tanh(self.fc1(x))
        #x = F.dropout(x, p, training)
        #x = F.tanh(self.fc2(x))
        #x = F.dropout(x, p, training)
        #x = F.tanh(self.fc3(x))
        #x = F.dropout(x, p, training)
        #x = torch.tanh(self.fc4(x))


        x = F.leaky_relu(self.fc1(x))
        #x = F.dropout(x, p, training)
        x = F.leaky_relu(self.fc2(x))
        #x = F.dropout(x, p, training)
        x = F.leaky_relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        #self.fc1 = nn.Linear(5, 32)
        #torch.nn.init.kaiming_normal_(self.fc1.weight)
        #self.fc2 = nn.Linear(32, 16)
        #torch.nn.init.kaiming_normal_(self.fc2.weight)
        #self.fc3 = nn.Linear(16, 8)
        #torch.nn.init.kaiming_normal_(self.fc3.weight)
        #self.fc4 = nn.Linear(8, 6)
        #torch.nn.init.kaiming_normal_(self.fc4.weight)
        #self.fc5 = nn.Linear(6, 1)
        #torch.nn.init.kaiming_normal_(self.fc5.weight)

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
        #x = F.tanh(self.fc1(x))
        #x = F.dropout(x, p, training)
        #x = F.tanh(self.fc2(x))
        #x = F.dropout(x, p, training)
        #x = F.tanh(self.fc3(x))
        #x = F.tanh(self.fc4(x))
        #x = torch.sigmoid(self.fc5(x))

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


    def training_gan(self, train, lr, beta1, b_size, epoch_dis, fixed_noise,opt_name):
        try:
            num_epochs= int(len(train)/b_size)*2+700
            criterion = nn.BCELoss()
            optimizerD = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
            optimizerG = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))

            real_label = 1.0
            fake_label = 0.

            G_losses = []
            D_losses = []
            D_real_losses = []
            D_fake_losses = []

            for e in  tqdm(range(1, num_epochs + 1)):
                # print(f"'Epoch {e}' ")
                for i in range(epoch_dis):
                    # Update discriminator network
                    ## Train with all-real batch
                    self.discriminator.train()
                    self.generator.eval()
                    self.discriminator.zero_grad()
                    real_features = train[np.random.randint(low=0, high=train.shape[0], size=int(b_size/2))]
                    label = torch.full((int(b_size/2),), real_label, dtype=torch.float, device=device)
                    output = self.discriminator(real_features, training=True).view(-1)
                    errD_real = criterion(output, label)
                    errD_real.backward()
                    D_x = output.mean().item()
                    ## Train with all-fake batch
                    noise = torch.from_numpy(np.random.normal(0, 0.2, [int(b_size/2), 6])).type(torch.FloatTensor)
                    noise = noise.to(device)
                    fake = self.generator(noise, training=False)
                    label.fill_(fake_label).type(torch.FloatTensor)
                    output = self.discriminator(fake.detach(), training=True).view(-1)
                    errD_fake = criterion(output, label)
                    errD_fake.backward()
                    D_G_z1 = output.mean().item()
                    # Compute error of D as sum over the fake and the real batches
                    errD = errD_real + errD_fake
                    print('err_total:', errD_real.item())
                    # Update D
                    optimizerD.step()

                # Update generator network:
                self.generator.train()
                self.discriminator.eval()
                self.generator.zero_grad()
                label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                noise = torch.from_numpy(np.random.normal(0, 0.2, [b_size, 6])).type(torch.FloatTensor)
                noise=noise.to(device)
                fake = self.generator(noise, training=True)
                output = self.discriminator(fake, training=False).view(-1)
                errG = criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())
                D_real_losses.append(errD_real.item())
                D_fake_losses.append(errD_fake.item())

                # Check how the generator and discriminator are doing
                if (e % 100 == 0) or (e == num_epochs):
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (e, num_epochs, i, b_size,
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                    real_features = train[np.random.randint(low=0, high=train.shape[0], size=int(b_size))]
                    #plot_all(fake.detach().numpy(), real_features.detach().numpy(), "epoch"+str(e), opt_name)
                    gen_data_fix_noise = self.generator(fixed_noise, training=False)
                    #plot_all(gen_data_fix_noise.detach().numpy(), real_features.detach().numpy(), "epoch" + str(e)+ "for fix_noise")
        except:
            torch.save(self, os.path.join(PATH_gan, str(opt_name)+".pth"))
        else:
            print("Training is complete")
            torch.save(self, os.path.join(PATH_gan, str(opt_name)+".pth"))

        return G_losses, D_losses, D_real_losses, D_fake_losses

def plot_all(gan_outcome, actual, title, model_name):
    pred1, pred2, pred3, pred4, pred5 = gan_outcome[:, 0], gan_outcome[:, 1], gan_outcome[:, 2], gan_outcome[:, 3], gan_outcome[:, 4]
    actual1, actual2, actual3, actual4, actual5 = actual[:, 0], actual[:, 1], actual[:, 2], actual[:, 3], actual[:, 4]
    x = np.arange(1, len(pred1) + 1)
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True, figsize=(10, 10))
    val = 0.
    fig.suptitle(title)
    ax1.plot(pred1 , np.zeros_like(pred1) + val, 'x', label='predicted')
    ax1.plot(actual1, np.zeros_like(actual1) + val, 'x', label='actual')
    ax1.set_ylabel('engine1')
    ax2.plot(pred2, np.zeros_like(pred2) , 'x', label='predicted')
    ax2.plot(actual2, np.zeros_like(actual2) , 'x', label='actual')
    ax2.set_ylabel('engine2')
    ax3.plot(pred3, np.zeros_like(pred3), 'x', label='predicted')
    ax3.plot(actual3, np.zeros_like(actual3), 'x', label='actual')
    ax3.set_ylabel('dx')
    ax4.plot(pred4, np.zeros_like(pred4), 'x', label='predicted')
    ax4.plot(actual4, np.zeros_like(actual4),  'x', label='actual')
    ax4.set_ylabel('dy')
    ax5.plot(pred5, np.zeros_like(pred5), 'x', label='predicted')
    ax5.plot(actual5, np.zeros_like(actual5), 'x', label='actual')
    ax5.set_ylabel('d_theta')
    ax5.set_xlabel('time step')
    plt.legend()
    plt.savefig('true&fake_data '+str(model_name)+'.png')
    plt.show()


def Normalization(x, x_min, x_max):
    x_nor = (x - x_min) / (x_max - x_min)
    x_nor = torch.from_numpy(x_nor).type(torch.float)
    return x_nor

def CarNormalization(pred_seq, real_seq, x_min=-120, x_max=120, dx_min=-21, dx_max=21,
                     dy_min=-50, dy_max=44, dtheta_min=-1.4, dtheta_max=1.3
):
    x_stand = Normalization(pred_seq, x_min, x_max)
    y_dx = Normalization(real_seq[:,0], dx_min, dx_max).view(-1,1)
    y_dy = Normalization(real_seq[:,1], dy_min, dy_max).view(-1,1)
    y_dtheta = Normalization(real_seq[:,2], dtheta_min, dtheta_max).view(-1,1)
    y_stand = torch.cat([y_dx, y_dy, y_dtheta], dim=1)
    return x_stand, y_stand

def loss_graph(G_losses, D_losses, D_real_losses, D_fake_losses, model_name):
    plt.plot(range(len(G_losses)),G_losses, label='generator_loss')
    plt.plot(range(len(D_losses)), D_losses, label='discriminator_loss')
    plt.plot(range(len(D_real_losses)),D_real_losses, label='discriminator_real_loss')
    plt.plot(range(len(D_fake_losses)), D_fake_losses, label='discriminator_fake_loss')
    plt.legend()
    plt.savefig(str(model_name)+' loss gen&dis.png')
    plt.show()

def process_data(data):
    dataset = pd.DataFrame({'engine1': data[0][:,0], 'engine2': data[0][:,1]})
    labels = pd.DataFrame({'label1': data[1][:, 0], 'label2': data[1][:, 1], 'label3': data[1][:, 2]})
    print(dataset.isna().sum())
    print(labels.isna().sum())

    # Scale features
    x_stnd, y_stand = CarNormalization(data[0], data[1])

    # max_engine = np.amax(data[0], axis=0)
    # min_engine = np.amin(data[0], axis=0)
    # max_dx_dy_dtheta = np.amax(data[1], axis=0)
    # min_dx_dy_dtheta = np.amin(data[1], axis=0)
    # scaler1 = MinMaxScaler()
    # data[0] = scaler1.fit_transform(data[0])
    # data[0] = data[0]-0.5
    #
    # scaler2 = MinMaxScaler()
    # data[1] = scaler2.fit_transform(data[1])
    # data[1] = data[1]-0.5


    x_train_tensor = x_stnd.type(torch.FloatTensor)
    y_train_tensor = y_stand.type(torch.FloatTensor)
    train = torch.cat((x_train_tensor, y_train_tensor), 1)
    return train



if __name__=='__main__':
    # hyper-parameters
    #model_dic= {'b_size': 64, 'lr': 0.0002,'beta_1': 0.4, 'epoch_dis': 3,'name_option': "gan_without_dropout"}
    model_dic= {'b_size': 64, 'lr': 0.0002,'beta_1': 0.4, 'epoch_dis': 10,'name_option': "gan_without_dropout_and_relu"}

    b_size = model_dic["b_size"]
    lr = model_dic["lr"]
    beta_1 = model_dic["beta_1"]
    epoch_dis = model_dic["epoch_dis"]
    opt_name = model_dic["name_option"]
    fixed_noise = torch.from_numpy(np.random.normal(0, 0.2, [b_size, 6])).type(torch.FloatTensor)
    #fixed_noise = torch.from_numpy(np.random.uniform(-0.5, 0.5, [b_size, 6])).type(torch.FloatTensor)

    t= open(r"E:\CarProject\NewCode_Project\gan\train.pkl", 'rb')
    data=pickle.load(t)

    train = process_data(data)


    gan = GanModel()
    G_losses, D_losses, D_real_losses, D_fake_losses= gan.training_gan(train, lr, beta_1, b_size,epoch_dis, fixed_noise,opt_name)
    loss_graph(G_losses, D_losses, D_real_losses, D_fake_losses, opt_name)

    neigh = NearestNeighbors(n_neighbors=8, radius=0.3)
    neigh.fit(train.numpy())
    test= gan.generator(fixed_noise, training=False)
    score = neigh.kneighbors(test.detach().numpy())
    print("NearestNeighbors score for gan model: "+str(score[0].mean()))


    #noise = torch.from_numpy(np.random.normal(0, 0.2, [b_size, 6])).type(torch.FloatTensor)
    #generated_data = gan.generator(noise)
    #generated_data_dataset = data_utils.TensorDataset(generated_data[:,0:2], generated_data[:,2:])
    #generated_data_loader = data_utils.DataLoader(generated_data_dataset, batch_size = b_size, num_workers=0,  shuffle = False)
    #PATH_pre = r"C:\Users\נטע\Documents\נטע\פרימרוז\שיעורי בית\project\fc_pre_2featues\fc_model.pth"
    #fc_model = torch.load(PATH_pre)
    #fc_model.eval()
    #pre, fc_model_loss, y_gan = fc_model.evaluate(generated_data_loader)
    #plot_pre_vs_gan(pre, y_gan, "GanOutcome_vs_prediction", fc_model)
    #print("loss gan outcome vs prediction: "+ str(fc_model_loss))
    m=10