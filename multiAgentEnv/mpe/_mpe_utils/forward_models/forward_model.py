import torch
import numpy as np
# import yaml
import os
from ..utils import load_params, get_scaler, get_rescaler


class ForwardModel():
    def __init__(self, robot_type=None):
        try:
            filename = f'/home/roblab1/PycharmProjects/MultiAgentEnv/maenv/mpe/_mpe_utils/{robot_type}_config.yaml'
            params = load_params(filename).get('forward_model')
        except FileNotFoundError:
            print(f"{filename} not found. Note that robot_type param should 'car' or 'fish' be defined")

        self.model_name = params['fd_model']
        self.robot_type = robot_type
        self.device = set_device(params['fdm_device'])
        self.dir = '/home/roblab1/PycharmProjects/MultiAgentEnv/maenv/mpe/_mpe_utils/forward_models'

        self.model = self.load_forward_model(params['fdm_input_dim'], params['fdm_output_dim'])
        self.model.to(self.device)

    def load_state_dict(self, path, net):
        state = torch.load(path, map_location=self.device)
        try:
            net.load_state_dict(state)
        except:
            try:
                net.load_state_dict(state['best_state'])
            except:
                net.load_state_dict(state['best_state_val'])
        net.eval()
        return net

    def load_forward_model(self, input_dim, output_dim):
        '''
        load forward model for eval mode
        '''
        if self.robot_type=='car':
            from maenv.mpe._mpe_utils.forward_models.car.NN import get_fd_model
            NN, scalers = get_fd_model(self.model_name)
            net = NN(input_dim=2, output_dim=3)
            min_x = [scalers[0], scalers[0]]
            max_x = [scalers[1], scalers[1]]
            min_y = [scalers[2], scalers[4], scalers[6]]
            max_y = [scalers[3], scalers[5], scalers[7]]

        elif self.robot_type == 'fish':
            from maenv.mpe._mpe_utils.forward_models.fish.NN import get_fd_model
            NN, scalers = get_fd_model(self.model_name)
            net = NN(input_dim, output_dim)
            min_x = [scalers[0], scalers[2], scalers[4]]
            max_x = [scalers[1], scalers[3], scalers[5]]
            min_y = [scalers[6], scalers[8], scalers[10]]
            max_y = [scalers[7], scalers[9], scalers[11]]

        else:
            raise NameError(self.robot_type)


        model_path = os.path.join(self.dir, self.robot_type, 'models', self.model_name)
        model = self.load_state_dict(model_path, net)

        self.scaledown_x = get_scaler(np.array(min_x), np.array(max_x))
        self.scaledown_y = get_scaler(np.array(min_y), np.array(max_y))
        self.rescale_x = get_rescaler(np.array(min_x), np.array(max_x))
        self.rescale_y = get_rescaler(np.array(min_y), np.array(max_y))

        return model

    def predict_single(self, x):
        x = torch.from_numpy(x).unsqueeze(0).to(self.device)
        with torch.no_grad():
            y = self.model(x)
        y = y.squeeze().detach().cpu().numpy()
        y = self.rescale_y(y)
        return y

    def predict_batch(self, x):
        x = torch.from_numpy(x).to(self.device)
        with torch.no_grad():
            y = self.model(x)
        y = y.detach().cpu().numpy()
        y = self.rescale_y(y)
        return y


def set_device(device):
    return torch.device(device if torch.cuda.is_available() else 'cpu') if device != 'cpu' else torch.device('cpu')



