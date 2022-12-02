import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.nn import init, Parameter
import math 
from torch.autograd import Variable 

###############################################
## Naive Q Network
###############################################
class LinearDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims):
        super(LinearDeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        layer1 = F.relu(self.fc1(state))
        actions = self.fc2(layer1)

        return actions
    
###############################################
## Deep Q Network
###############################################
class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.fc2 = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        # conv3 shape is BS x n_filters x H x W
        conv_state = conv3.view(conv3.size()[0], -1)
        # conv_state shape is BS x (n_filters * H * W)
        flat1 = F.relu(self.fc1(conv_state))
        actions = self.fc2(flat1)

        return actions

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))
        
        
###############################################
## Dueling Deep Q Network
###############################################
class DuelingDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DuelingDeepQNetwork, self).__init__()

        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.fc1 = nn.Linear(fc_input_dims, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        conv_state = conv3.view(conv3.size()[0], -1)
        flat1 = F.relu(self.fc1(conv_state))
        flat2 = F.relu(self.fc2(flat1))

        V = self.V(flat2)
        A = self.A(flat2)

        return V, A

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))
        

# Noisy linear layer with independent Gaussian noise
# class NoisyLinear(nn.Linear):
#     def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
#         super(NoisyLinear, self).__init__(in_features, out_features, bias=True)  # TODO: Adapt for no bias
#         # µ^w and µ^b reuse self.weight and self.bias
#         self.sigma_init = sigma_init
#         self.sigma_weight = Parameter(T.Tensor(out_features, in_features))  # σ^w
#         self.sigma_bias = Parameter(T.Tensor(out_features))  # σ^b
#         self.register_buffer('epsilon_weight', T.zeros(out_features, in_features))
#         self.register_buffer('epsilon_bias', T.zeros(out_features))
#         self.reset_parameters()

#     def reset_parameters(self):
#         if hasattr(self, 'sigma_weight'):  # Only init after all params added (otherwise super().__init__() fails)
#             init.uniform(self.weight, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
#             init.uniform(self.bias, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
#             init.constant(self.sigma_weight, self.sigma_init)
#             init.constant(self.sigma_bias, self.sigma_init)

#     def forward(self, input):
#         return F.linear(input, self.weight + self.sigma_weight * Variable(self.epsilon_weight), self.bias + self.sigma_bias * Variable(self.epsilon_bias))

#     def sample_noise(self):
#         self.epsilon_weight = T.randn(self.out_features, self.in_features)
#         self.epsilon_bias = T.randn(self.out_features)

#     def remove_noise(self):
#         self.epsilon_weight = T.zeros(self.out_features, self.in_features)
#         self.epsilon_bias = T.zeros(self.out_features)


# class ActorCritic(nn.Module):
#     def __init__(self, observation_space, action_space, hidden_size, sigma_init, no_noise):
#         super(ActorCritic, self).__init__()
#         self.no_noise = no_noise
#         self.state_size = observation_space.shape[0]
#         self.action_size = action_space.n

#         self.relu = nn.ReLU(inplace=True)
#         self.softmax = nn.Softmax(dim=1)

#         self.fc1 = nn.Linear(self.state_size, hidden_size)
#         self.lstm = nn.LSTMCell(hidden_size, hidden_size)
#         if no_noise:
#             self.fc_actor = nn.Linear(hidden_size, self.action_size)
#             self.fc_critic = nn.Linear(hidden_size, 1)
#         else:
#             self.fc_actor = NoisyLinear(hidden_size, self.action_size, sigma_init=sigma_init)
#             self.fc_critic = NoisyLinear(hidden_size, 1, sigma_init=sigma_init)

#     def forward(self, x, h):
#         x = self.relu(self.fc1(x))
#         h = self.lstm(x, h)  # h is (hidden state, cell state)
#         x = h[0]
#         policy = self.softmax(self.fc_actor(x)).clamp(max=1 - 1e-20)  # Prevent 1s and hence NaNs
#         V = self.fc_critic(x)
#         return policy, V, (h[0], h[1])

#     def sample_noise(self):
#         if not self.no_noise:
#             self.fc_actor.sample_noise()
#             self.fc_critic.sample_noise()

#     def remove_noise(self):
#         if not self.no_noise:
#             self.fc_actor.remove_noise()
#             self.fc_critic.remove_noise()
            
#     def save_checkpoint(self):
#         print('... saving checkpoint ...')
#         T.save(self.state_dict(), self.checkpoint_file)

#     def load_checkpoint(self):
#         print('... loading checkpoint ...')
#         self.load_state_dict(T.load(self.checkpoint_file))