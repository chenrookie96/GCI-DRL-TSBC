import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import copy
import warnings
warnings.filterwarnings("ignore")

np.random.seed(10086)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 定义一些超参数
learning_rate = 0.01  # 学习率
batch_size = 32  # 批次大小
discount_rate = 0.4  # 折扣率
memory_size = 3000  # 经验回放的容量

torch.manual_seed(1234)


class Net(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(n_states, 300)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(300, 300)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(300, 300)
        self.fc3.weight.data.normal_(0, 0.1)
        self.fc4 = nn.Linear(300, 300)
        self.fc4.weight.data.normal_(0, 0.1)
        self.fc5 = nn.Linear(300, 300)
        self.fc5.weight.data.normal_(0, 0.1)
        self.fc6 = nn.Linear(300, 300)
        self.fc6.weight.data.normal_(0, 0.1)
        self.fc7 = nn.Linear(300, 300)
        self.fc7.weight.data.normal_(0, 0.1)
        self.fc8 = nn.Linear(300, 300)
        self.fc8.weight.data.normal_(0, 0.1)
        self.fc9 = nn.Linear(300, 300)
        self.fc9.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(300, n_actions)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)
        x = F.relu(x)
        x = self.fc7(x)
        x = F.relu(x)
        x = self.fc8(x)
        x = F.relu(x)
        x = self.fc9(x)
        x = F.relu(x)
        # 前向传播，返回输出层的值
        return self.out(x)


class DQN:
    def __init__(self, n_states, n_actions, model_load_path):
        # 创建DQN网络的实例
        self.eval_net = torch.load(model_load_path)  # 创建网络
        self.target_net = torch.load(model_load_path)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=learning_rate)  # 创建优化器
        self.criterion = nn.MSELoss().to(device)  # 创建损失函数

        # 创建一个列表来存储经验
        self.memory = np.zeros((memory_size, n_states * 2 + 2))
        self.memory_counter = 0
        self.learn_step_counter = 0
        self.n_actions = n_actions  # 状态空间个数
        self.n_states = n_states  # 动作空间大小
        self.count_loss = 0

    def choose_action(self, x, Tmin, Tmax, epsilon, tml):
        x = torch.unsqueeze(torch.FloatTensor(x), 0).to(device)
        # 以一定的概率随机选择一个动作
        if np.random.uniform() < 0.96:
            if tml < Tmin:
                action = 0
            elif tml > Tmax:
                action = 1
            else:
                action_value = self.eval_net.forward(x).to(device)
                # print(action_value)
                action = torch.max(action_value.cpu(), 1)[1].data.numpy()[0]
        else:
            if tml < Tmin:
                action = 0
            elif tml > Tmax:
                action = 1
            else:
                action = np.random.randint(0, self.n_actions)

            # print("action=", action)
        return action

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def train_network(self):
        global epsilon
        if self.learn_step_counter % 100 == 0:
            # 把最新的eval 预测网络 推 给target Q现实网络
            # 也就是变成，还未变化的eval网
            # print("save target net")
            self.target_net.load_state_dict((self.eval_net.state_dict()))
        self.learn_step_counter += 1
        # 如果经验的数量小于批次大小，不进行训练
        # 从经验中随机抽取一个批次的数据
        sample_index = np.random.choice(memory_size, batch_size)
        memory = self.memory[sample_index, :]
        # 将数据分解为状态，动作，奖励，下一个状态，是否结束的张量
        states = torch.FloatTensor(memory[:, :5]).to(device)
        actions = torch.LongTensor(memory[:, 5:6]).to(device)
        rewards = torch.FloatTensor(memory[:, 6:7]).to(device)
        next_states = torch.FloatTensor(memory[:, 7:12]).to(device)
        # 计算网络输出的当前状态的动作值
        current_values = self.eval_net(states).gather(1, actions).to(device)
        # 计算网络输出的下一个状态的最大动作值
        next_values = self.target_net(next_states).detach().to(device)
        # 计算目标值
        target_values = rewards + discount_rate * next_values.max(1)[0].unsqueeze(1).to(device)
        # 计算均方误差损失
        loss = self.criterion(current_values, target_values).to(device)
        self.count_loss += loss.cpu().detach().numpy()
        if (self.learn_step_counter % 100 == 0):
            print("loss=", self.count_loss / 100)
            self.count_loss = 0
        if (self.learn_step_counter % 500 == 0):
            torch.save(self.target_net, './test_data/208/model0_3000.pth')
            np.save('memory.npy', self.memory)
            print('save model')
            print('memory save')
        # 反向传播，更新网络参数
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()