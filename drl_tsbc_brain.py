"""
DRL-TSBC 神经网络模块
改动说明：
1. 基于RL_brain.py改造
2. 状态空间从5维改为10维（论文要求：2维时间+4维上行+4维下行）
3. 动作空间从2个改为4个（论文要求：(0,0), (0,1), (1,0), (1,1)）
4. 网络结构改为12层隐藏层，每层500神经元（论文表2-2要求）
5. 学习率改为0.001，batch_size改为64，gamma改为0.4（论文表2-2要求）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
warnings.filterwarnings("ignore")

np.random.seed(10086)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 论文表2-2中的超参数
learning_rate = 0.01  # 与原始代码保持一致
batch_size = 64  # 论文要求：批次大小B
discount_rate = 0.4  # 论文表2-2：折扣系数γ
memory_size = 3000  # 论文表2-2：经验池大小M

torch.manual_seed(1234)

class Net(nn.Module):
    """
    DQN网络结构
    改动：从9层300神经元改为12层500神经元（论文表2-2要求）
    """
    def __init__(self, n_states, n_actions):
        super(Net, self).__init__()
        
        # 12层隐藏层，每层500个神经元（论文表2-2要求）
        self.fc1 = nn.Linear(n_states, 500)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(500, 500)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(500, 500)
        self.fc3.weight.data.normal_(0, 0.1)
        self.fc4 = nn.Linear(500, 500)
        self.fc4.weight.data.normal_(0, 0.1)
        self.fc5 = nn.Linear(500, 500)
        self.fc5.weight.data.normal_(0, 0.1)
        self.fc6 = nn.Linear(500, 500)
        self.fc6.weight.data.normal_(0, 0.1)
        self.fc7 = nn.Linear(500, 500)
        self.fc7.weight.data.normal_(0, 0.1)
        self.fc8 = nn.Linear(500, 500)
        self.fc8.weight.data.normal_(0, 0.1)
        self.fc9 = nn.Linear(500, 500)
        self.fc9.weight.data.normal_(0, 0.1)
        self.fc10 = nn.Linear(500, 500)
        self.fc10.weight.data.normal_(0, 0.1)
        self.fc11 = nn.Linear(500, 500)
        self.fc11.weight.data.normal_(0, 0.1)
        self.fc12 = nn.Linear(500, 500)
        self.fc12.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(500, n_actions)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        x = F.relu(self.fc10(x))
        x = F.relu(self.fc11(x))
        x = F.relu(self.fc12(x))
        return self.out(x)


class DQN:
    """
    DQN智能体
    改动：
    1. 状态空间从5维改为10维
    2. 动作空间从2个改为4个
    3. 添加双向发车间隔约束检查
    """
    def __init__(self, n_states, n_actions, model_save_path=None):
        import os
        # 创建DQN网络的实例
        if model_save_path and os.path.exists(model_save_path):
            self.eval_net = torch.load(model_save_path, map_location=device, weights_only=False)
            self.target_net = torch.load(model_save_path, map_location=device, weights_only=False)
        else:
            self.eval_net = Net(n_states, n_actions).to(device)
            self.target_net = Net(n_states, n_actions).to(device)
        
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # 创建经验回放池
        self.memory = np.zeros((memory_size, n_states * 2 + 2))
        self.memory_counter = 0 
        self.learn_step_counter = 0
        self.n_actions = n_actions  # 4个动作组合
        self.n_states = n_states  # 10维状态
        self.count_loss = 0
    
    def choose_action(self, x, Tmin, Tmax, epsilon, interval_up, interval_down, 
                     up_count=0, down_count=0, balance_threshold=1):
        """
        选择动作
        改动：
        1. 添加双向发车间隔约束（论文算法2.1第8-17行）
        2. 添加发车次数平衡硬约束（论文算法2.1要求）
        3. 动作空间改为4个：(0,0), (0,1), (1,0), (1,1)
        
        参数：
        - x: 状态
        - Tmin: 最小发车间隔
        - Tmax: 最大发车间隔
        - epsilon: ε-贪婪策略参数
        - interval_up: 上行当前发车间隔
        - interval_down: 下行当前发车间隔
        - up_count: 上行已发车次数
        - down_count: 下行已发车次数
        - balance_threshold: 发车次数差异阈值（默认1）
        """
        x = torch.unsqueeze(torch.FloatTensor(x), 0).to(device)
        
        # ε-贪婪策略（原始代码逻辑：epsilon是使用网络的概率）
        if np.random.uniform() < epsilon:
            # 使用网络选择动作
            with torch.no_grad():
                action_value = self.eval_net.forward(x)
                action = torch.max(action_value, 1)[1].cpu().numpy()[0]
        else:
            # 随机探索
            action = np.random.randint(0, self.n_actions)
        
        # 将动作索引转换为(aup, adown)
        # 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
        aup = action // 2
        adown = action % 2
        
        # 步骤2: 应用发车间隔约束（论文算法2.1第13-20行）
        if interval_up < Tmin:
            aup = 0  # 强制不发车
        elif interval_up >= Tmax:
            aup = 1  # 强制发车
        
        if interval_down < Tmin:
            adown = 0  # 强制不发车
        elif interval_down >= Tmax:
            adown = 1  # 强制发车
        
        # 步骤3: 应用发车次数平衡硬约束（论文要求）
        diff = up_count - down_count
        
        # 如果上行多发车超过阈值
        if diff > balance_threshold:
            if aup == 1:  # 禁止上行发车
                aup = 0
            # 如果下行满足发车条件，强制下行发车
            if adown == 0 and interval_down >= Tmin:
                adown = 1
        
        # 如果下行多发车超过阈值
        elif diff < -balance_threshold:
            if adown == 1:  # 禁止下行发车
                adown = 0
            # 如果上行满足发车条件，强制上行发车
            if aup == 0 and interval_up >= Tmin:
                aup = 1
        
        # 将(aup, adown)转换回动作索引
        action = aup * 2 + adown
        
        return action, (aup, adown)
    
    def store_transition(self, state, action, reward, next_state):
        """存储经验到回放池"""
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % memory_size  
        self.memory[index, :] = transition 
        self.memory_counter += 1
    
    def train_network(self, model_save_path):
        """
        训练网络
        改动：状态维度从5改为10
        """
        # 每100步更新目标网络（论文表2-2：参数更新频率O=100）
        if self.learn_step_counter % 100 == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        
        self.learn_step_counter += 1
        
        # 从经验池随机采样
        sample_index = np.random.choice(memory_size, batch_size)
        memory = self.memory[sample_index, :]
        
        # 分解经验：状态10维，动作1维，奖励1维，下一状态10维
        states = torch.FloatTensor(memory[:, :10]).to(device)
        actions = torch.LongTensor(memory[:, 10:11]).to(device)
        rewards = torch.FloatTensor(memory[:, 11:12]).to(device)
        next_states = torch.FloatTensor(memory[:, 12:22]).to(device)
        
        # 计算当前Q值
        current_values = self.eval_net(states).gather(1, actions)
        
        # 计算目标Q值（论文公式2.18）
        with torch.no_grad():
            next_values = self.target_net(next_states)
            target_values = rewards + discount_rate * next_values.max(1)[0].unsqueeze(1)
        
        # 计算损失
        loss = self.criterion(current_values, target_values)
        self.count_loss += loss.item()
        
        if self.learn_step_counter % 100 == 0:
            print(f"loss = {self.count_loss / 100:.6f}")
            self.count_loss = 0
        
        # 每500步保存模型
        if self.learn_step_counter % 500 == 0:
            torch.save(self.target_net, model_save_path)
            np.save('memory.npy', self.memory)
            print('Model and memory saved')
        
        # 反向传播更新参数
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


import os
