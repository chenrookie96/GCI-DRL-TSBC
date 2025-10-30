# src/algorithms/drl_tsbc.py
"""
DRL-TSBC: Deep Reinforcement Learning-based dynamic bus Timetable Scheduling method with Bidirectional Constraints
基于深度强化学习的双向动态公交时刻表排班算法实现

基于论文: 基于交通大数据的公交排班和调度机制研究 - 谢嘉昊, 王玺钧
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import Tuple, List, Dict, Any

class DQNNetwork(nn.Module):
    """DQN神经网络 - 用于DRL-TSBC双向时刻表排班
    
    根据论文表2-2规范:
    - 12个隐藏层
    - 每层500个神经元
    - 激活函数: ReLU
    - 学习率: 0.001
    """
    
    def __init__(self, state_dim: int = 10, action_dim: int = 4):
        """
        初始化DQN网络
        
        Args:
            state_dim: 状态维度 (双向状态特征，固定为10)
            action_dim: 动作维度 (4种组合: 00, 01, 10, 11)
        """
        super(DQNNetwork, self).__init__()
        
        layers = []
        
        # 输入层到第一个隐藏层
        layers.extend([
            nn.Linear(state_dim, 500),
            nn.ReLU()
        ])
        
        # 12个隐藏层，每层500个神经元
        for _ in range(11):  # 已经有1层，再加11层共12层
            layers.extend([
                nn.Linear(500, 500),
                nn.ReLU()
            ])
            
        # 输出层
        layers.append(nn.Linear(500, action_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class ReplayBuffer:
    """经验回放缓冲区
    
    根据论文表2-2规范:
    - 经验池大小: 3000
    """
    
    def __init__(self, capacity: int = 3000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        
        return (
            torch.FloatTensor(state),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(next_state),
            torch.BoolTensor(done)
        )
        
    def __len__(self) -> int:
        return len(self.buffer)

class DRLTSBCAgent:
    """DRL-TSBC智能体 - 双向动态公交时刻表排班
    
    根据论文表2-2规范:
    - 学习率: 0.001
    - 折现系数gamma: 0.4
    - epsilon: 0.1 (固定值)
    - 批次大小: 64
    - 经验池大小: 3000
    - 学习频率: 5
    - 参数更新频率: 100
    """
    
    def __init__(self, 
                 state_dim: int = 10,  # 双向状态特征
                 action_dim: int = 4,   # 4种动作组合
                 learning_rate: float = 0.001,
                 gamma: float = 0.4,  # 论文规范: 0.4
                 epsilon: float = 0.1,  # 论文规范: 固定0.1
                 batch_size: int = 64,  # 论文规范: 64
                 buffer_size: int = 3000,  # 论文规范: 3000
                 learning_freq: int = 5,  # 论文规范: 每5步学习一次
                 target_update_freq: int = 100,  # 论文规范: 100
                 omega: float = 1.0/1000,  # 论文规范: 1/1000
                 zeta: float = 0.002):  # 论文规范: 0.002
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon  # 固定值，不衰减
        self.batch_size = batch_size
        self.learning_freq = learning_freq
        self.target_update_freq = target_update_freq
        
        # 神经网络
        self.q_network = DQNNetwork(state_dim, action_dim=action_dim)
        self.target_network = DQNNetwork(state_dim, action_dim=action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        
        # 训练计数器
        self.train_step = 0
        self.step_counter = 0  # 用于控制学习频率
        
        # 约束参数 (基准配置)
        self.t_min = 5   # 最小发车间隔(分钟)
        self.t_max = 20  # 最大发车间隔(分钟)
        
        # 奖励函数参数
        self.omega = omega  # 等待时间权重
        self.zeta = zeta    # 发车次数差异权重
        self.balance_threshold = 1  # 硬约束阈值（论文要求上下行发车次数差≤1）
        
        # 动作映射: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        self.action_map = {
            0: (0, 0),  # 上下行都不发车
            1: (0, 1),  # 上行不发车，下行发车
            2: (1, 0),  # 上行发车，下行不发车
            3: (1, 1)   # 上下行都发车
        }
        
        # 同步目标网络
        self.update_target_network()
        
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def get_state_features(self, env_state: Dict[str, Any]) -> np.ndarray:
        """
        提取双向状态特征（严格按照论文公式2.1-2.14）
        
        论文状态空间定义：
        - 时间状态 (全局): a1_m = t_h/24, a2_m = t_m/60
        - 上行状态: x1_m, x2_m, x3_m, x4_m
        - 下行状态: y1_m, y2_m, y3_m, y4_m
        
        具体公式：
        - x1_m = C_max^(m,up) / C_max  (公式2.7: 满载率)
        - x2_m = W_m^up / μ            (公式2.8: 归一化等待时间, μ=5000)
        - x3_m = o_m / e_m             (公式2.9: 运力利用率)
        - x4_m = (c_m^up - c_m^down) / δ  (公式2.10: 发车次数差异, δ=200)
        
        注意：论文中x4和y4表示的是发车次数差异，而不是单独的发车次数
        """
        # 时间特征 (公式2.5-2.6)
        a1_m = env_state['time_hour'] / 24.0
        a2_m = env_state['time_minute'] / 60.0
        
        # 上行方向特征
        x1_m = env_state['up_load_factor']              # 公式2.7: 满载率
        x2_m = env_state['up_waiting_time']             # 公式2.8: 归一化等待时间
        x3_m = env_state['up_capacity_utilization']     # 公式2.9: 运力利用率
        
        # 公式2.10: 发车次数差异 (c_up - c_down) / δ
        c_up = env_state['up_departure_count']
        c_down = env_state['down_departure_count']
        delta = 200  # 论文规范
        x4_m = (c_up - c_down) / delta
        
        # 下行方向特征 (公式2.11-2.14)
        y1_m = env_state['down_load_factor']            # 公式2.11: 满载率
        y2_m = env_state['down_waiting_time']           # 公式2.12: 归一化等待时间
        y3_m = env_state['down_capacity_utilization']   # 公式2.13: 运力利用率
        
        # 公式2.14: 发车次数 (注意：论文这里只用c_down/δ，不是差异)
        y4_m = c_down / delta
        
        features = np.array([
            a1_m, a2_m,      # 时间特征
            x1_m, x2_m, x3_m, x4_m,  # 上行特征
            y1_m, y2_m, y3_m, y4_m   # 下行特征
        ])
        
        return features
        
    def select_action(self, state: np.ndarray, last_intervals: Dict[str, int], 
                     up_count: int = 0, down_count: int = 0) -> Tuple[int, int]:
        """
        选择动作（带双向约束的ε-贪婪策略）
        
        根据论文算法2.1，流程为：
        1. 先用DQN选择动作
        2. 然后应用发车间隔约束进行修正
        3. 应用发车次数平衡硬约束
        
        Args:
            state: 当前状态特征
            last_intervals: {'up': 上行间隔, 'down': 下行间隔}
            up_count: 当前上行发车次数
            down_count: 当前下行发车次数
            
        Returns:
            (a_up, a_down): 上行和下行的发车决策
        """
        
        # 步骤1: ε-贪婪策略选择初始动作
        if random.random() < self.epsilon:
            # 随机选择动作
            action_idx = random.randint(0, self.action_dim - 1)
        else:
            # 贪婪选择
            with torch.no_grad():
                # 确保tensor在正确的设备上
                device = next(self.q_network.parameters()).device
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.q_network(state_tensor).squeeze()
                action_idx = q_values.argmax().item()
        
        # 获取初始动作
        a_up, a_down = self.action_map[action_idx]
        
        # 步骤2: 应用发车间隔约束（论文算法2.1第13-20行）
        # 上行约束检查
        if last_intervals['up'] < self.t_min:
            a_up = 0  # 强制不发车
        elif last_intervals['up'] >= self.t_max:
            a_up = 1  # 强制发车
            
        # 下行约束检查  
        if last_intervals['down'] < self.t_min:
            a_down = 0  # 强制不发车
        elif last_intervals['down'] >= self.t_max:
            a_down = 1  # 强制发车
        
        # 步骤3: 应用发车次数平衡硬约束
        diff = up_count - down_count
        
        # 如果上行多发车超过阈值
        if diff > self.balance_threshold:
            if a_up == 1:  # 禁止上行发车
                a_up = 0
            # 如果下行满足发车条件，强制下行发车
            if a_down == 0 and last_intervals['down'] >= self.t_min:
                a_down = 1
        
        # 如果下行多发车超过阈值
        elif diff < -self.balance_threshold:
            if a_down == 1:  # 禁止下行发车
                a_down = 0
            # 如果上行满足发车条件，强制上行发车
            if a_up == 0 and last_intervals['up'] >= self.t_min:
                a_up = 1
                
        return (a_up, a_down)
                
    def calculate_reward(self, action: Tuple[int, int], env_state: Dict[str, Any]) -> float:
        """
        严格按照论文公式2.16-2.17计算奖励
        
        论文公式：
        - 公式2.16 (上行): 
          a^up=0: r_up = 1 - (o_up/e_up) - (ω×W_up) - (β×d_up) + ζ(c_up - c_down)
          a^up=1: r_up = (o_up/e_up) - (β×d_up) - ζ(c_up - c_down)
        
        - 公式2.17 (下行):
          a^down=0: r_down = 1 - (o_down/e_down) - (ω×W_down) - (β×d_down) - ζ(c_up - c_down)
          a^down=1: r_down = (o_down/e_down) - (β×d_down) + ζ(c_up - c_down)
        
        注意：论文中使用的是 o_m/e_m (运力利用率)，而不是满载率
        """
        a_up, a_down = action
        
        # 获取状态特征（严格对应论文符号）
        # o_m/e_m: 运力利用率 (capacity_utilization)
        o_up_div_e_up = env_state['up_capacity_utilization']
        o_down_div_e_down = env_state['down_capacity_utilization']
        
        # W_m: 等待时间（已归一化）
        W_up = env_state['up_waiting_time']
        W_down = env_state['down_waiting_time']
        
        # d_m: 滞留乘客数
        d_up = env_state['up_stranded_passengers']
        d_down = env_state['down_stranded_passengers']
        
        # c_m: 发车次数
        c_up = env_state['up_departure_count']
        c_down = env_state['down_departure_count']
        
        # 论文规范参数
        beta = 0.2      # β: 滞留乘客惩罚权重
        omega = self.omega  # ω: 等待时间权重 (1/1000)
        zeta = self.zeta    # ζ: 发车次数平衡权重 (0.002)
        
        # 上行奖励（论文公式2.16）
        if a_up == 0:  # 不发车
            r_up = 1 - o_up_div_e_up - (omega * W_up) - (beta * d_up) + (zeta * (c_up - c_down))
        else:  # 发车
            r_up = o_up_div_e_up - (beta * d_up) - (zeta * (c_up - c_down))
        
        # 下行奖励（论文公式2.17）
        if a_down == 0:  # 不发车
            r_down = 1 - o_down_div_e_down - (omega * W_down) - (beta * d_down) - (zeta * (c_up - c_down))
        else:  # 发车
            r_down = o_down_div_e_down - (beta * d_down) + (zeta * (c_up - c_down))
        
        # 总奖励（论文公式2.15）
        total_reward = r_up + r_down
        
        return total_reward
    
    def calculate_improved_reward(self, action: Tuple[int, int], env_state: Dict[str, Any], 
                                  down_weight: float = 1.5, delta: float = 0.005,
                                  min_acceptable: int = 70, max_acceptable: int = 76) -> float:
        """
        v1.4改进的奖励函数 - 软约束发车次数控制
        
        核心改进：
        1. 保持down_weight对下行的额外关注
        2. 使用软约束而非硬惩罚控制发车次数
        3. 允许70-76次的合理范围，只惩罚严重偏离
        
        Args:
            action: (a_up, a_down) 发车动作
            env_state: 环境状态
            down_weight: 下行权重因子（默认1.5）
            delta: 发车次数偏离惩罚系数（默认0.005，比v1.2的0.01更温和）
            min_acceptable: 最小可接受发车次数（默认70）
            max_acceptable: 最大可接受发车次数（默认76）
            
        Returns:
            总奖励值
        """
        a_up, a_down = action
        
        # 获取状态信息
        o_up = env_state['up_capacity_utilization']
        w_up = env_state['up_waiting_time'] 
        d_up = env_state['up_stranded_passengers']
        c_up = env_state['up_departure_count']
        waiting_up = env_state.get('up_waiting_passengers', 0)
        
        o_down = env_state['down_capacity_utilization']
        w_down = env_state['down_waiting_time']
        d_down = env_state['down_stranded_passengers']
        c_down = env_state['down_departure_count']
        waiting_down = env_state.get('down_waiting_passengers', 0)
        
        # 论文规范参数（严格不变）
        beta = 0.2
        omega = self.omega  # 1/1000
        zeta = self.zeta    # 0.002
        gamma = 0.05
        
        # 计算发车次数差异
        departure_diff = c_up - c_down
        
        # 计算平均载客率
        o_up_per_trip = o_up / c_up if c_up > 0 else 0
        o_down_per_trip = o_down / c_down if c_down > 0 else 0
        
        # 归一化等待乘客数
        waiting_up_norm = min(waiting_up / 100.0, 1.0)
        waiting_down_norm = min(waiting_down / 100.0, 1.0)
        
        # 上行奖励（保持论文原有逻辑）
        if a_up == 0:
            r_up = 1 - o_up_per_trip - (omega * w_up) - (beta * d_up) - (gamma * waiting_up_norm) + (zeta * departure_diff)
        else:
            r_up = o_up_per_trip - (beta * d_up) + (gamma * waiting_up_norm * 0.5) - (zeta * departure_diff)
        
        # 下行奖励（应用权重因子）
        if a_down == 0:
            # 不发车时的惩罚增强
            base_penalty = 1 - o_down_per_trip - (omega * w_down) - (beta * d_down) - (gamma * waiting_down_norm) - (zeta * departure_diff)
            r_down = base_penalty * down_weight
        else:
            # 发车时的奖励增强
            base_reward = o_down_per_trip - (beta * d_down) + (gamma * waiting_down_norm * 0.5) + (zeta * departure_diff)
            r_down = base_reward * down_weight
        
        # 基础奖励
        base_total = r_up + r_down
        
        # v1.4新增：软约束发车次数控制
        # 计算上下行的平均发车次数
        avg_departures = (c_up + c_down) / 2.0
        
        # 计算偏离惩罚（只在超出合理范围时）
        departure_penalty = 0.0
        if avg_departures < min_acceptable:
            # 发车太少，轻微惩罚
            deviation = min_acceptable - avg_departures
            departure_penalty = delta * deviation
        elif avg_departures > max_acceptable:
            # 发车太多，轻微惩罚
            deviation = avg_departures - max_acceptable
            departure_penalty = delta * deviation
        # 在min_acceptable到max_acceptable范围内，不惩罚
        
        final_reward = base_total - departure_penalty
        
        return final_reward
        
    def train(self):
        """训练智能体"""
        # 预热期：收集足够的经验再开始训练
        # 标准DQN实践：batch_size的10倍（640步）
        if len(self.replay_buffer) < 640:
            return None
            
        # 采样批次数据
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # 确保所有tensor在正确的设备上
        device = next(self.q_network.parameters()).device
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)
        
        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            
        # 计算损失
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # epsilon固定为0.1，不衰减（论文规范）
        
        # 更新目标网络
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.update_target_network()
            
        return loss.item()
        
    def action_to_index(self, action: Tuple[int, int]) -> int:
        """将动作元组转换为索引"""
        for idx, mapped_action in self.action_map.items():
            if mapped_action == action:
                return idx
        return 0  # 默认返回0
        
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step
        }, filepath)
        
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.train_step = checkpoint['train_step']

class BidirectionalTimetableOptimizer:
    """双向时刻表优化器"""
    
    def __init__(self, agent: DRLTSBCAgent):
        self.agent = agent
    
    def balance_timetable(self, up_departures: List[int], down_departures: List[int], 
                         service_end: int, t_max: int = 20) -> Tuple[List[int], List[int]]:
        """
        严格按照论文算法2.2实现的后处理算法
        
        论文算法2.2步骤8-13：
        1. 选择发车次数更多的方向
        2. 从时刻表中删除该方向的倒数第二次发车
        3. k ← 此时该方向的总发车次数
        4. WHILE 第k次和第k-1次发车间隔 > T_max:
        5.   将第k-1次发车时间推迟，使其与第k次间隔为T_max
        6.   k = k - 1
        
        Args:
            up_departures: 上行发车时间列表
            down_departures: 下行发车时间列表
            service_end: 服务结束时间
            t_max: 最大发车间隔（默认20分钟）
            
        Returns:
            调整后的(上行发车时间, 下行发车时间)
        """
        up_count = len(up_departures)
        down_count = len(down_departures)
        
        # 如果发车次数相等，直接返回
        if up_count == down_count:
            return sorted(up_departures), sorted(down_departures)
        
        # 步骤1: 选择发车次数更多的方向
        if up_count > down_count:
            more_deps = sorted(up_departures.copy())
            less_deps = sorted(down_departures.copy())
            is_up_more = True
        else:
            more_deps = sorted(down_departures.copy())
            less_deps = sorted(up_departures.copy())
            is_up_more = False
        
        # 步骤2-6: 重复删除倒数第二次发车，直到发车次数相等
        while len(more_deps) > len(less_deps):
            if len(more_deps) >= 2:
                # 删除倒数第二次发车
                more_deps.pop(-2)
                
                # 向前调整发车时刻
                k = len(more_deps) - 1  # 最后一次发车的索引
                
                while k > 0:
                    # 计算第k次和第k-1次发车的间隔
                    interval = more_deps[k] - more_deps[k-1]
                    
                    if interval > t_max:
                        # 将第k-1次发车时间推迟，使其与第k次间隔为t_max
                        more_deps[k-1] = more_deps[k] - t_max
                        k -= 1
                    else:
                        # 间隔不超过t_max，停止调整
                        break
            else:
                # 只剩一次发车，无法再删除
                break
        
        # 返回调整后的时刻表
        if is_up_more:
            return more_deps, less_deps
        else:
            return less_deps, more_deps
        
    def optimize_timetable(self, env, episodes: int = 1000) -> List[Dict[str, Any]]:
        """
        优化双向时刻表
        
        Args:
            env: 双向公交仿真环境
            episodes: 训练轮数
            
        Returns:
            训练历史记录
        """
        training_history = []
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            episode_actions = []
            last_intervals = {'up': 0, 'down': 0}
            
            while not env.is_done():
                # 获取状态特征
                state_features = self.agent.get_state_features(state)
                
                # 选择动作
                action = self.agent.select_action(state_features, last_intervals)
                
                # 执行动作
                next_state, _, done = env.step(action)
                
                # 计算实际奖励
                actual_reward = self.agent.calculate_reward(action, state)
                
                # 存储经验
                next_state_features = self.agent.get_state_features(next_state)
                action_idx = self.agent.action_to_index(action)
                
                self.agent.replay_buffer.push(
                    state_features, action_idx, actual_reward, next_state_features, done
                )
                
                # 训练 (根据论文规范，每5步学习一次)
                self.agent.step_counter += 1
                loss = None
                if self.agent.step_counter % self.agent.learning_freq == 0:
                    loss = self.agent.train()
                
                # 更新状态和间隔
                state = next_state
                episode_reward += actual_reward
                episode_actions.append(action)
                
                # 更新发车间隔
                if action[0] == 1:  # 上行发车
                    last_intervals['up'] = 0
                else:
                    last_intervals['up'] += 1
                    
                if action[1] == 1:  # 下行发车
                    last_intervals['down'] = 0
                else:
                    last_intervals['down'] += 1
                    
            # 记录训练历史
            training_history.append({
                'episode': episode,
                'reward': episode_reward,
                'epsilon': self.agent.epsilon,
                'actions': episode_actions,
                'loss': loss if loss else 0,
                'up_departures': sum(1 for a in episode_actions if a[0] == 1),
                'down_departures': sum(1 for a in episode_actions if a[1] == 1)
            })
            
            if episode % 100 == 0:
                up_deps = training_history[-1]['up_departures']
                down_deps = training_history[-1]['down_departures']
                print(f"Episode {episode}, Reward: {episode_reward:.2f}, "
                      f"Up/Down Departures: {up_deps}/{down_deps}, "
                      f"Epsilon: {self.agent.epsilon:.3f}")
                
        return training_history






