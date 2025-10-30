# DRL-TSBC算法复现设计文档

## 概述

本设计文档详细描述了DRL-TSBC（Deep Reinforcement Learning-based dynamic bus Timetable Scheduling method with Bidirectional Constraints）算法的系统架构、核心组件设计和实现策略。该算法通过深度强化学习解决双向公交时刻表排班问题，在保证上下行发车次数相等的约束下，优化乘客等待时间。

### 设计目标

1. 实现完整的双向公交仿真环境，支持分钟级时间推进
2. 构建符合论文规格的DQN网络架构（12层隐藏层，每层500神经元）
3. 实现10维双向状态空间和4种动作组合
4. 设计双向奖励函数，平衡乘客等待时间和发车次数
5. 支持GPU加速训练（CUDA 11.8 + RTX 3060）
6. 复现论文实验结果，验证算法有效性

### 技术栈

- **编程语言**: Python 3.8+
- **深度学习框架**: PyTorch 2.0+
- **数值计算**: NumPy, Pandas
- **可视化**: Matplotlib, Seaborn
- **配置管理**: YAML
- **GPU支持**: CUDA 11.8

## 系统架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                      DRL-TSBC 系统架构                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │   训练模块       │  │   推理模块       │  │  评估模块    │  │
│  │                  │  │                  │  │              │  │
│  │  - DQN Agent     │  │  - 模型加载      │  │  - AWT计算   │  │
│  │  - 主网络        │  │  - 动作选择      │  │  - 发车统计  │  │
│  │  - 目标网络      │  │  - 约束检查      │  │  - 滞留统计  │  │
│  │  - 经验回放      │  │  - 后处理        │  │  - 图表生成  │  │
│  │  - ε-贪婪策略    │  │                  │  │              │  │
│  └──────────────────┘  └──────────────────┘  └──────────────┘  │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                    双向公交仿真环境层                            │
│                                                                  │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────┐     │
│  │  上行环境    │  │  环境控制器      │  │  下行环境    │     │
│  │              │  │                  │  │              │     │
│  │ - 车站状态   │  │ - 时间管理       │  │ - 车站状态   │     │
│  │ - 乘客队列   │  │ - 车辆调度       │  │ - 乘客队列   │     │
│  │ - 车辆位置   │  │ - 状态计算       │  │ - 车辆位置   │     │
│  │ - 满载率     │  │ - 奖励计算       │  │ - 满载率     │     │
│  └──────────────┘  └──────────────────┘  └──────────────┘     │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                         数据层                                   │
│                                                                  │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────┐     │
│  │  乘客数据    │  │  交通状况数据    │  │  线路配置    │     │
│  │              │  │                  │  │              │     │
│  │ - 上车时间   │  │ - 行驶时间       │  │ - 站点信息   │     │
│  │ - 上车站点   │  │ - 拥堵情况       │  │ - 运营时间   │     │
│  │ - 下车站点   │  │ - 站间距离       │  │ - 车辆参数   │     │
│  └──────────────┘  └──────────────────┘  └──────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 数据流设计

#### 训练数据流
```
乘客数据加载 → 环境初始化 → 状态计算(10维) → DQN决策(4动作) 
     ↓              ↓              ↓                ↓
发车执行 ← 约束检查 ← 动作选择 ← ε-贪婪策略
     ↓
奖励计算(双向) → 经验存储 → 批量采样 → 网络训练 → 参数更新
     ↓                                              ↓
环境更新 ←─────────────────────────────────── 目标网络更新
```

#### 推理数据流
```
环境状态 → 状态归一化 → DQN推理 → 动作选择 → 约束检查
     ↓                                           ↓
环境更新 ←─────────────────────────────── 发车执行
     ↓
时刻表记录 → 后处理(发车次数平衡) → 最终时刻表输出
```



## 核心组件设计

### 1. 双向公交仿真环境 (BidirectionalBusEnvironment)

#### 类设计

```python
class BidirectionalBusEnvironment:
    """双向公交仿真环境主类"""
    
    def __init__(self, line_config: dict):
        """
        初始化双向仿真环境
        
        Args:
            line_config: 线路配置，包含站点数、运营时间等
        """
        self.upward = DirectionEnvironment('upward', line_config['upward'])
        self.downward = DirectionEnvironment('downward', line_config['downward'])
        self.time_controller = TimeController()
        self.state_calculator = StateCalculator()
        self.reward_calculator = RewardCalculator()
        
    def reset(self) -> np.ndarray:
        """重置环境到初始状态，返回初始状态向量"""
        
    def step(self, action: tuple) -> tuple:
        """
        执行一步仿真
        
        Args:
            action: (a_up, a_down) 双向发车决策
            
        Returns:
            (next_state, reward, done, info)
        """
        
    def get_state(self) -> np.ndarray:
        """获取当前10维状态向量"""
```

#### 关键设计决策

1. **组合模式**: 使用两个DirectionEnvironment实例分别管理上行和下行方向，避免代码重复
2. **时间控制**: TimeController负责分钟级时间推进和首末班车判断
3. **状态计算**: StateCalculator集中处理10维状态向量的计算和归一化
4. **奖励计算**: RewardCalculator实现论文公式(2.16)和(2.17)的双向奖励函数

### 2. 单方向环境 (DirectionEnvironment)

#### 类设计

```python
class DirectionEnvironment:
    """单方向公交环境"""
    
    def __init__(self, direction: str, config: dict):
        """
        Args:
            direction: 'upward' 或 'downward'
            config: 该方向的配置（站点数、运营时间等）
        """
        self.direction = direction
        self.num_stations = config['num_stations']
        self.operating_hours = config['operating_hours']
        
        # 状态变量
        self.stations = [Station(i) for i in range(self.num_stations)]
        self.buses = []  # 在线公交车辆列表
        self.departure_times = []  # 发车时刻记录
        self.stranded_passengers = 0  # 滞留乘客数
        
    def dispatch_bus(self, minute: int):
        """在指定时刻发车"""
        
    def update(self, minute: int):
        """更新车辆位置和乘客状态"""
        
    def get_max_section_flow(self, minute: int) -> int:
        """计算最大断面客流"""
        
    def get_total_waiting_time(self, minute: int) -> float:
        """计算总等待时间"""
        
    def get_capacity_utilization(self, minute: int) -> float:
        """计算客运容量利用率"""
```

#### 核心功能

1. **车站管理**: 维护每个站点的等待乘客队列
2. **车辆追踪**: 跟踪所有在线车辆的位置和载客量
3. **乘客处理**: 模拟乘客上下车，记录滞留乘客
4. **指标计算**: 实时计算满载率、等待时间、容量利用率等

### 3. 状态计算器 (StateCalculator)

#### 类设计

```python
class StateCalculator:
    """10维状态空间计算器"""
    
    def __init__(self, mu: float = 5000, delta: float = 200, 
                 max_capacity: int = 48):
        """
        Args:
            mu: 等待时间归一化参数
            delta: 发车次数差异归一化参数
            max_capacity: 最大载客量 (32座位 * 1.5站立系数)
        """
        self.mu = mu
        self.delta = delta
        self.max_capacity = max_capacity
        
    def calculate_state(self, env: BidirectionalBusEnvironment, 
                       minute: int) -> np.ndarray:
        """
        计算10维状态向量
        
        Returns:
            [a1, a2, x1, x2, x3, x4, y1, y2, y3, y4]
        """
        # 时间状态 (2维)
        a1, a2 = self._calculate_time_state(minute)
        
        # 上行状态 (4维)
        x1 = self._calculate_load_rate(env.upward, minute)
        x2 = self._calculate_normalized_waiting_time(env.upward, minute)
        x3 = self._calculate_capacity_utilization(env.upward, minute)
        x4 = self._calculate_departure_diff(env, minute, 'upward')
        
        # 下行状态 (4维)
        y1 = self._calculate_load_rate(env.downward, minute)
        y2 = self._calculate_normalized_waiting_time(env.downward, minute)
        y3 = self._calculate_capacity_utilization(env.downward, minute)
        y4 = self._calculate_departure_diff(env, minute, 'downward')
        
        return np.array([a1, a2, x1, x2, x3, x4, y1, y2, y3, y4], 
                       dtype=np.float32)
```

#### 状态计算公式实现

**时间状态**:
```python
def _calculate_time_state(self, minute: int) -> tuple:
    """计算时间状态 a1, a2"""
    hour = minute // 60
    min_in_hour = minute % 60
    a1 = hour / 24.0  # 小时归一化
    a2 = min_in_hour / 60.0  # 分钟归一化
    return a1, a2
```

**满载率** (公式2.7/2.11):
```python
def _calculate_load_rate(self, direction_env: DirectionEnvironment, 
                        minute: int) -> float:
    """计算满载率 x1/y1 = C_max^m / C_max"""
    max_section_flow = direction_env.get_max_section_flow(minute)
    return min(max_section_flow / self.max_capacity, 1.0)
```

**归一化等待时间** (公式2.8/2.12):
```python
def _calculate_normalized_waiting_time(self, 
                                      direction_env: DirectionEnvironment,
                                      minute: int) -> float:
    """计算归一化等待时间 x2/y2 = W_m / μ"""
    total_waiting_time = direction_env.get_total_waiting_time(minute)
    return total_waiting_time / self.mu
```

**客运容量利用率** (公式2.9/2.13):
```python
def _calculate_capacity_utilization(self, 
                                   direction_env: DirectionEnvironment,
                                   minute: int) -> float:
    """计算客运容量利用率 x3/y3 = o_m / e_m"""
    consumed = direction_env.get_consumed_capacity(minute)
    provided = direction_env.get_provided_capacity(minute)
    return consumed / provided if provided > 0 else 0.0
```

**发车次数差异** (公式2.10/2.14):
```python
def _calculate_departure_diff(self, env: BidirectionalBusEnvironment,
                             minute: int, direction: str) -> float:
    """计算发车次数差异 x4/y4 = (c_up - c_down) / δ"""
    c_up = len(env.upward.departure_times)
    c_down = len(env.downward.departure_times)
    diff = (c_up - c_down) / self.delta
    return diff if direction == 'upward' else -diff
```



### 4. 奖励计算器 (RewardCalculator)

#### 类设计

```python
class RewardCalculator:
    """双向奖励函数计算器"""
    
    def __init__(self, mu: float = 5000, delta: float = 200,
                 beta: float = 0.2, zeta: float = 0.002,
                 omega_up: float = 1/1000, omega_down: float = 1/1000):
        """
        Args:
            mu: 等待时间归一化参数
            delta: 发车次数差异归一化参数
            beta: 滞留乘客惩罚权重
            zeta: 发车次数平衡权重
            omega_up: 上行等待时间惩罚参数
            omega_down: 下行等待时间惩罚参数
        """
        self.mu = mu
        self.delta = delta
        self.beta = beta
        self.zeta = zeta
        self.omega_up = omega_up
        self.omega_down = omega_down
        
    def calculate_reward(self, env: BidirectionalBusEnvironment,
                        action: tuple, minute: int) -> float:
        """
        计算双向总奖励 r_m = r_up + r_down
        
        Args:
            env: 仿真环境
            action: (a_up, a_down) 发车决策
            minute: 当前时刻
            
        Returns:
            总奖励值
        """
        r_up = self._calculate_direction_reward(
            env.upward, env.downward, action[0], minute, 'upward'
        )
        r_down = self._calculate_direction_reward(
            env.downward, env.upward, action[1], minute, 'downward'
        )
        return r_up + r_down
```

#### 奖励函数实现 (公式2.16和2.17)

```python
def _calculate_direction_reward(self, current_dir: DirectionEnvironment,
                               other_dir: DirectionEnvironment,
                               action: int, minute: int,
                               direction: str) -> float:
    """
    计算单方向奖励
    
    公式2.16 (上行):
    - 不发车: r = (1 - o/e) - (ω×W) - (β×d) + ζ(c_up - c_down)
    - 发车:   r = (o/e) - (β×d) - ζ(c_up - c_down)
    
    公式2.17 (下行):
    - 不发车: r = (1 - o/e) - (ω×W) - (β×d) - ζ(c_up - c_down)
    - 发车:   r = (o/e) - (β×d) + ζ(c_up - c_down)
    """
    # 获取基础参数
    o_m = current_dir.get_consumed_capacity(minute)
    e_m = current_dir.get_provided_capacity(minute)
    capacity_util = o_m / e_m if e_m > 0 else 0
    
    W_m = current_dir.get_total_waiting_time(minute)
    d_m = current_dir.stranded_passengers
    
    # 发车次数差异
    c_current = len(current_dir.departure_times)
    c_other = len(other_dir.departure_times)
    count_diff = c_current - c_other
    
    # 选择对应的ω参数
    omega = self.omega_up if direction == 'upward' else self.omega_down
    
    if action == 0:  # 不发车
        reward = (1 - capacity_util) - (omega * W_m) - (self.beta * d_m)
        # 上行: +ζ×diff, 下行: -ζ×diff
        if direction == 'upward':
            reward += self.zeta * count_diff
        else:
            reward -= self.zeta * count_diff
    else:  # 发车
        reward = capacity_util - (self.beta * d_m)
        # 上行: -ζ×diff, 下行: +ζ×diff
        if direction == 'upward':
            reward -= self.zeta * count_diff
        else:
            reward += self.zeta * count_diff
            
    return reward
```

### 5. DQN网络架构

#### 网络结构设计 (表2-2规格)

```python
class DQNNetwork(nn.Module):
    """DQN神经网络 - 12层隐藏层，每层500神经元"""
    
    def __init__(self, state_dim: int = 10, action_dim: int = 4,
                 hidden_size: int = 500, num_hidden_layers: int = 12):
        """
        Args:
            state_dim: 状态维度 (10维)
            action_dim: 动作维度 (4种动作组合)
            hidden_size: 隐藏层神经元数 (500)
            num_hidden_layers: 隐藏层数量 (12)
        """
        super(DQNNetwork, self).__init__()
        
        # 输入层
        self.input_layer = nn.Linear(state_dim, hidden_size)
        
        # 12个隐藏层
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) 
            for _ in range(num_hidden_layers)
        ])
        
        # 输出层
        self.output_layer = nn.Linear(hidden_size, action_dim)
        
        # 权重初始化 - 正态分布
        self._initialize_weights()
        
    def _initialize_weights(self):
        """使用正态分布初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                nn.init.constant_(m.bias, 0.0)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 状态张量 [batch_size, 10]
            
        Returns:
            Q值 [batch_size, 4]
        """
        # 输入层 + ReLU
        x = F.relu(self.input_layer(x))
        
        # 12个隐藏层 + ReLU
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        
        # 输出层 (无激活函数)
        q_values = self.output_layer(x)
        
        return q_values
```

#### 网络架构说明

- **输入**: 10维状态向量 [a1, a2, x1, x2, x3, x4, y1, y2, y3, y4]
- **隐藏层**: 12层，每层500个神经元，使用ReLU激活
- **输出**: 4个Q值，对应4种动作组合 [(0,0), (0,1), (1,0), (1,1)]
- **参数量**: 约 10×500 + 11×500×500 + 500×4 = 2,757,000 参数

### 6. DQN智能体 (DQNAgent)

#### 类设计

```python
class DQNAgent:
    """DQN智能体 - 实现算法2.1"""
    
    def __init__(self, state_dim: int = 10, action_dim: int = 4,
                 learning_rate: float = 0.001, gamma: float = 0.4,
                 epsilon: float = 0.1, batch_size: int = 64,
                 replay_buffer_size: int = 3000,
                 update_frequency: int = 5,
                 target_update_frequency: int = 100,
                 device: str = 'cuda'):
        """
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            learning_rate: 学习率 (0.001)
            gamma: 折扣因子 (0.4)
            epsilon: ε-贪婪探索率 (0.1)
            batch_size: 批次大小 (64)
            replay_buffer_size: 经验池大小 (3000)
            update_frequency: 学习频率P (每5步学习一次)
            target_update_frequency: 目标网络更新频率O (每100次学习)
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() 
                                  else 'cpu')
        
        # 主网络和目标网络
        self.main_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.main_network.state_dict())
        
        # 优化器 - Adam
        self.optimizer = optim.Adam(self.main_network.parameters(), 
                                   lr=learning_rate)
        
        # 经验回放池
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        
        # 超参数
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.target_update_frequency = target_update_frequency
        
        # 计数器
        self.step_count = 0
        self.learn_count = 0
```

#### 核心方法实现

**动作选择 - ε-贪婪策略**:
```python
def select_action(self, state: np.ndarray, epsilon: float = None) -> int:
    """
    使用ε-贪婪策略选择动作
    
    Args:
        state: 当前状态 (10维)
        epsilon: 探索率，默认使用self.epsilon
        
    Returns:
        动作索引 (0-3)
    """
    if epsilon is None:
        epsilon = self.epsilon
        
    if random.random() < epsilon:
        # 随机探索
        return random.randint(0, 3)
    else:
        # 贪婪选择
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.main_network(state_tensor)
            return q_values.argmax().item()
```

**经验存储**:
```python
def store_experience(self, state: np.ndarray, action: int, 
                    reward: float, next_state: np.ndarray):
    """存储经验到回放池"""
    self.replay_buffer.push(state, action, reward, next_state)
    self.step_count += 1
```

**网络训练**:
```python
def train(self) -> float:
    """
    训练主网络
    
    Returns:
        损失值
    """
    # 检查是否需要训练
    if self.step_count % self.update_frequency != 0:
        return 0.0
        
    if len(self.replay_buffer) < self.batch_size:
        return 0.0
    
    # 采样批次
    batch = self.replay_buffer.sample(self.batch_size)
    states, actions, rewards, next_states = batch
    
    # 转换为张量
    states = torch.FloatTensor(states).to(self.device)
    actions = torch.LongTensor(actions).to(self.device)
    rewards = torch.FloatTensor(rewards).to(self.device)
    next_states = torch.FloatTensor(next_states).to(self.device)
    
    # 计算当前Q值
    current_q_values = self.main_network(states).gather(1, actions.unsqueeze(1))
    
    # 计算目标Q值: y = r + γ * max_a' Q(s', a'; θ^-)
    with torch.no_grad():
        next_q_values = self.target_network(next_states).max(1)[0]
        target_q_values = rewards + self.gamma * next_q_values
    
    # 计算损失 - MSE
    loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
    
    # 反向传播和优化
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    
    self.learn_count += 1
    
    # 更新目标网络
    if self.learn_count % self.target_update_frequency == 0:
        self.update_target_network()
    
    return loss.item()
```

**目标网络更新**:
```python
def update_target_network(self):
    """将主网络参数复制到目标网络"""
    self.target_network.load_state_dict(self.main_network.state_dict())
```



### 7. 经验回放池 (ReplayBuffer)

#### 类设计

```python
from collections import deque
import random

class ReplayBuffer:
    """经验回放池 - 容量3000"""
    
    def __init__(self, capacity: int = 3000):
        """
        Args:
            capacity: 经验池最大容量
        """
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state: np.ndarray, action: int, 
            reward: float, next_state: np.ndarray):
        """
        存储经验 (s, a, r, s')
        
        当缓冲区满时，自动删除最旧的经验
        """
        self.buffer.append((state, action, reward, next_state))
        
    def sample(self, batch_size: int) -> tuple:
        """
        随机采样批次
        
        Args:
            batch_size: 批次大小 (64)
            
        Returns:
            (states, actions, rewards, next_states)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        
        return (np.array(states), np.array(actions), 
                np.array(rewards), np.array(next_states))
    
    def __len__(self):
        return len(self.buffer)
```

### 8. 发车约束处理

#### 约束检查器设计

```python
class DepartureConstraintChecker:
    """发车间隔约束检查器"""
    
    def __init__(self, T_min: int = 3, T_max: int = 15):
        """
        Args:
            T_min: 最小发车间隔(分钟)
            T_max: 最大发车间隔(分钟)
        """
        self.T_min = T_min
        self.T_max = T_max
        
    def apply_constraints(self, action: tuple, env: BidirectionalBusEnvironment,
                         minute: int) -> tuple:
        """
        应用发车间隔约束
        
        规则:
        - 如果距上次发车 < T_min，强制不发车
        - 如果距上次发车 > T_max，强制发车
        - 首班车和末班车强制发车
        
        Args:
            action: (a_up, a_down) 原始动作
            env: 仿真环境
            minute: 当前时刻
            
        Returns:
            约束后的动作
        """
        a_up, a_down = action
        
        # 检查上行约束
        a_up = self._check_direction_constraint(
            a_up, env.upward, minute
        )
        
        # 检查下行约束
        a_down = self._check_direction_constraint(
            a_down, env.downward, minute
        )
        
        return (a_up, a_down)
    
    def _check_direction_constraint(self, action: int,
                                   direction_env: DirectionEnvironment,
                                   minute: int) -> int:
        """检查单方向约束"""
        # 首班车判断
        if self._is_first_departure(direction_env, minute):
            return 1  # 强制发车
        
        # 末班车判断
        if self._is_last_departure(direction_env, minute):
            return 1  # 强制发车
        
        # 计算距上次发车的间隔
        interval = self._get_last_departure_interval(direction_env, minute)
        
        # 应用间隔约束
        if interval < self.T_min:
            return 0  # 强制不发车
        elif interval > self.T_max:
            return 1  # 强制发车
        else:
            return action  # 保持原动作
    
    def _get_last_departure_interval(self, direction_env: DirectionEnvironment,
                                    minute: int) -> int:
        """获取距上次发车的时间间隔"""
        if not direction_env.departure_times:
            return float('inf')
        return minute - direction_env.departure_times[-1]
    
    def _is_first_departure(self, direction_env: DirectionEnvironment,
                           minute: int) -> bool:
        """判断是否为首班车时刻"""
        start_time = direction_env.operating_hours[0]  # 如 6:00
        return minute == start_time and not direction_env.departure_times
    
    def _is_last_departure(self, direction_env: DirectionEnvironment,
                          minute: int) -> bool:
        """判断是否为末班车时刻"""
        end_time = direction_env.operating_hours[1]  # 如 21:00
        return minute >= end_time - 30  # 末班前30分钟
```

## 训练流程设计

### 训练主循环 (算法2.1实现)

```python
def train_drl_tsbc(agent: DQNAgent, 
                   env: BidirectionalBusEnvironment,
                   num_episodes: int = 50,
                   constraint_checker: DepartureConstraintChecker = None,
                   save_path: str = 'models/drl_tsbc.pth') -> dict:
    """
    DRL-TSBC训练主循环
    
    Args:
        agent: DQN智能体
        env: 双向仿真环境
        num_episodes: 训练轮数 (50)
        constraint_checker: 约束检查器
        save_path: 模型保存路径
        
    Returns:
        训练历史 (losses, rewards, etc.)
    """
    if constraint_checker is None:
        constraint_checker = DepartureConstraintChecker()
    
    history = {
        'episode_rewards': [],
        'episode_losses': [],
        'episode_departures': []
    }
    
    for episode in range(1, num_episodes + 1):
        # 重置环境
        state = env.reset()
        episode_reward = 0
        episode_loss = 0
        loss_count = 0
        
        # 模拟一天 (运营时间内的每一分钟)
        for minute in range(env.start_time, env.end_time + 1):
            # 选择动作 (ε-贪婪)
            action_idx = agent.select_action(state)
            action = index_to_action(action_idx)  # 0-3 -> (0,0)/(0,1)/(1,0)/(1,1)
            
            # 应用约束
            constrained_action = constraint_checker.apply_constraints(
                action, env, minute
            )
            
            # 执行动作
            next_state, reward, done, info = env.step(constrained_action)
            
            # 存储经验
            constrained_action_idx = action_to_index(constrained_action)
            agent.store_experience(state, constrained_action_idx, 
                                 reward, next_state)
            
            # 训练网络
            loss = agent.train()
            if loss > 0:
                episode_loss += loss
                loss_count += 1
            
            # 更新状态和奖励
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # 记录历史
        avg_loss = episode_loss / loss_count if loss_count > 0 else 0
        history['episode_rewards'].append(episode_reward)
        history['episode_losses'].append(avg_loss)
        history['episode_departures'].append({
            'upward': len(env.upward.departure_times),
            'downward': len(env.downward.departure_times)
        })
        
        # 打印进度
        print(f"Episode {episode}/{num_episodes} - "
              f"Reward: {episode_reward:.2f}, Loss: {avg_loss:.4f}, "
              f"Departures: {len(env.upward.departure_times)}↑ "
              f"{len(env.downward.departure_times)}↓")
        
        # 定期保存模型
        if episode % 10 == 0:
            agent.save(save_path)
    
    # 保存最终模型
    agent.save(save_path)
    
    return history

def index_to_action(idx: int) -> tuple:
    """动作索引转换为元组"""
    actions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    return actions[idx]

def action_to_index(action: tuple) -> int:
    """动作元组转换为索引"""
    actions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    return actions.index(action)
```

## 推理流程设计

### 推理主流程 (算法2.2实现)

```python
def inference_drl_tsbc(agent: DQNAgent,
                       env: BidirectionalBusEnvironment,
                       constraint_checker: DepartureConstraintChecker = None,
                       post_process: bool = True) -> dict:
    """
    DRL-TSBC推理流程 - 生成时刻表
    
    Args:
        agent: 训练好的DQN智能体
        env: 双向仿真环境
        constraint_checker: 约束检查器
        post_process: 是否进行后处理
        
    Returns:
        时刻表 {'upward': [...], 'downward': [...]}
    """
    if constraint_checker is None:
        constraint_checker = DepartureConstraintChecker()
    
    # 重置环境
    state = env.reset()
    
    # 模拟一天
    for minute in range(env.start_time, env.end_time + 1):
        # 使用主网络选择最优动作 (不使用ε-贪婪)
        action_idx = agent.select_action(state, epsilon=0.0)
        action = index_to_action(action_idx)
        
        # 应用约束
        constrained_action = constraint_checker.apply_constraints(
            action, env, minute
        )
        
        # 执行动作
        next_state, _, done, _ = env.step(constrained_action)
        
        state = next_state
        
        if done:
            break
    
    # 获取时刻表
    timetable = {
        'upward': env.upward.departure_times.copy(),
        'downward': env.downward.departure_times.copy()
    }
    
    # 后处理 - 确保发车次数相等
    if post_process:
        timetable = post_process_timetable(timetable, constraint_checker)
    
    return timetable
```

### 后处理算法

```python
def post_process_timetable(timetable: dict,
                          constraint_checker: DepartureConstraintChecker) -> dict:
    """
    后处理时刻表，确保上下行发车次数相等
    
    算法步骤:
    1. 比较上下行发车次数
    2. 如果不相等，删除发车次数较多方向的倒数第二次发车
    3. 以T_max为间隔向前调整发车时刻
    4. 确保所有发车间隔满足约束
    
    Args:
        timetable: 原始时刻表
        constraint_checker: 约束检查器
        
    Returns:
        调整后的时刻表
    """
    upward_count = len(timetable['upward'])
    downward_count = len(timetable['downward'])
    
    # 如果发车次数已经相等，直接返回
    if upward_count == downward_count:
        return timetable
    
    # 确定需要调整的方向
    if upward_count > downward_count:
        excess_direction = 'upward'
    else:
        excess_direction = 'downward'
    
    print(f"后处理: {excess_direction}方向发车次数较多，进行调整...")
    
    # 删除倒数第二次发车
    departure_list = timetable[excess_direction]
    if len(departure_list) >= 2:
        del departure_list[-2]
        print(f"删除倒数第二次发车: {departure_list[-2] if len(departure_list) >= 2 else 'N/A'}")
    
    # 向前调整发车时刻
    adjusted_list = adjust_departure_times(
        departure_list, 
        constraint_checker.T_max
    )
    timetable[excess_direction] = adjusted_list
    
    # 验证约束
    validate_timetable(timetable, constraint_checker)
    
    return timetable

def adjust_departure_times(departure_list: list, T_max: int) -> list:
    """
    以T_max为间隔向前调整发车时刻
    
    Args:
        departure_list: 发车时刻列表
        T_max: 最大发车间隔
        
    Returns:
        调整后的发车时刻列表
    """
    if len(departure_list) <= 1:
        return departure_list
    
    adjusted = [departure_list[0]]  # 保留首班车
    
    for i in range(1, len(departure_list)):
        prev_time = adjusted[-1]
        current_time = departure_list[i]
        interval = current_time - prev_time
        
        # 如果间隔超过T_max，向前调整
        if interval > T_max:
            adjusted_time = prev_time + T_max
            adjusted.append(adjusted_time)
        else:
            adjusted.append(current_time)
    
    return adjusted

def validate_timetable(timetable: dict,
                      constraint_checker: DepartureConstraintChecker):
    """验证时刻表是否满足所有约束"""
    for direction in ['upward', 'downward']:
        times = timetable[direction]
        for i in range(1, len(times)):
            interval = times[i] - times[i-1]
            assert constraint_checker.T_min <= interval <= constraint_checker.T_max, \
                f"{direction}方向第{i}次发车间隔{interval}不满足约束"
    
    # 验证发车次数相等
    assert len(timetable['upward']) == len(timetable['downward']), \
        f"发车次数不相等: 上行{len(timetable['upward'])}, 下行{len(timetable['downward'])}"
    
    print("✓ 时刻表验证通过")
```



## 数据处理设计

### 数据加载器 (DataLoader)

```python
class BusDataLoader:
    """公交数据加载器"""
    
    def __init__(self, data_dir: str = 'test_data'):
        """
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = data_dir
        
    def load_passenger_data(self, line: str, direction: str) -> pd.DataFrame:
        """
        加载乘客数据
        
        Args:
            line: 线路编号 ('208' 或 '211')
            direction: 方向 ('upward' 或 'downward')
            
        Returns:
            DataFrame包含: [passenger_id, arrival_time, board_station, 
                           alight_station, board_time]
        """
        file_path = os.path.join(self.data_dir, f'{line}_{direction}_passengers.csv')
        df = pd.read_csv(file_path)
        
        # 数据预处理
        df['arrival_time'] = pd.to_datetime(df['arrival_time'])
        df['board_time'] = pd.to_datetime(df['board_time'])
        
        return df
    
    def load_traffic_data(self, line: str) -> pd.DataFrame:
        """
        加载交通状况数据
        
        Args:
            line: 线路编号
            
        Returns:
            DataFrame包含: [from_station, to_station, travel_time, time_period]
        """
        file_path = os.path.join(self.data_dir, f'{line}_traffic.csv')
        df = pd.read_csv(file_path)
        return df
    
    def load_line_config(self, line: str) -> dict:
        """
        加载线路配置
        
        Args:
            line: 线路编号
            
        Returns:
            配置字典
        """
        config_path = os.path.join('config', f'line_{line}.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
```

### 配置管理 (ConfigManager)

```python
class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str = 'config/drl_tsbc_config.yaml'):
        """
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
    
    def get_model_config(self) -> dict:
        """获取模型配置"""
        return self.config['model']
    
    def get_training_config(self) -> dict:
        """获取训练配置"""
        return self.config['training']
    
    def get_reward_config(self) -> dict:
        """获取奖励函数配置"""
        return self.config['reward']
    
    def get_constraint_config(self) -> dict:
        """获取约束配置"""
        return self.config['constraints']
```

## 评估模块设计

### 性能指标计算

```python
class PerformanceEvaluator:
    """性能评估器"""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate_timetable(self, timetable: dict, 
                          env: BidirectionalBusEnvironment) -> dict:
        """
        评估时刻表性能
        
        Args:
            timetable: 时刻表
            env: 仿真环境
            
        Returns:
            性能指标字典
        """
        metrics = {}
        
        # 1. 发车次数
        metrics['upward_departures'] = len(timetable['upward'])
        metrics['downward_departures'] = len(timetable['downward'])
        metrics['total_departures'] = (metrics['upward_departures'] + 
                                      metrics['downward_departures'])
        
        # 2. 乘客平均等待时间 (AWT)
        metrics['upward_awt'] = self.calculate_average_waiting_time(
            env.upward
        )
        metrics['downward_awt'] = self.calculate_average_waiting_time(
            env.downward
        )
        metrics['overall_awt'] = (metrics['upward_awt'] + 
                                 metrics['downward_awt']) / 2
        
        # 3. 滞留乘客数量
        metrics['upward_stranded'] = env.upward.stranded_passengers
        metrics['downward_stranded'] = env.downward.stranded_passengers
        metrics['total_stranded'] = (metrics['upward_stranded'] + 
                                    metrics['downward_stranded'])
        
        # 4. 客运容量利用率
        metrics['upward_capacity_util'] = self.calculate_capacity_utilization(
            env.upward
        )
        metrics['downward_capacity_util'] = self.calculate_capacity_utilization(
            env.downward
        )
        
        return metrics
    
    def calculate_average_waiting_time(self, 
                                      direction_env: DirectionEnvironment) -> float:
        """
        计算乘客平均等待时间
        
        AWT = Σ(board_time - arrival_time) / total_passengers
        """
        total_waiting_time = 0
        total_passengers = 0
        
        for station in direction_env.stations:
            for passenger in station.served_passengers:
                waiting_time = passenger.board_time - passenger.arrival_time
                total_waiting_time += waiting_time
                total_passengers += 1
        
        return total_waiting_time / total_passengers if total_passengers > 0 else 0
    
    def calculate_capacity_utilization(self,
                                      direction_env: DirectionEnvironment) -> float:
        """计算平均客运容量利用率"""
        total_util = 0
        count = 0
        
        for bus in direction_env.buses:
            util = bus.total_passengers_served / bus.total_capacity_provided
            total_util += util
            count += 1
        
        return total_util / count if count > 0 else 0
    
    def compare_with_baseline(self, drl_metrics: dict, 
                             baseline_metrics: dict) -> dict:
        """
        与基线方案对比
        
        Args:
            drl_metrics: DRL-TSBC指标
            baseline_metrics: 人工方案指标
            
        Returns:
            改进百分比
        """
        improvements = {}
        
        # AWT改进
        awt_improvement = ((baseline_metrics['overall_awt'] - 
                          drl_metrics['overall_awt']) / 
                          baseline_metrics['overall_awt'] * 100)
        improvements['awt_improvement'] = awt_improvement
        
        # 滞留乘客减少
        stranded_reduction = (baseline_metrics['total_stranded'] - 
                             drl_metrics['total_stranded'])
        improvements['stranded_reduction'] = stranded_reduction
        
        # 发车次数变化
        departure_change = (drl_metrics['total_departures'] - 
                           baseline_metrics['total_departures'])
        improvements['departure_change'] = departure_change
        
        return improvements
```

### 可视化模块

```python
class ResultVisualizer:
    """结果可视化器"""
    
    def __init__(self, save_dir: str = 'results/figures'):
        """
        Args:
            save_dir: 图表保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_training_history(self, history: dict, line: str):
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # 奖励曲线
        axes[0].plot(history['episode_rewards'])
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Total Reward')
        axes[0].set_title(f'Training Rewards - Line {line}')
        axes[0].grid(True)
        
        # 损失曲线
        axes[1].plot(history['episode_losses'])
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Loss')
        axes[1].set_title(f'Training Loss - Line {line}')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'training_history_{line}.png'))
        plt.close()
    
    def plot_capacity_comparison(self, drl_capacity: list, 
                                demand: list, line: str):
        """
        绘制总客运容量与真实需求对比图 (对应论文图2-3, 2-4)
        
        Args:
            drl_capacity: DRL-TSBC提供的客运容量
            demand: 真实客运需求
            line: 线路编号
        """
        plt.figure(figsize=(12, 6))
        
        time_points = range(len(drl_capacity))
        plt.plot(time_points, drl_capacity, label='DRL-TSBC Capacity', 
                linewidth=2)
        plt.plot(time_points, demand, label='Real Demand', 
                linewidth=2, linestyle='--')
        
        plt.xlabel('Time (minutes)')
        plt.ylabel('Capacity / Demand')
        plt.title(f'Capacity vs Demand - Line {line}')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(self.save_dir, 
                                f'capacity_comparison_{line}.png'))
        plt.close()
    
    def plot_performance_comparison(self, metrics_dict: dict, line: str):
        """
        绘制性能对比柱状图 (对应论文表2-3)
        
        Args:
            metrics_dict: {'人工方案': metrics, 'DRL-TO': metrics, 
                          'DRL-TSBC': metrics}
            line: 线路编号
        """
        methods = list(metrics_dict.keys())
        awt_values = [m['overall_awt'] for m in metrics_dict.values()]
        stranded_values = [m['total_stranded'] for m in metrics_dict.values()]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # AWT对比
        axes[0].bar(methods, awt_values)
        axes[0].set_ylabel('Average Waiting Time (min)')
        axes[0].set_title(f'AWT Comparison - Line {line}')
        axes[0].grid(axis='y')
        
        # 滞留乘客对比
        axes[1].bar(methods, stranded_values)
        axes[1].set_ylabel('Stranded Passengers')
        axes[1].set_title(f'Stranded Passengers - Line {line}')
        axes[1].grid(axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 
                                f'performance_comparison_{line}.png'))
        plt.close()
    
    def plot_omega_sensitivity(self, omega_results: dict, line: str):
        """
        绘制ω参数敏感性分析 (对应论文图2-5)
        
        Args:
            omega_results: {omega_value: metrics}
            line: 线路编号
        """
        omega_values = sorted(omega_results.keys())
        awt_values = [omega_results[w]['overall_awt'] for w in omega_values]
        
        plt.figure(figsize=(10, 6))
        plt.plot(omega_values, awt_values, marker='o', linewidth=2)
        plt.xlabel('ω (Waiting Time Penalty)')
        plt.ylabel('Average Waiting Time (min)')
        plt.title(f'ω Sensitivity Analysis - Line {line}')
        plt.grid(True)
        
        plt.savefig(os.path.join(self.save_dir, 
                                f'omega_sensitivity_{line}.png'))
        plt.close()
```

## 项目文件结构

```
drl_tsbc/
├── config/                          # 配置文件
│   ├── drl_tsbc_config.yaml        # 主配置文件
│   ├── line_208.yaml               # 208线配置
│   └── line_211.yaml               # 211线配置
│
├── core/                            # 核心模块
│   ├── __init__.py
│   ├── environment.py              # 双向仿真环境
│   ├── direction_env.py            # 单方向环境
│   ├── time_controller.py          # 时间控制器
│   ├── state_calculator.py         # 状态计算器
│   ├── reward_calculator.py        # 奖励计算器
│   ├── network.py                  # DQN网络
│   ├── agent.py                    # DQN智能体
│   ├── replay_buffer.py            # 经验回放池
│   ├── constraint_checker.py       # 约束检查器
│   └── post_processor.py           # 后处理器
│
├── utils/                           # 工具模块
│   ├── __init__.py
│   ├── data_loader.py              # 数据加载
│   ├── config_manager.py           # 配置管理
│   ├── logger.py                   # 日志系统
│   ├── evaluator.py                # 性能评估
│   └── visualizer.py               # 结果可视化
│
├── test_data/                       # 测试数据
│   ├── 208_upward_passengers.csv
│   ├── 208_downward_passengers.csv
│   ├── 208_traffic.csv
│   ├── 211_upward_passengers.csv
│   ├── 211_downward_passengers.csv
│   └── 211_traffic.csv
│
├── models/                          # 模型保存
│   ├── drl_tsbc_208.pth
│   └── drl_tsbc_211.pth
│
├── results/                         # 实验结果
│   ├── figures/                    # 图表
│   ├── timetables/                 # 生成的时刻表
│   └── reports/                    # 实验报告
│
├── train.py                         # 训练脚本
├── inference.py                     # 推理脚本
├── evaluate.py                      # 评估脚本
├── run_experiment.py                # 完整实验脚本
├── requirements.txt                 # 依赖包
└── README.md                        # 使用说明
```

## GPU加速设计

### GPU管理器

```python
class GPUManager:
    """GPU加速管理器"""
    
    def __init__(self):
        self.device = self._setup_device()
        self._print_device_info()
    
    def _setup_device(self) -> torch.device:
        """设置计算设备"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            print("警告: CUDA不可用，使用CPU")
            return torch.device('cpu')
    
    def _print_device_info(self):
        """打印设备信息"""
        if self.device.type == 'cuda':
            print(f"使用GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("使用CPU进行计算")
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """将张量移动到设备"""
        return tensor.to(self.device)
    
    def clear_cache(self):
        """清理GPU缓存"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
```

### 性能优化策略

1. **批量处理**: 使用批次大小64进行训练，充分利用GPU并行计算能力
2. **混合精度**: 可选使用FP16混合精度训练，提升速度并减少显存占用
3. **数据预加载**: 提前加载和预处理数据，减少I/O等待时间
4. **经验池优化**: 使用deque实现O(1)的插入和删除操作
5. **目标网络更新**: 每100次学习才更新目标网络，减少计算开销



## 测试策略

### 单元测试

#### 1. 状态计算器测试

```python
class TestStateCalculator(unittest.TestCase):
    """状态计算器单元测试"""
    
    def setUp(self):
        self.calculator = StateCalculator(mu=5000, delta=200, max_capacity=48)
        self.env = create_mock_environment()
    
    def test_state_dimension(self):
        """测试状态维度为10"""
        state = self.calculator.calculate_state(self.env, 420)
        self.assertEqual(len(state), 10)
    
    def test_state_normalization(self):
        """测试所有状态值在[0,1]范围内"""
        state = self.calculator.calculate_state(self.env, 420)
        self.assertTrue(all(0 <= s <= 1 for s in state))
    
    def test_time_state(self):
        """测试时间状态计算"""
        # 7:30 = 450分钟
        state = self.calculator.calculate_state(self.env, 450)
        a1, a2 = state[0], state[1]
        self.assertAlmostEqual(a1, 7/24, places=4)  # 7小时
        self.assertAlmostEqual(a2, 30/60, places=4)  # 30分钟
    
    def test_departure_diff_symmetry(self):
        """测试发车次数差异的对称性"""
        state = self.calculator.calculate_state(self.env, 420)
        x4, y4 = state[5], state[9]
        self.assertAlmostEqual(x4, -y4, places=4)
```

#### 2. 奖励函数测试

```python
class TestRewardCalculator(unittest.TestCase):
    """奖励函数单元测试"""
    
    def setUp(self):
        self.calculator = RewardCalculator(
            mu=5000, delta=200, beta=0.2, zeta=0.002
        )
        self.env = create_mock_environment()
    
    def test_reward_structure(self):
        """测试奖励函数结构"""
        # 测试4种动作组合
        actions = [(0,0), (0,1), (1,0), (1,1)]
        for action in actions:
            reward = self.calculator.calculate_reward(self.env, action, 420)
            self.assertIsInstance(reward, float)
    
    def test_departure_balance_incentive(self):
        """测试发车次数平衡激励"""
        # 当上行发车次数多时，上行发车应得到负奖励
        self.env.upward.departure_times = [360, 375, 390]  # 3次
        self.env.downward.departure_times = [360, 375]     # 2次
        
        r_dispatch = self.calculator.calculate_reward(self.env, (1,0), 420)
        r_no_dispatch = self.calculator.calculate_reward(self.env, (0,0), 420)
        
        # 不发车应该得到更高奖励
        self.assertGreater(r_no_dispatch, r_dispatch)
```

#### 3. DQN网络测试

```python
class TestDQNNetwork(unittest.TestCase):
    """DQN网络单元测试"""
    
    def setUp(self):
        self.network = DQNNetwork(state_dim=10, action_dim=4)
    
    def test_network_structure(self):
        """测试网络结构"""
        # 检查层数
        self.assertEqual(len(self.network.hidden_layers), 12)
        
        # 检查每层神经元数
        for layer in self.network.hidden_layers:
            self.assertEqual(layer.out_features, 500)
    
    def test_forward_pass(self):
        """测试前向传播"""
        batch_size = 64
        state = torch.randn(batch_size, 10)
        q_values = self.network(state)
        
        # 检查输出形状
        self.assertEqual(q_values.shape, (batch_size, 4))
    
    def test_weight_initialization(self):
        """测试权重初始化"""
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                # 检查权重不全为零
                self.assertFalse(torch.all(m.weight == 0))
```

### 集成测试

#### 1. 训练流程测试

```python
class TestTrainingPipeline(unittest.TestCase):
    """训练流程集成测试"""
    
    def test_full_training_episode(self):
        """测试完整训练episode"""
        agent = DQNAgent(device='cpu')
        env = BidirectionalBusEnvironment(load_test_config())
        
        # 运行1个episode
        history = train_drl_tsbc(agent, env, num_episodes=1)
        
        # 验证历史记录
        self.assertEqual(len(history['episode_rewards']), 1)
        self.assertEqual(len(history['episode_losses']), 1)
    
    def test_constraint_enforcement(self):
        """测试约束执行"""
        agent = DQNAgent(device='cpu')
        env = BidirectionalBusEnvironment(load_test_config())
        checker = DepartureConstraintChecker(T_min=3, T_max=15)
        
        train_drl_tsbc(agent, env, num_episodes=1, 
                      constraint_checker=checker)
        
        # 验证所有发车间隔满足约束
        for times in [env.upward.departure_times, env.downward.departure_times]:
            for i in range(1, len(times)):
                interval = times[i] - times[i-1]
                self.assertGreaterEqual(interval, 3)
                self.assertLessEqual(interval, 15)
```

#### 2. 推理流程测试

```python
class TestInferencePipeline(unittest.TestCase):
    """推理流程集成测试"""
    
    def test_timetable_generation(self):
        """测试时刻表生成"""
        agent = DQNAgent(device='cpu')
        env = BidirectionalBusEnvironment(load_test_config())
        
        # 生成时刻表
        timetable = inference_drl_tsbc(agent, env)
        
        # 验证时刻表结构
        self.assertIn('upward', timetable)
        self.assertIn('downward', timetable)
        self.assertIsInstance(timetable['upward'], list)
        self.assertIsInstance(timetable['downward'], list)
    
    def test_departure_count_equality(self):
        """测试发车次数相等约束"""
        agent = DQNAgent(device='cpu')
        env = BidirectionalBusEnvironment(load_test_config())
        
        timetable = inference_drl_tsbc(agent, env, post_process=True)
        
        # 验证发车次数相等
        self.assertEqual(len(timetable['upward']), 
                        len(timetable['downward']))
```

### 性能测试

```python
class TestPerformance(unittest.TestCase):
    """性能测试"""
    
    def test_training_speed(self):
        """测试训练速度"""
        agent = DQNAgent(device='cuda' if torch.cuda.is_available() else 'cpu')
        env = BidirectionalBusEnvironment(load_test_config())
        
        start_time = time.time()
        train_drl_tsbc(agent, env, num_episodes=1)
        duration = time.time() - start_time
        
        # 单个episode应在5分钟内完成
        self.assertLess(duration, 300)
    
    def test_memory_usage(self):
        """测试内存使用"""
        import psutil
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss / 1024**2  # MB
        
        agent = DQNAgent(device='cpu')
        env = BidirectionalBusEnvironment(load_test_config())
        train_drl_tsbc(agent, env, num_episodes=5)
        
        final_memory = process.memory_info().rss / 1024**2  # MB
        memory_increase = final_memory - initial_memory
        
        # 内存增长应小于1GB
        self.assertLess(memory_increase, 1024)
```

## 关键设计决策

### 1. 为什么使用组合模式管理双向环境？

**决策**: 使用两个独立的DirectionEnvironment实例

**理由**:
- 代码复用: 避免重复实现上行和下行的相同逻辑
- 清晰分离: 每个方向的状态独立管理，便于调试
- 扩展性: 未来可以轻松支持多条线路或多个方向

### 2. 为什么状态空间设计为10维？

**决策**: 时间(2维) + 上行(4维) + 下行(4维)

**理由**:
- 论文规格: 严格遵循谢嘉昊论文的状态空间定义
- 信息完整: 包含时间、客流、车辆利用率和发车平衡信息
- 维度适中: 10维既包含足够信息，又不会导致维度灾难

### 3. 为什么使用12层隐藏层？

**决策**: 12层×500神经元的深度网络

**理由**:
- 论文规格: 表2-2明确规定的网络结构
- 表达能力: 深度网络能够学习复杂的非线性关系
- 实验验证: 论文实验证明该结构有效

### 4. 为什么折扣因子γ=0.4？

**决策**: 使用较小的折扣因子

**理由**:
- 短期优化: 公交排班更关注短期效果（当前时段的等待时间）
- 论文设置: 表2-2规定的参数值
- 实验调优: 该值在实验中表现最佳

### 5. 为什么经验池容量为3000？

**决策**: 相对较小的经验池

**理由**:
- 内存效率: 避免占用过多内存
- 数据新鲜度: 保持经验的时效性
- 论文规格: 表2-2规定的容量

### 6. 为什么需要后处理算法？

**决策**: 推理后进行时刻表调整

**理由**:
- 硬约束: 上下行发车次数必须严格相等
- DQN局限: 神经网络难以保证硬约束
- 实用性: 后处理确保生成的时刻表可直接使用

### 7. 为什么使用ε-贪婪而不是其他探索策略？

**决策**: 固定ε=0.1的ε-贪婪策略

**理由**:
- 简单有效: ε-贪婪是最经典的探索策略
- 论文规格: 表2-2规定使用ε=0.1
- 平衡探索: 10%的探索率在探索和利用之间取得平衡

## 错误处理策略

### 1. 数据加载错误

```python
try:
    passenger_data = data_loader.load_passenger_data('208', 'upward')
except FileNotFoundError:
    logger.error("乘客数据文件不存在")
    raise
except pd.errors.EmptyDataError:
    logger.error("乘客数据文件为空")
    raise
```

### 2. GPU内存不足

```python
try:
    agent = DQNAgent(device='cuda')
except RuntimeError as e:
    if "out of memory" in str(e):
        logger.warning("GPU内存不足，切换到CPU")
        agent = DQNAgent(device='cpu')
    else:
        raise
```

### 3. 训练不收敛

```python
if episode > 20 and np.mean(history['episode_rewards'][-5:]) < threshold:
    logger.warning("训练可能不收敛，建议检查超参数")
```

### 4. 约束违反

```python
def validate_timetable(timetable, constraint_checker):
    """验证时刻表约束"""
    try:
        # 检查发车间隔
        for direction in ['upward', 'downward']:
            times = timetable[direction]
            for i in range(1, len(times)):
                interval = times[i] - times[i-1]
                assert constraint_checker.T_min <= interval <= constraint_checker.T_max
        
        # 检查发车次数
        assert len(timetable['upward']) == len(timetable['downward'])
        
    except AssertionError as e:
        logger.error(f"时刻表约束验证失败: {e}")
        raise
```

## 日志系统设计

```python
import logging

class DRLTSBCLogger:
    """DRL-TSBC日志系统"""
    
    def __init__(self, log_file: str = 'logs/drl_tsbc.log'):
        """
        Args:
            log_file: 日志文件路径
        """
        self.logger = logging.getLogger('DRL-TSBC')
        self.logger.setLevel(logging.INFO)
        
        # 文件处理器
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)
        
        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 格式化
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def log_training_start(self, config: dict):
        """记录训练开始"""
        self.logger.info("="*50)
        self.logger.info("开始训练 DRL-TSBC")
        self.logger.info(f"配置: {config}")
        self.logger.info("="*50)
    
    def log_episode(self, episode: int, reward: float, loss: float, 
                   departures: dict):
        """记录episode信息"""
        self.logger.info(
            f"Episode {episode} - "
            f"Reward: {reward:.2f}, Loss: {loss:.4f}, "
            f"Departures: {departures['upward']}↑ {departures['downward']}↓"
        )
    
    def log_evaluation(self, metrics: dict):
        """记录评估结果"""
        self.logger.info("="*50)
        self.logger.info("评估结果:")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("="*50)
```

## 总结

本设计文档详细描述了DRL-TSBC算法的完整实现方案，包括：

1. **系统架构**: 清晰的三层架构（训练/推理/评估 - 仿真环境 - 数据层）
2. **核心组件**: 双向环境、状态计算、奖励函数、DQN网络等关键模块
3. **算法实现**: 严格遵循论文公式和算法流程
4. **约束处理**: 完善的发车间隔约束和后处理机制
5. **性能优化**: GPU加速、批量处理等优化策略
6. **测试策略**: 单元测试、集成测试和性能测试
7. **错误处理**: 完善的异常处理和日志系统

该设计确保了算法实现的正确性、可维护性和可扩展性，为后续的代码实现提供了清晰的指导。

