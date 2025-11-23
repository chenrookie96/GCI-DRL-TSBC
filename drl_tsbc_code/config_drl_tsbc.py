"""
DRL-TSBC 配置文件
改动说明：
1. 集中管理所有可配置参数
2. 对应论文表2-1和表2-2的参数设置
"""

# ==================== 线路配置 ====================
# 线路号
BUSLINE = 208

# 方向定义
DIRECTION_UPWARD = 0
DIRECTION_DOWNWARD = 1

# 站点数量（根据论文表2-1）
# 208线：上行26站，下行24站
# 数据文件中：s0-s25（26列）对应26个站点，s0-s23（24列）对应24个站点
STATION_NUM = {
    208: {'upward': 26, 'downward': 24},
    211: {'upward': 17, 'downward': 11}
}

# 营运时间（根据论文表2-1）
OPERATING_TIME = {
    208: {'first': '06:00', 'last': '21:00'},
    211: {'first': '06:00', 'last': '22:00'}
}

# ==================== 车辆参数 ====================
# 座位数
SEAT_COUNT = 32

# 站立系数α（论文中α=1.5）
ALPHA = 1.5

# 最大载客量（座位数 × α）
MAX_CAPACITY = int(SEAT_COUNT * ALPHA)  # 48

# ==================== 发车间隔约束 ====================
# 最小发车间隔Tmin（分钟）
MIN_INTERVAL = 5

# 最大发车间隔Tmax（分钟）
MAX_INTERVAL = 22

# ==================== 均匀排班配置（可选功能） ====================
# 均匀排班控制标志
# 0: 不使用均匀排班
# 1: 特定区间等间隔排班
# 2: 方差最小化均匀调整（全局）
# 3: 方差最小化均匀调整（特定区间）
AVG_FLAG = 1

# 均匀排班时间段
UNIFORM_START_TIME = "10:00"
UNIFORM_END_TIME = "16:00"

# 理想发车间隔（分钟）
IDEAL_INTERVAL = 12

# ==================== DQN网络参数（论文表2-2） ====================
# 学习率
LEARNING_RATE = 0.01

# 权重初始化方法
WEIGHT_INIT = 'normal'  # 正态分布初始化

# 激活函数
ACTIVATION = 'ReLU'

# 隐层数量
HIDDEN_LAYERS = 12

# 隐层神经元数量
HIDDEN_SIZE = 500

# 批次大小B
BATCH_SIZE = 64

# 折扣系数γ
GAMMA = 0.4

# 经验池大小M
MEMORY_SIZE = 3000

# ε-贪婪策略参数
EPSILON = 0.1

# 最大模拟次数E（论文表2-2明确规定）
# 论文：E=50
# 单向示例：500（可能是为了更充分训练）
# 建议：先用50复现论文，效果不好再调整
MAX_EPISODES = 50  # 可选：50（论文）, 500（充分训练）

# 预热步数（经验池填满后才开始训练）
WARMUP_STEPS = 3000  # 等于MEMORY_SIZE

# 学习频率P（每P步学习一次）
LEARNING_FREQUENCY = 5

# 参数更新频率O（每O次学习更新目标网络）
UPDATE_FREQUENCY = 100

# ==================== 奖励函数参数 ====================
# 等待时间归一化参数μ
MU = 5000

# 发车次数差异归一化参数δ
DELTA = 200

# 滞留乘客惩罚权重β
BETA = 0.2

# 发车次数平衡权重ζ
ZETA = 0.002

# ==================== ω参数（论文中不同线路不同） ====================
# 等待时间惩罚参数ω = 1/omega_factor
OMEGA_FACTOR = {
    208: {'upward': 1000, 'downward': 1000},
    211: {'upward': 900, 'downward': 900}
}

# ==================== 数据路径配置 ====================
def get_data_paths(busline, direction):
    """
    获取指定线路和方向的数据路径
    
    参数：
    - busline: 线路号（208或211）
    - direction: 方向（0=上行，1=下行）
    
    返回：
    - dict: 包含各种数据路径的字典
    """
    data_dir = f"./test_data/{busline}"
    
    return {
        'passenger': f"{data_dir}/passenger_dataframe_direction{direction}.csv",
        'traffic': f"{data_dir}/traffic-{direction}.csv",
        'model': f"{data_dir}/drl_tsbc_model_{busline}.pth",
        'result': f"{data_dir}/drl_tsbc_result_{busline}.txt"
    }


def get_line_config(busline):
    """
    获取指定线路的完整配置
    
    参数：
    - busline: 线路号（208或211）
    
    返回：
    - dict: 线路配置字典
    """
    return {
        'busline': busline,
        'station_num_up': STATION_NUM[busline]['upward'],
        'station_num_down': STATION_NUM[busline]['downward'],
        'first_time': OPERATING_TIME[busline]['first'],
        'last_time': OPERATING_TIME[busline]['last'],
        'omega_up': 1 / OMEGA_FACTOR[busline]['upward'],
        'omega_down': 1 / OMEGA_FACTOR[busline]['downward'],
        'max_capacity': MAX_CAPACITY,
        'min_interval': MIN_INTERVAL,
        'max_interval': MAX_INTERVAL
    }


# ==================== 训练配置 ====================
TRAIN_CONFIG = {
    'learning_rate': LEARNING_RATE,
    'batch_size': BATCH_SIZE,
    'gamma': GAMMA,
    'memory_size': MEMORY_SIZE,
    'epsilon': EPSILON,
    'max_episodes': MAX_EPISODES,
    'learning_frequency': LEARNING_FREQUENCY,
    'update_frequency': UPDATE_FREQUENCY,
    'hidden_layers': HIDDEN_LAYERS,
    'hidden_size': HIDDEN_SIZE
}

# ==================== 状态空间维度 ====================
# 论文公式2.1：10维状态空间
# [a1, a2, x1, x2, x3, x4, y1, y2, y3, y4]
STATE_DIM = 10

# ==================== 动作空间维度 ====================
# 4个动作组合：(0,0), (0,1), (1,0), (1,1)
ACTION_DIM = 4


if __name__ == "__main__":
    # 打印配置信息
    print("="*60)
    print("DRL-TSBC Configuration")
    print("="*60)
    
    print("\nNetwork Configuration (Table 2-2):")
    for key, value in TRAIN_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\nLine 208 Configuration:")
    config_208 = get_line_config(208)
    for key, value in config_208.items():
        print(f"  {key}: {value}")
    
    print("\nLine 211 Configuration:")
    config_211 = get_line_config(211)
    for key, value in config_211.items():
        print(f"  {key}: {value}")
    
    print("\nReward Function Parameters:")
    print(f"  μ (mu): {MU}")
    print(f"  δ (delta): {DELTA}")
    print(f"  β (beta): {BETA}")
    print(f"  ζ (zeta): {ZETA}")
    
    print("\n" + "="*60)
