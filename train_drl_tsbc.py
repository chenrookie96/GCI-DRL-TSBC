"""
DRL-TSBC 训练脚本
改动说明：
1. 基于Environment.py的主程序改造
2. 实现论文算法2.1的完整训练流程
3. 同时训练上下行两个方向
4. 使用10维状态空间和4个动作
"""

import pandas as pd
import numpy as np
import torch
import time
import os
from drl_tsbc_brain import DQN
from drl_tsbc_environment import (
    Station, BidirectionalBusSystem, 
    first_time, last_time, min_Interval, max_Interval,
    epsilon, mu, delta, beta, zeta, alpha, device,
    start_time, end_time, ideal_interval, avg_flag
)
from data_loader import BusDataLoader, check_data_files  # 新增：使用统一的数据加载器

# 线路配置
busline = 208
direction_up = 0  # 上行方向
direction_down = 1  # 下行方向

# omega参数（训练1/3000版本）
omega_factor = 3000
omega = 1 / omega_factor

# 训练参数（论文表2-2）
# E=50: 论文表2-2规定的最大模拟次数
# 注意：单向示例代码使用500轮，可能是为了更充分的训练
# 当前：使用100轮进行训练
max_episode = 100  # 最大模拟次数E
train_counter = 5  # 学习频率P

# 数据路径
data_dir = f"./test_data/{busline}"
model_save_path = f"{data_dir}/drl_tsbc_model_{busline}_{omega_factor}.pth"

# 检查数据文件是否存在
if not check_data_files(busline, "./test_data"):
    print("\nError: Required data files not found!")
    print("Please ensure the following files exist:")
    print(f"  - ./test_data/{busline}/passenger_dataframe_direction0.csv")
    print(f"  - ./test_data/{busline}/passenger_dataframe_direction1.csv")
    print(f"  - ./test_data/{busline}/traffic-0.csv")
    print(f"  - ./test_data/{busline}/traffic-1.csv")
    exit(1)

# 使用数据加载器加载数据
loader = BusDataLoader(data_dir="./test_data")
upward_passenger_df, downward_passenger_df, config = loader.load_all_data(busline)

# 从配置中获取站点数量和交通数据
station_num_up = config['upward']['station_num']
station_num_down = config['downward']['station_num']
trf_con_up = config['upward']['traffic']
trf_con_down = config['downward']['traffic']

# 最大载客量
pn_on_max = 48  # 座位数约32，α=1.5，所以最大载客量=32*1.5≈48

# 数据路径（用于Station类）
passenger_info_path_up = f"{data_dir}/passenger_dataframe_direction{direction_up}.csv"
passenger_info_path_down = f"{data_dir}/passenger_dataframe_direction{direction_down}.csv"

# 计算时间阈值（分钟）
first_minute_th = (int(first_time[:-3]) - int(trf_con_up.iloc[0, 0])) * 60 + (
    int(first_time[-2:]) - int(trf_con_up.iloc[0, 1])
)
last_minute_th = (int(last_time[:-3]) - int(trf_con_up.iloc[0, 0])) * 60 + (
    int(last_time[-2:]) - int(trf_con_up.iloc[0, 1])
)
# 均匀排班时间阈值
start_minute_th = (int(start_time[:-3]) - int(trf_con_up.iloc[0, 0])) * 60 + (
    int(start_time[-2:]) - int(trf_con_up.iloc[0, 1])
)
end_minute_th = (int(end_time[:-3]) - int(trf_con_up.iloc[0, 0])) * 60 + (
    int(end_time[-2:]) - int(trf_con_up.iloc[0, 1])
)

print("="*60)
print("DRL-TSBC Training")
print("="*60)
print(f"Line: {busline}")
print(f"Upward stations: {station_num_up}, Downward stations: {station_num_down}")
print(f"Operating time: {first_time} - {last_time}")
print(f"Omega: {omega} (1/{omega_factor})")
print(f"Device: {device}")
print(f"Max episodes: {max_episode}")
print(f"Warm-up steps: 3000 (training starts after memory is full)")
print(f"Training frequency: every {train_counter} steps")
print(f"Uniform scheduling: {'Enabled' if avg_flag > 0 else 'Disabled'}")
if avg_flag == 1:
    print(f"  Mode: Fixed interval in specific period")
    print(f"  Period: {start_time} - {end_time}")
    print(f"  Ideal interval: {ideal_interval} min")
print("="*60)


def train():
    """训练DRL-TSBC模型（论文算法2.1）"""
    
    # 初始化DQN（10维状态，4个动作）
    model = DQN(n_states=10, n_actions=4)
    
    step = 0  # 总训练步数
    
    for episode in range(max_episode):
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{max_episode}")
        print(f"{'='*60}")
        
        # 初始化环境（论文算法2.1第5行）
        # 创建上行车站
        upward_station = Station(
            station_num_up,
            passenger_info_path_up,
            first_minute_th
        )
        
        # 创建下行车站
        downward_station = Station(
            station_num_down,
            passenger_info_path_down,
            first_minute_th
        )
        
        # 创建双向公交系统
        bus_system = BidirectionalBusSystem(
            upward_station,
            downward_station,
            pn_on_max,
            trf_con_up,
            trf_con_down
        )
        
        # 初始化时间
        right_minute_th = first_minute_th
        
        # 首班车强制发车（论文算法2.1中首班车时间ts）
        bus_system.Action((1, 1), right_minute_th, min_Interval, max_Interval)
        
        # 获取初始状态
        state = bus_system.get_full_state(right_minute_th)
        
        # 统计变量
        ep_reward = 0
        departure_time_up = [right_minute_th]
        departure_time_down = [right_minute_th]
        
        # AWT计算变量（与原始代码对齐）
        tot_wait_time_up = 0
        tot_wait_time_down = 0
        
        # 滞留乘客统计
        total_stranded_up = 0
        total_stranded_down = 0
        
        # 主循环（论文算法2.1第6行）
        while True:
            # 检查是否结束
            if right_minute_th > last_minute_th + 50:
                bus_system.end_label = 1
            
            if bus_system.end_label == 1:
                break
            
            # 选择动作（论文算法2.1第7-10行）
            # 添加发车次数参数以支持硬约束
            action_idx, action_tuple = model.choose_action(
                state,
                min_Interval,
                max_Interval,
                epsilon,
                bus_system.upward_system.Interval,
                bus_system.downward_system.Interval,
                up_count=len(departure_time_up),
                down_count=len(departure_time_down),
                balance_threshold=1
            )
            
            # 末班车强制发车
            if right_minute_th == last_minute_th:
                action_tuple = (1, 1)
                action_idx = 3
            
            # 首班车之后才开始决策
            if right_minute_th != first_minute_th:
                # 执行动作（论文算法2.1第18行）
                bus_system.Action(
                    action_tuple, right_minute_th, min_Interval, max_Interval,
                    start_minute_th, end_minute_th, ideal_interval, avg_flag
                )
            
            # 记录发车时间
            if action_tuple[0] == 1 and bus_system.upward_system.Interval == 0:
                departure_time_up.append(right_minute_th)
            if action_tuple[1] == 1 and bus_system.downward_system.Interval == 0:
                departure_time_down.append(right_minute_th)
            
            # 计算奖励（论文算法2.1第18行，论文公式2.15-2.17）
            reward = bus_system.calculate_reward(action_tuple, omega)
            
            # 累积等待时间（与原始代码对齐：每一步都累积，无论是否发车）
            tot_wait_time_up += bus_system.upward_system.if_depart_wait_time
            tot_wait_time_down += bus_system.downward_system.if_depart_wait_time
            
            # 累积滞留乘客数
            total_stranded_up += bus_system.upward_system.Cant_taken_once
            total_stranded_down += bus_system.downward_system.Cant_taken_once
            
            # 环境前进一步
            bus_system.step_forward(right_minute_th)
            
            # 时间前进
            right_minute_th += 1
            
            # 获取下一个状态（论文算法2.1第18行）
            next_state = bus_system.get_full_state(right_minute_th)
            
            # 存储经验（论文算法2.1第19-20行）
            model.store_transition(state, action_idx, reward, next_state)
            
            # 训练网络（论文算法2.1第22-26行）
            # 当经验池满且每P步训练一次
            if model.memory_counter > 3000 and step % train_counter == 0:
                model.train_network(model_save_path)
            
            step += 1
            ep_reward += reward
            state = next_state
            
            # 定期打印信息
            if step % 1000 == 0:
                print(f"Step {step}, Time {right_minute_th}, "
                      f"Reward {reward:.4f}, Action {action_tuple}, "
                      f"Up departures: {len(departure_time_up)}, "
                      f"Down departures: {len(departure_time_down)}")
        
        # Episode结束统计
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1} finished")
        print(f"Total reward: {ep_reward:.3f}")
        print(f"Upward departures: {len(departure_time_up)}")
        print(f"Downward departures: {len(departure_time_down)}")
        print(f"Departure difference: {abs(len(departure_time_up) - len(departure_time_down))}")
        
        # 计算平均等待时间（与原始代码完全对齐）
        # 原始代码：AWT = tot_wait_time / (last_minute_th - first_minute_th) / station_num
        # tot_wait_time 是每一步累积的 if_depart_wait_time
        total_minutes = right_minute_th - first_minute_th
        
        awt_up = tot_wait_time_up / total_minutes / station_num_up if total_minutes > 0 else 0
        awt_down = tot_wait_time_down / total_minutes / station_num_down if total_minutes > 0 else 0
        
        print(f"Upward AWT: {awt_up:.3f} min")
        print(f"Downward AWT: {awt_down:.3f} min")
        
        # 显示滞留乘客数
        print(f"Upward stranded passengers: {int(total_stranded_up)}")
        print(f"Downward stranded passengers: {int(total_stranded_down)}")
        print(f"Total stranded passengers: {int(total_stranded_up + total_stranded_down)}")
        print(f"{'='*60}")
        
        # 每个episode结束时保存模型（用于后续选择最佳模型）
        # 创建专门的文件夹：项目根目录下的 training_checkpoints/omega500/
        checkpoint_dir = f"./training_checkpoints/omega{omega_factor}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        episode_model_path = f"{checkpoint_dir}/ep{episode+1:03d}_dep{len(departure_time_up)+len(departure_time_down)}_awt{(awt_up+awt_down)/2:.2f}.pth"
        torch.save(model.target_net, episode_model_path)
        
        # 保存episode统计信息到CSV
        episode_stats_file = f"{checkpoint_dir}/training_stats.csv"
        stats_data = {
            'episode': episode + 1,
            'total_reward': ep_reward,
            'up_departures': len(departure_time_up),
            'down_departures': len(departure_time_down),
            'awt_up': awt_up,
            'awt_down': awt_down,
            'awt_avg': (awt_up + awt_down) / 2,
            'stranded_up': int(total_stranded_up),
            'stranded_down': int(total_stranded_down),
            'stranded_total': int(total_stranded_up + total_stranded_down),
            'model_path': episode_model_path
        }
        
        # 追加到CSV文件
        import pandas as pd
        if os.path.exists(episode_stats_file):
            df = pd.read_csv(episode_stats_file)
            df = pd.concat([df, pd.DataFrame([stats_data])], ignore_index=True)
        else:
            df = pd.DataFrame([stats_data])
        df.to_csv(episode_stats_file, index=False)
        print(f"Episode model saved: {episode_model_path}")
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Model saved to: {model_save_path}")
    print("="*60)


if __name__ == "__main__":
    train()
