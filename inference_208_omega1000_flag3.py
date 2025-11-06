"""
专门的推理脚本：208线，omega=1/1000，使用avg_flag=3
生成方差最小化后的时刻表
"""

import pandas as pd
import numpy as np
import torch
import os
from drl_tsbc_brain import DQN
from drl_tsbc_environment import (
    Station, BidirectionalBusSystem,
    first_time, last_time, min_Interval, max_Interval,
    device
)
from data_loader import BusDataLoader, check_data_files

# ==================== 配置 ====================
busline = 208
direction_up = 0
direction_down = 1

# omega参数
omega_factor = 1000
omega = 1 / omega_factor

# 均匀排班配置
avg_flag = 3  # 特定区间方差最小化
start_time = "10:00"  # 方差最小化的开始时间
end_time = "16:00"    # 方差最小化的结束时间

# 数据路径
data_dir = f"./test_data/{busline}"
model_load_path = f"saved_models/{busline}_omega{omega_factor}.pth"

# 输出路径
output_dir = f"training_checkpoints/Omega_{busline}_{omega_factor}_flag3"
os.makedirs(output_dir, exist_ok=True)

print("="*80)
print(f"推理配置：208线，omega=1/{omega_factor}，avg_flag=3")
print("="*80)
print(f"模型路径: {model_load_path}")
print(f"输出目录: {output_dir}")
print(f"方差最小化区间: {start_time} - {end_time}")
print("="*80)

# ==================== 检查数据 ====================
if not check_data_files(busline, "./test_data"):
    print("\nError: Required data files not found!")
    exit(1)

# ==================== 加载数据 ====================
loader = BusDataLoader(data_dir="./test_data")
upward_passenger_df, downward_passenger_df, config = loader.load_all_data(busline)

station_num_up = config['upward']['station_num']
station_num_down = config['downward']['station_num']
trf_con_up = config['upward']['traffic']
trf_con_down = config['downward']['traffic']
pn_on_max = 48

passenger_info_path_up = f"{data_dir}/passenger_dataframe_direction{direction_up}.csv"
passenger_info_path_down = f"{data_dir}/passenger_dataframe_direction{direction_down}.csv"

# 计算时间阈值
first_minute_th = (int(first_time[:-3]) - int(trf_con_up.iloc[0, 0])) * 60 + (
    int(first_time[-2:]) - int(trf_con_up.iloc[0, 1])
)
last_minute_th = (int(last_time[:-3]) - int(trf_con_up.iloc[0, 0])) * 60 + (
    int(last_time[-2:]) - int(trf_con_up.iloc[0, 1])
)
start_minute_th = (int(start_time[:-3]) - int(trf_con_up.iloc[0, 0])) * 60 + (
    int(start_time[-2:]) - int(trf_con_up.iloc[0, 1])
)
end_minute_th = (int(end_time[:-3]) - int(trf_con_up.iloc[0, 0])) * 60 + (
    int(end_time[-2:]) - int(trf_con_up.iloc[0, 1])
)

# ==================== 方差最小化函数 ====================
def calculate_variance(schedule, start_minute_th=None, end_minute_th=None):
    """计算发车间隔的方差"""
    schedule = np.array(schedule)
    
    if start_minute_th is not None and end_minute_th is not None:
        # 只计算特定区间的方差
        start_index = np.argmax(schedule >= start_minute_th)
        end_index = np.argmax(schedule >= end_minute_th)
        if end_index == 0:
            end_index = len(schedule)
        selected_schedule = schedule[start_index:end_index]
        if len(selected_schedule) < 2:
            return 0
        return np.var(np.diff(selected_schedule))
    else:
        # 计算全局方差
        if len(schedule) < 2:
            return 0
        return np.var(np.diff(schedule))


def minimize_variance(departure_time, avg_flag=3, max_attempts=50000, 
                     start_minute_th=None, end_minute_th=None):
    """方差最小化均匀调整"""
    if avg_flag == 3 and start_minute_th is not None and end_minute_th is not None:
        original_variance = calculate_variance(departure_time, start_minute_th, end_minute_th)
    elif avg_flag == 2:
        original_variance = calculate_variance(departure_time)
    else:
        return departure_time
    
    print(f'  原始方差: {original_variance:.4f}')
    
    current_variance = original_variance
    min_variance = current_variance
    min_variance_schedule = departure_time.copy()
    attempts = 0
    
    while current_variance > 0 and attempts < max_attempts:
        adjusted_schedule = departure_time.copy()
        
        for i in range(1, len(adjusted_schedule) - 1):
            # 根据avg_flag判断是否跳过
            if avg_flag == 2 and (i == 0 or i == len(adjusted_schedule) - 1):
                continue
            
            if avg_flag == 3 and (i == 0 or i == len(adjusted_schedule) - 1 or
                                 (start_minute_th is not None and end_minute_th is not None and
                                  (adjusted_schedule[i] < start_minute_th or adjusted_schedule[i] > end_minute_th))):
                continue
            
            # 随机调整，最大不超过2分钟
            adjustment = np.random.uniform(-2, 2)
            adjusted_schedule[i] += adjustment
        
        # 更新方差
        if avg_flag == 2:
            current_variance = calculate_variance(adjusted_schedule)
        elif avg_flag == 3 and start_minute_th is not None and end_minute_th is not None:
            current_variance = calculate_variance(adjusted_schedule, start_minute_th, end_minute_th)
        
        attempts += 1
        
        # 保留最小方差时的发车时间表
        if current_variance < min_variance:
            min_variance = current_variance
            min_variance_schedule = adjusted_schedule.copy()
    
    print(f'  最小方差: {min_variance:.4f} (经过 {attempts} 次尝试)')
    return min_variance_schedule


# ==================== 推理 ====================
def inference():
    """推理过程"""
    
    # 检查模型是否存在
    if not os.path.exists(model_load_path):
        print(f"Error: Model file not found at {model_load_path}")
        return
    
    # 加载模型
    print(f"\n加载模型...")
    model = DQN(n_states=10, n_actions=4, model_save_path=model_load_path)
    
    # 初始化环境
    upward_station = Station(station_num_up, passenger_info_path_up, first_minute_th)
    downward_station = Station(station_num_down, passenger_info_path_down, first_minute_th)
    
    bus_system = BidirectionalBusSystem(
        upward_station, downward_station, pn_on_max, trf_con_up, trf_con_down
    )
    
    # 初始化时间
    right_minute_th = first_minute_th
    
    # 首班车发车
    bus_system.Action((1, 1), right_minute_th, min_Interval, max_Interval,
                     start_minute_th=None, end_minute_th=None, ideal_interval=None, avg_flag=0)
    
    # 获取初始状态
    state = bus_system.get_full_state(right_minute_th)
    
    # 记录发车时间
    departure_times_up = [right_minute_th]
    departure_times_down = [right_minute_th]
    
    # 统计变量
    total_wait_up = 0
    total_wait_down = 0
    cant_taken_up = 0
    cant_taken_down = 0
    
    print("\n开始推理...")
    
    # 主循环
    while True:
        if right_minute_th > last_minute_th + 50:
            bus_system.end_label = 1
        
        if bus_system.end_label == 1:
            break
        
        # 使用网络选择动作
        action_idx, action_tuple = model.choose_action(
            state, min_Interval, max_Interval, 1.0,
            bus_system.upward_system.Interval,
            bus_system.downward_system.Interval,
            up_count=len(departure_times_up),
            down_count=len(departure_times_down),
            balance_threshold=1
        )
        
        # 末班车强制发车
        if right_minute_th == last_minute_th:
            action_tuple = (1, 1)
        
        # 执行动作（不使用avg_flag，在推理后统一处理）
        if right_minute_th != first_minute_th:
            bus_system.Action(
                action_tuple, right_minute_th, min_Interval, max_Interval,
                start_minute_th=None, end_minute_th=None, ideal_interval=None, avg_flag=0
            )
        
        # 记录发车时间
        if action_tuple[0] == 1 and bus_system.upward_system.Interval == 0:
            departure_times_up.append(right_minute_th)
        if action_tuple[1] == 1 and bus_system.downward_system.Interval == 0:
            departure_times_down.append(right_minute_th)
        
        # 环境前进
        bus_system.step_forward(right_minute_th)
        
        # 累计统计
        total_wait_up += bus_system.upward_system.if_depart_wait_time
        total_wait_down += bus_system.downward_system.if_depart_wait_time
        cant_taken_up += bus_system.upward_system.Cant_taken_once
        cant_taken_down += bus_system.downward_system.Cant_taken_once
        
        # 时间前进
        right_minute_th += 1
        
        # 获取下一个状态
        state = bus_system.get_full_state(right_minute_th)
    
    print("推理完成。")
    
    # 后处理：方差最小化
    print("\n" + "="*80)
    print("应用方差最小化（avg_flag=3）")
    print("="*80)
    
    print(f"\n上行方向 ({start_time}-{end_time}):")
    departure_times_up = minimize_variance(
        departure_times_up, avg_flag=3, 
        start_minute_th=start_minute_th, end_minute_th=end_minute_th
    )
    
    print(f"\n下行方向 ({start_time}-{end_time}):")
    departure_times_down = minimize_variance(
        departure_times_down, avg_flag=3,
        start_minute_th=start_minute_th, end_minute_th=end_minute_th
    )
    
    # 计算统计指标
    total_minutes = last_minute_th - first_minute_th
    awt_up = total_wait_up / total_minutes / station_num_up if total_minutes > 0 else 0
    awt_down = total_wait_down / total_minutes / station_num_down if total_minutes > 0 else 0
    
    # 输出结果
    print("\n" + "="*80)
    print("推理结果")
    print("="*80)
    print(f"\n{'线路':<10} {'方向':<10} {'指标':<30} {'结果':<15}")
    print("-" * 80)
    print(f"{busline:<10} {'上行':<10} {'发车次数':<30} {len(departure_times_up):<15}")
    print(f"{'':<10} {'':<10} {'乘客平均等待时间 (m)':<30} {awt_up:<15.2f}")
    print(f"{'':<10} {'':<10} {'被滞留乘客数量':<30} {int(cant_taken_up):<15}")
    print(f"{'':<10} {'':<10} {'ω':<30} {'1/' + str(omega_factor):<15}")
    print(f"{'':<10} {'':<10} {'avg_flag':<30} {avg_flag:<15}")
    print("-" * 80)
    print(f"{'':<10} {'下行':<10} {'发车次数':<30} {len(departure_times_down):<15}")
    print(f"{'':<10} {'':<10} {'乘客平均等待时间 (m)':<30} {awt_down:<15.2f}")
    print(f"{'':<10} {'':<10} {'被滞留乘客数量':<30} {int(cant_taken_down):<15}")
    print(f"{'':<10} {'':<10} {'ω':<30} {'1/' + str(omega_factor):<15}")
    print(f"{'':<10} {'':<10} {'avg_flag':<30} {avg_flag:<15}")
    print("=" * 80)
    
    # 保存结果
    result_file = f"{output_dir}/result_omega{omega_factor}_flag{avg_flag}.txt"
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"推理结果：208线，omega=1/{omega_factor}，avg_flag={avg_flag}\n")
        f.write(f"{'='*80}\n\n")
        
        f.write(f"{'线路':<10} {'方向':<10} {'指标':<30} {'结果':<15}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{busline:<10} {'上行':<10} {'发车次数':<30} {len(departure_times_up):<15}\n")
        f.write(f"{'':<10} {'':<10} {'乘客平均等待时间 (m)':<30} {awt_up:<15.2f}\n")
        f.write(f"{'':<10} {'':<10} {'被滞留乘客数量':<30} {int(cant_taken_up):<15}\n")
        f.write(f"{'':<10} {'':<10} {'ω':<30} {'1/' + str(omega_factor):<15}\n")
        f.write(f"{'':<10} {'':<10} {'avg_flag':<30} {avg_flag:<15}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'':<10} {'下行':<10} {'发车次数':<30} {len(departure_times_down):<15}\n")
        f.write(f"{'':<10} {'':<10} {'乘客平均等待时间 (m)':<30} {awt_down:<15.2f}\n")
        f.write(f"{'':<10} {'':<10} {'被滞留乘客数量':<30} {int(cant_taken_down):<15}\n")
        f.write(f"{'':<10} {'':<10} {'ω':<30} {'1/' + str(omega_factor):<15}\n")
        f.write(f"{'':<10} {'':<10} {'avg_flag':<30} {avg_flag:<15}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("\n上行发车时间:\n")
        for i, t in enumerate(departure_times_up):
            hour = int(t // 60)
            minute = int(t % 60)
            f.write(f"  {i+1}. {hour:02d}:{minute:02d}\n")
        
        f.write("\n下行发车时间:\n")
        for i, t in enumerate(departure_times_down):
            hour = int(t // 60)
            minute = int(t % 60)
            f.write(f"  {i+1}. {hour:02d}:{minute:02d}\n")
    
    print(f"\n结果已保存到: {result_file}")
    print("="*80)


if __name__ == "__main__":
    inference()
