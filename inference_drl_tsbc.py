"""
DRL-TSBC 推理脚本
改动说明：
1. 基于work_env.py改造
2. 实现论文算法2.2的推理流程
3. 使用训练好的模型生成双向时刻表
4. 实现后处理算法确保上下行发车次数相等
"""

import pandas as pd
import numpy as np
import torch
import os
from drl_tsbc_brain import DQN
from drl_tsbc_environment import (
    Station, BidirectionalBusSystem,
    first_time, last_time, min_Interval, max_Interval,
    epsilon, mu, delta, beta, zeta, alpha, device,
    start_time, end_time, ideal_interval, avg_flag  # 新增：均匀排班参数
)
from data_loader import BusDataLoader, check_data_files  # 新增：使用统一的数据加载器

# 线路配置
busline = 211
direction_up = 0
direction_down = 1

# omega参数（与训练时保持一致）
omega_factor = 900
omega = 1 / omega_factor

# 均匀排班配置（可在此修改）
#avg_flag = 0  # 不使用均匀排班
# avg_flag = 1  # 特定区间等间隔排班（10:00-16:00，间隔12分钟）
# avg_flag = 2  # 方差最小化均匀调整（全局）
avg_flag = 3  # 方差最小化均匀调整（特定区间）

# 数据路径
data_dir = f"./test_data/{busline}"
# 使用81轮的checkpoint模型
model_load_path = f"./training_checkpoints/Omega_{busline}_{omega_factor}/ep081_dep162_awt3.88.pth"

# 检查数据文件
if not check_data_files(busline, "./test_data"):
    print("\nError: Required data files not found!")
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
pn_on_max = 48

# 数据路径（用于Station类）
passenger_info_path_up = f"{data_dir}/passenger_dataframe_direction{direction_up}.csv"
passenger_info_path_down = f"{data_dir}/passenger_dataframe_direction{direction_down}.csv"

# 计算时间阈值
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
print("DRL-TSBC Inference")
print("="*60)
print(f"Line: {busline}")
print(f"Model path: {model_load_path}")
print(f"Device: {device}")
print(f"Uniform scheduling: {'Enabled' if avg_flag > 0 else 'Disabled'}")
if avg_flag == 1:
    print(f"  Mode: Fixed interval in specific period")
    print(f"  Period: {start_time} - {end_time}")
    print(f"  Ideal interval: {ideal_interval} min")
elif avg_flag == 2:
    print(f"  Mode: Variance minimization (global)")
elif avg_flag == 3:
    print(f"  Mode: Variance minimization (specific period)")
    print(f"  Period: {start_time} - {end_time}")
print("="*60)


def calculate_variance(schedule, start_minute_th=None, end_minute_th=None):
    """
    计算发车时刻表的方差
    
    参数：
    - schedule: 发车时间列表
    - start_minute_th: 开始时间（可选，用于特定区间）
    - end_minute_th: 结束时间（可选，用于特定区间）
    
    返回：
    - float: 发车间隔的方差
    """
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


def minimize_variance(departure_time, avg_flag=2, max_attempts=50000, 
                     start_minute_th=None, end_minute_th=None):
    """
    方差最小化均匀调整（参考work_env.py的adjust_schedule函数）
    
    参数：
    - departure_time: 发车时间列表
    - avg_flag: 2=全局方差最小化，3=特定区间方差最小化
    - max_attempts: 最大尝试次数
    - start_minute_th: 开始时间（avg_flag=3时使用）
    - end_minute_th: 结束时间（avg_flag=3时使用）
    
    返回：
    - 调整后的发车时间列表
    """
    if avg_flag == 3 and start_minute_th is not None and end_minute_th is not None:
        original_variance = calculate_variance(departure_time, start_minute_th, end_minute_th)
    elif avg_flag == 2:
        original_variance = calculate_variance(departure_time)
    else:
        return departure_time
    
    print(f'  Original variance: {original_variance:.4f}')
    
    current_variance = original_variance
    min_variance = current_variance
    min_variance_schedule = departure_time.copy()
    attempts = 0
    
    while current_variance > 0 and attempts < max_attempts:
        adjusted_schedule = departure_time.copy()
        
        for i in range(1, len(adjusted_schedule) - 1):
            # 根据avg_flag判断是否跳过第一个和最后一个发车时间
            if avg_flag == 2 and (i == 0 or i == len(adjusted_schedule) - 1):
                continue
            
            # 根据avg_flag判断是否跳过非特定时间段内的发车时间
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
    
    print(f'  Minimum variance: {min_variance:.4f} (after {attempts} attempts)')
    return min_variance_schedule


def adjust_schedule(departure_times_up, departure_times_down, Tmax):
    """
    后处理算法：调整时刻表使上下行发车次数相等
    改动：实现论文算法2.2第16-21行
    
    参数：
    - departure_times_up: 上行发车时间列表
    - departure_times_down: 下行发车时间列表
    - Tmax: 最大发车间隔
    
    返回：
    - 调整后的上行和下行发车时间列表
    """
    up_count = len(departure_times_up)
    down_count = len(departure_times_down)
    
    if up_count == down_count:
        print("Departure counts are already equal, no adjustment needed.")
        return departure_times_up, departure_times_down
    
    # 选择发车次数更多的方向（论文算法2.2第16行）
    if up_count > down_count:
        print(f"Adjusting upward schedule (from {up_count} to {down_count} departures)")
        times_to_adjust = departure_times_up.copy()
        direction = "upward"
    else:
        print(f"Adjusting downward schedule (from {down_count} to {up_count} departures)")
        times_to_adjust = departure_times_down.copy()
        direction = "downward"
    
    # 删除倒数第二次发车（论文算法2.2第17行）
    if len(times_to_adjust) >= 2:
        del times_to_adjust[-2]
        print(f"Removed second-to-last departure")
    
    # 从后向前调整发车时间（论文算法2.2第18-21行）
    k = len(times_to_adjust) - 1
    while k > 0:
        interval = times_to_adjust[k] - times_to_adjust[k-1]
        if interval > Tmax:
            # 将第k-1次发车时间推迟
            times_to_adjust[k-1] = times_to_adjust[k] - Tmax
            print(f"Adjusted departure {k-1}: moved to maintain Tmax interval")
            k -= 1
        else:
            break
    
    # 返回调整后的结果
    if direction == "upward":
        return times_to_adjust, departure_times_down
    else:
        return departure_times_up, times_to_adjust


def inference():
    """推理过程（论文算法2.2）"""
    
    # 检查模型是否存在
    if not os.path.exists(model_load_path):
        print(f"Error: Model file not found at {model_load_path}")
        print("Please train the model first using train_drl_tsbc.py")
        return
    
    # 加载训练好的模型
    print(f"Loading model from {model_load_path}")
    model = DQN(n_states=10, n_actions=4, model_save_path=model_load_path)
    
    # 初始化环境（论文算法2.2第1行）
    upward_station = Station(station_num_up, passenger_info_path_up, first_minute_th)
    downward_station = Station(station_num_down, passenger_info_path_down, first_minute_th)
    
    bus_system = BidirectionalBusSystem(
        upward_station,
        downward_station,
        pn_on_max,
        trf_con_up,
        trf_con_down
    )
    
    # 初始化时间
    right_minute_th = first_minute_th
    
    # 首班车发车
    bus_system.Action((1, 1), right_minute_th, min_Interval, max_Interval)
    
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
    
    print("\nStarting inference...")
    
    # 主循环（论文算法2.2第2行）
    while True:
        # 检查是否结束
        if right_minute_th > last_minute_th + 50:
            bus_system.end_label = 1
        
        if bus_system.end_label == 1:
            break
        
        # 使用网络选择动作（论文算法2.2第3行）
        # 推理时不使用随机探索，epsilon设为1.0
        # 添加发车次数参数以支持硬约束
        action_idx, action_tuple = model.choose_action(
            state,
            min_Interval,
            max_Interval,
            1.0,  # 推理时不探索
            bus_system.upward_system.Interval,
            bus_system.downward_system.Interval,
            up_count=len(departure_times_up),
            down_count=len(departure_times_down),
            balance_threshold=1
        )
        
        # 末班车强制发车
        if right_minute_th == last_minute_th:
            action_tuple = (1, 1)
        
        # 执行动作（论文算法2.2第11行）
        if right_minute_th != first_minute_th:
            bus_system.Action(
                action_tuple, right_minute_th, min_Interval, max_Interval,
                start_minute_th, end_minute_th, ideal_interval, avg_flag
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
        
        # 获取下一个状态（论文算法2.2第12行）
        state = bus_system.get_full_state(right_minute_th)
    
    print("Inference completed.")
    
    # 后处理：调整时刻表使发车次数相等（论文算法2.2第16-21行）
    print("\n" + "="*60)
    print("Post-processing: Adjusting schedules")
    print("="*60)
    
    # 步骤1：方差最小化（如果启用）
    if avg_flag == 2:
        print("\nApplying variance minimization (global)...")
        departure_times_up = minimize_variance(departure_times_up, avg_flag=2)
        departure_times_down = minimize_variance(departure_times_down, avg_flag=2)
    elif avg_flag == 3:
        print(f"\nApplying variance minimization (period: {start_time}-{end_time})...")
        departure_times_up = minimize_variance(
            departure_times_up, avg_flag=3, 
            start_minute_th=start_minute_th, end_minute_th=end_minute_th
        )
        departure_times_down = minimize_variance(
            departure_times_down, avg_flag=3,
            start_minute_th=start_minute_th, end_minute_th=end_minute_th
        )
    
    # 步骤2：确保上下行发车次数相等
    print("\nEnsuring equal departure counts...")
    departure_times_up, departure_times_down = adjust_schedule(
        departure_times_up,
        departure_times_down,
        max_Interval
    )
    
    # 输出结果（论文表2-3格式）
    print("\n" + "="*80)
    print("DRL-TSBC Inference Results (Table 2-3 Format)")
    print("="*80)
    
    # 计算统计指标
    total_minutes = last_minute_th - first_minute_th
    awt_up = total_wait_up / total_minutes / station_num_up if total_minutes > 0 else 0
    awt_down = total_wait_down / total_minutes / station_num_down if total_minutes > 0 else 0
    
    # 输出表格
    print(f"\n{'线路':<10} {'方向':<10} {'指标':<30} {'结果':<15}")
    print("-" * 80)
    print(f"{busline:<10} {'上行':<10} {'发车次数':<30} {len(departure_times_up):<15}")
    print(f"{'':<10} {'':<10} {'乘客平均等待时间 (m)':<30} {awt_up:<15.2f}")
    print(f"{'':<10} {'':<10} {'被滞留乘客数量':<30} {int(cant_taken_up):<15}")
    print(f"{'':<10} {'':<10} {'ω':<30} {'1/' + str(omega_factor):<15}")
    print("-" * 80)
    print(f"{'':<10} {'下行':<10} {'发车次数':<30} {len(departure_times_down):<15}")
    print(f"{'':<10} {'':<10} {'乘客平均等待时间 (m)':<30} {awt_down:<15.2f}")
    print(f"{'':<10} {'':<10} {'被滞留乘客数量':<30} {int(cant_taken_down):<15}")
    print(f"{'':<10} {'':<10} {'ω':<30} {'1/' + str(omega_factor):<15}")
    print("=" * 80)
    
    # 额外统计信息
    print(f"\n{'统计摘要':<20} {'上行':<15} {'下行':<15} {'总计/平均':<15}")
    print("-" * 80)
    print(f"{'发车次数':<20} {len(departure_times_up):<15} {len(departure_times_down):<15} {len(departure_times_up) + len(departure_times_down):<15}")
    print(f"{'平均等待时间 (m)':<20} {awt_up:<15.2f} {awt_down:<15.2f} {(awt_up + awt_down) / 2:<15.2f}")
    print(f"{'滞留乘客数':<20} {int(cant_taken_up):<15} {int(cant_taken_down):<15} {int(cant_taken_up + cant_taken_down):<15}")
    print(f"{'发车次数差':<20} {'':<15} {'':<15} {abs(len(departure_times_up) - len(departure_times_down)):<15}")
    print("=" * 80)
    
    # 输出发车时刻表
    print(f"\n{'='*60}")
    print("Departure Timetable")
    print(f"{'='*60}")
    
    print("\nUpward departures:")
    for i, t in enumerate(departure_times_up):
        hour = t // 60
        minute = t % 60
        print(f"  {i+1}. {hour:02d}:{minute:02d}")
    
    print("\nDownward departures:")
    for i, t in enumerate(departure_times_down):
        hour = t // 60
        minute = t % 60
        print(f"  {i+1}. {hour:02d}:{minute:02d}")
    
    # 保存结果到文件（论文表2-3格式）
    result_file = f"{data_dir}/drl_tsbc_result_{busline}_{omega_factor}.txt"
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"DRL-TSBC Inference Results for Line {busline}\n")
        f.write(f"{'='*80}\n\n")
        
        # 表2-3格式
        f.write(f"{'线路':<10} {'方向':<10} {'指标':<30} {'结果':<15}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{busline:<10} {'上行':<10} {'发车次数':<30} {len(departure_times_up):<15}\n")
        f.write(f"{'':<10} {'':<10} {'乘客平均等待时间 (m)':<30} {awt_up:<15.2f}\n")
        f.write(f"{'':<10} {'':<10} {'被滞留乘客数量':<30} {int(cant_taken_up):<15}\n")
        f.write(f"{'':<10} {'':<10} {'ω':<30} {'1/' + str(omega_factor):<15}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'':<10} {'下行':<10} {'发车次数':<30} {len(departure_times_down):<15}\n")
        f.write(f"{'':<10} {'':<10} {'乘客平均等待时间 (m)':<30} {awt_down:<15.2f}\n")
        f.write(f"{'':<10} {'':<10} {'被滞留乘客数量':<30} {int(cant_taken_down):<15}\n")
        f.write(f"{'':<10} {'':<10} {'ω':<30} {'1/' + str(omega_factor):<15}\n")
        f.write("=" * 80 + "\n\n")
        
        # 统计摘要
        f.write(f"{'统计摘要':<20} {'上行':<15} {'下行':<15} {'总计/平均':<15}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'发车次数':<20} {len(departure_times_up):<15} {len(departure_times_down):<15} {len(departure_times_up) + len(departure_times_down):<15}\n")
        f.write(f"{'平均等待时间 (m)':<20} {awt_up:<15.2f} {awt_down:<15.2f} {(awt_up + awt_down) / 2:<15.2f}\n")
        f.write(f"{'滞留乘客数':<20} {int(cant_taken_up):<15} {int(cant_taken_down):<15} {int(cant_taken_up + cant_taken_down):<15}\n")
        f.write(f"{'发车次数差':<20} {'':<15} {'':<15} {abs(len(departure_times_up) - len(departure_times_down)):<15}\n")
        f.write("=" * 80 + "\n\n")
        
        # 发车时刻表
        f.write("\n发车时刻表\n")
        f.write("=" * 80 + "\n\n")
        f.write("上行发车时间:\n")
        for i, t in enumerate(departure_times_up):
            hour = t // 60
            minute = t % 60
            f.write(f"  {i+1}. {hour:02d}:{minute:02d}\n")
        
        f.write("\n下行发车时间:\n")
        for i, t in enumerate(departure_times_down):
            hour = t // 60
            minute = t % 60
            f.write(f"  {i+1}. {hour:02d}:{minute:02d}\n")
    
    print(f"\n结果已保存到: {result_file}")
    print("="*80)


if __name__ == "__main__":
    inference()
