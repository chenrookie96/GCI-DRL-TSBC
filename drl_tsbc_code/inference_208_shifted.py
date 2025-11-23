"""
使用提前晚高峰的数据进行推理
展示DRL-TSBC的在线调整能力
"""

import pandas as pd
import numpy as np
import torch
import os
from drl_tsbc_brain import DQN
from drl_tsbc_environment import (
    Station, BidirectionalBusSystem,
    first_time, last_time, min_Interval, max_Interval,
    device,
    start_time, end_time, ideal_interval, avg_flag  # 均匀排班参数
)
from data_loader import BusDataLoader

# 线路配置
busline = 208
direction_up = 0
direction_down = 1
omega_factor = 1000  # 使用omega1000
omega = 1 / omega_factor

# 数据路径 - 使用提前晚高峰的数据
data_dir = f"./test_data/{busline}"
# 使用saved_models中的模型
model_load_path = f"./saved_models/{busline}_omega{omega_factor}.pth"

# 最大载客量
pn_on_max = 48

print("="*60)
print("DRL-TSBC 提前晚高峰推理")
print("="*60)
print(f"线路: {busline}")
print(f"模型: {model_load_path}")
print(f"使用提前晚高峰的乘客数据")
print("="*60)

# 加载数据
passenger_info_path_up = f"{data_dir}/passenger_dataframe_direction{direction_up}_shifted.csv"
passenger_info_path_down = f"{data_dir}/passenger_dataframe_direction{direction_down}.csv"

# 检查文件是否存在
if not os.path.exists(passenger_info_path_up):
    print(f"\n错误: 找不到提前晚高峰的数据文件: {passenger_info_path_up}")
    print("请先运行 create_shifted_passenger_data.py 创建数据")
    exit(1)

# 加载配置
loader = BusDataLoader(data_dir="./test_data")
_, _, config = loader.load_all_data(busline)

station_num_up = config['upward']['station_num']
station_num_down = config['downward']['station_num']
trf_con_up = config['upward']['traffic']
trf_con_down = config['downward']['traffic']

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

# 加载模型
print(f"\n加载模型: {model_load_path}")
model = DQN(n_states=10, n_actions=4, model_save_path=model_load_path)

# 初始化环境
upward_station = Station(station_num_up, passenger_info_path_up, first_minute_th)
downward_station = Station(station_num_down, passenger_info_path_down, first_minute_th)

bus_system = BidirectionalBusSystem(
    upward_station,
    downward_station,
    pn_on_max,
    trf_con_up,
    trf_con_down
)

# 初始化
right_minute_th = first_minute_th
bus_system.Action((1, 1), right_minute_th, min_Interval, max_Interval)
state = bus_system.get_full_state(right_minute_th)

departure_times_up = [right_minute_th]
departure_times_down = [right_minute_th]

total_wait_up = 0
total_wait_down = 0
cant_taken_up = 0
cant_taken_down = 0

def adjust_schedule(departure_times_up, departure_times_down, Tmax):
    """
    后处理算法：调整时刻表使上下行发车次数相等
    与inference_drl_tsbc.py保持一致
    """
    up_count = len(departure_times_up)
    down_count = len(departure_times_down)
    
    if up_count == down_count:
        print("上下行发车次数已相等，无需调整")
        return departure_times_up, departure_times_down
    
    # 选择发车次数更多的方向
    if up_count > down_count:
        print(f"调整上行时刻表（从 {up_count} 调整到 {down_count} 次）")
        times_to_adjust = departure_times_up.copy()
        direction = "upward"
    else:
        print(f"调整下行时刻表（从 {down_count} 调整到 {up_count} 次）")
        times_to_adjust = departure_times_down.copy()
        direction = "downward"
    
    # 删除倒数第二次发车
    if len(times_to_adjust) >= 2:
        del times_to_adjust[-2]
        print(f"删除倒数第二次发车")
    
    # 从后向前调整发车时间
    k = len(times_to_adjust) - 1
    while k > 0:
        interval = times_to_adjust[k] - times_to_adjust[k-1]
        if interval > Tmax:
            times_to_adjust[k-1] = times_to_adjust[k] - Tmax
            print(f"调整第 {k-1} 次发车时间以保持最大间隔")
            k -= 1
        else:
            break
    
    # 返回调整后的结果
    if direction == "upward":
        return times_to_adjust, departure_times_down
    else:
        return departure_times_up, times_to_adjust


print("\n开始推理...")

# 主循环
while True:
    if right_minute_th > last_minute_th + 50:
        bus_system.end_label = 1
    
    if bus_system.end_label == 1:
        break
    
    # 选择动作
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
    
    # 执行动作
    if right_minute_th != first_minute_th:
        bus_system.Action(
            action_tuple, right_minute_th, min_Interval, max_Interval,
            start_minute_th, end_minute_th, ideal_interval, avg_flag
        )
    
    # 记录发车
    if action_tuple[0] == 1 and bus_system.upward_system.Interval == 0:
        departure_times_up.append(right_minute_th)
    if action_tuple[1] == 1 and bus_system.downward_system.Interval == 0:
        departure_times_down.append(right_minute_th)
    
    # 环境前进
    bus_system.step_forward(right_minute_th)
    
    # 统计
    total_wait_up += bus_system.upward_system.if_depart_wait_time
    total_wait_down += bus_system.downward_system.if_depart_wait_time
    cant_taken_up += bus_system.upward_system.Cant_taken_once
    cant_taken_down += bus_system.downward_system.Cant_taken_once
    
    right_minute_th += 1
    state = bus_system.get_full_state(right_minute_th)

print("推理完成")

# 后处理：确保上下行发车次数相等
print("\n" + "="*80)
print("后处理：调整时刻表")
print("="*80)
departure_times_up, departure_times_down = adjust_schedule(
    departure_times_up,
    departure_times_down,
    max_Interval
)

# 计算统计
total_minutes = last_minute_th - first_minute_th
awt_up = total_wait_up / total_minutes / station_num_up if total_minutes > 0 else 0
awt_down = total_wait_down / total_minutes / station_num_down if total_minutes > 0 else 0

# 输出结果
print("\n" + "="*80)
print("推理结果（提前晚高峰）")
print("="*80)
print(f"\n{'线路':<10} {'方向':<10} {'指标':<30} {'结果':<15}")
print("-" * 80)
print(f"{busline:<10} {'上行':<10} {'发车次数':<30} {len(departure_times_up):<15}")
print(f"{'':<10} {'':<10} {'乘客平均等待时间 (m)':<30} {awt_up:<15.2f}")
print(f"{'':<10} {'':<10} {'被滞留乘客数量':<30} {int(cant_taken_up):<15}")
print("-" * 80)
print(f"{'':<10} {'下行':<10} {'发车次数':<30} {len(departure_times_down):<15}")
print(f"{'':<10} {'':<10} {'乘客平均等待时间 (m)':<30} {awt_down:<15.2f}")
print(f"{'':<10} {'':<10} {'被滞留乘客数量':<30} {int(cant_taken_down):<15}")
print("=" * 80)

# 保存结果 - 保存到saved_models文件夹
result_file = f"saved_models/{busline}_omega{omega_factor}_shifted.txt"
with open(result_file, 'w', encoding='utf-8') as f:
    f.write(f"DRL-TSBC 推理结果（提前晚高峰）\n")
    f.write(f"线路: {busline}, omega: 1/{omega_factor}\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"{'线路':<10} {'方向':<10} {'指标':<30} {'结果':<15}\n")
    f.write("-" * 80 + "\n")
    f.write(f"{busline:<10} {'上行':<10} {'发车次数':<30} {len(departure_times_up):<15}\n")
    f.write(f"{'':<10} {'':<10} {'乘客平均等待时间 (m)':<30} {awt_up:<15.2f}\n")
    f.write(f"{'':<10} {'':<10} {'被滞留乘客数量':<30} {int(cant_taken_up):<15}\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'':<10} {'下行':<10} {'发车次数':<30} {len(departure_times_down):<15}\n")
    f.write(f"{'':<10} {'':<10} {'乘客平均等待时间 (m)':<30} {awt_down:<15.2f}\n")
    f.write(f"{'':<10} {'':<10} {'被滞留乘客数量':<30} {int(cant_taken_down):<15}\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("\n上行发车时间:\n")
    for i, t in enumerate(departure_times_up):
        hour = t // 60
        minute = t % 60
        f.write(f"  {i+1}. {hour:02d}:{minute:02d}\n")
    
    f.write("\n下行发车时间:\n")
    for i, t in enumerate(departure_times_down):
        hour = t // 60
        minute = t % 60
        f.write(f"  {i+1}. {hour:02d}:{minute:02d}\n")

print(f"\n结果已保存: {result_file}")
print("="*80)
