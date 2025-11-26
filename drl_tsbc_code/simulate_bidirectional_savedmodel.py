"""
使用saved_models中的推理结果进行双向模拟
使用同一个发车时刻表（基于原始数据训练的模型）：
- 上行：模拟shifted数据（晚高峰提前）
- 下行：模拟原始数据（不提前）
用于绘制图2-7（展示同一时刻表在不对称需求下的表现）
"""

import sys
sys.path.insert(0, 'drl_tsbc_code')

import pandas as pd
import numpy as np
import re
from drl_tsbc_environment import Station, DirectionSystem

# 配置参数
busline = 208
max_capacity = 48
omega_factor = 1000

# 时间设置
first_minute_th = 360  # 从6:00开始
last_minute_th = 1260  # 到21:00

print("="*80)
print("图2-7实验：同一时刻表 + 不对称需求")
print("上行：shifted数据（晚高峰提前）")
print("下行：原始数据（不提前）")
print("="*80)

def parse_departure_times(result_file, direction_name):
    """解析发车时刻表"""
    with open(result_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if direction_name == "上行":
        pattern = r'上行发车时间:\n(.*?)\n\n下行发车时间:'
    else:
        pattern = r'下行发车时间:\n(.*?)(?:\n\n|$)'
    
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        raise ValueError(f"无法找到{direction_name}发车时间")
    
    time_section = match.group(1)
    departure_times = []
    
    for line in time_section.strip().split('\n'):
        time_match = re.search(r'\d+\.\s+(\d+):(\d+)', line)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2))
            total_minutes = hour * 60 + minute
            departure_times.append(total_minutes)
    
    return departure_times

def simulate_direction(direction, departure_times, use_shifted=False):
    """模拟单个方向"""
    direction_name = '上行' if direction == 0 else '下行'
    data_type = 'shifted' if use_shifted else '原始'
    
    print(f"\n{'='*80}")
    print(f"模拟方向 {direction} ({direction_name}) - 使用{data_type}数据")
    print(f"{'='*80}")
    
    # 数据路径 - 根据是否shifted选择不同的数据
    if use_shifted:
        passenger_info_path = f"test_data/{busline}/passenger_dataframe_direction{direction}_shifted.csv"
    else:
        passenger_info_path = f"test_data/{busline}/passenger_dataframe_direction{direction}.csv"
    trf_path = f"test_data/{busline}/traffic-{direction}.csv"
    
    # 读取数据
    trf_con = pd.read_csv(trf_path)
    passenger_df_temp = pd.read_csv(passenger_info_path)
    station_num = passenger_df_temp['Boarding station'].max() + 1
    
    print(f"  站点数: {station_num}")
    print(f"  发车次数: {len(departure_times)}")
    print(f"  首班车: {departure_times[0]//60}:{departure_times[0]%60:02d}")
    print(f"  末班车: {departure_times[-1]//60}:{departure_times[-1]%60:02d}")
    
    # 初始化环境
    station = Station(station_num, passenger_info_path, first_minute_th)
    system = DirectionSystem(station, max_capacity, trf_con)
    
    # 模拟运行
    print("\n开始模拟...")
    print("-"*80)
    
    Cap_half_hour = 0
    Cap_list = []
    real_need_half_hour = 0
    real_need_list = []
    
    departure_index = 0
    right_minute_th = first_minute_th
    
    while right_minute_th <= last_minute_th:
        # 检查是否需要发车
        if departure_index < len(departure_times) and right_minute_th == departure_times[departure_index]:
            system.Departure(right_minute_th)
            departure_index += 1
        
        # 统计当前在线车辆上的乘客数
        bus_on_line_now = []
        for bus in system.bus_online:
            if bus.arrv_mark != 1:
                bus_on_line_now.append(len(bus.pn_on))
        
        # 记录当前时刻的容量
        Cap_half_hour += system.All_cap_take / (station_num - 1)
        
        # 累加真实需求
        real_need_half_hour += sum(bus_on_line_now)
        
        # 每半小时统计一次
        if right_minute_th <= last_minute_th and right_minute_th % 30 == 0:
            Cap_half_hour = Cap_half_hour / 30
            Cap_list.append(Cap_half_hour)
            Cap_half_hour = 0
            
            real_need_half_hour = real_need_half_hour / 30
            real_need_list.append(real_need_half_hour)
            real_need_half_hour = 0
            
            # 打印进度
            hour = right_minute_th // 60
            minute = right_minute_th % 60
            active_buses = len([b for b in system.bus_online if b.arrv_mark != 1])
            print(f"  {hour:02d}:{minute:02d} - 活跃车辆: {active_buses:2d}, "
                  f"真实需求: {real_need_list[-1]:6.2f}, 容量: {Cap_list[-1]:6.2f}")
        
        # 环境前进一步
        system.step_forward(right_minute_th)
        
        # 推进到下一分钟
        right_minute_th += 1
    
    print("-"*80)
    
    # 提取从7:00开始的数据
    real_need_list_from_7 = real_need_list[2:31]
    Cap_list_from_7 = Cap_list[2:31]
    
    return real_need_list_from_7, Cap_list_from_7

# 主程序
# 使用mixed推理结果（上行shifted + 下行原始）
result_file = f"saved_models/{busline}_omega{omega_factor}_mixed.txt"

# 检查文件是否存在
import os
if not os.path.exists(result_file):
    print(f"\n错误: 找不到推理结果文件: {result_file}")
    print("请先运行: python drl_tsbc_code/inference_208_mixed.py")
    exit(1)

# 解析上下行发车时刻表
print("\n读取发车时刻表（上行shifted + 下行原始的推理结果）...")
upward_times = parse_departure_times(result_file, "上行")
downward_times = parse_departure_times(result_file, "下行")

print(f"\n上行发车次数: {len(upward_times)}")
print(f"下行发车次数: {len(downward_times)}")
print(f"发车次数是否相等: {'是' if len(upward_times) == len(downward_times) else '否'}")

print("\n实验设计:")
print("  - 使用针对不对称需求推理的发车时刻表")
print("  - 上行：模拟shifted数据（晚高峰提前1小时）")
print("  - 下行：模拟原始数据（保持不变）")
print("  - 目的：展示DRL-TSBC在不对称需求下保持发车次数一致")

# 模拟上行（使用shifted数据 - 晚高峰提前）
upward_real, upward_cap = simulate_direction(0, upward_times, use_shifted=True)

# 模拟下行（使用原始数据 - 不提前）
downward_real, downward_cap = simulate_direction(1, downward_times, use_shifted=False)

# 打印结果
print("\n" + "="*80)
print("模拟完成！")
print("="*80)

print(f"\n上行真实需求 (从7:00开始，共{len(upward_real)}个点):")
print([f'{x:.1f}' for x in upward_real])

print(f"\n上行DRL-TSBC容量 (从7:00开始，共{len(upward_cap)}个点):")
print([f'{c:.1f}' for c in upward_cap])

print(f"\n下行真实需求 (从7:00开始，共{len(downward_real)}个点):")
print([f'{x:.1f}' for x in downward_real])

print(f"\n下行DRL-TSBC容量 (从7:00开始，共{len(downward_cap)}个点):")
print([f'{c:.1f}' for c in downward_cap])

# 保存结果
output_file = f"test_data/{busline}/bidirectional_demand_capacity_{busline}_savedmodel.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("上行真实需求:\n")
    f.write(str(upward_real) + "\n\n")
    f.write("上行DRL-TSBC容量:\n")
    f.write(str(upward_cap) + "\n\n")
    f.write("下行真实需求:\n")
    f.write(str(downward_real) + "\n\n")
    f.write("下行DRL-TSBC容量:\n")
    f.write(str(downward_cap) + "\n\n")

print(f"\n结果已保存到: {output_file}")
print("="*80)
