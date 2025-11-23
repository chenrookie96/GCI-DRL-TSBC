"""
使用saved_models中的shifted推理结果进行模拟
计算调整后的真实需求和DRL-TSBC容量
用于绘制图2-6
"""

import sys
sys.path.insert(0, 'drl_tsbc_code')

import pandas as pd
import numpy as np
import re
from drl_tsbc_environment import Station, DirectionSystem

# 配置参数
busline = 208
direction = 0
max_capacity = 48
omega_factor = 1000

# 数据路径 - 使用shifted数据
passenger_info_path = f"test_data/{busline}/passenger_dataframe_direction{direction}_shifted.csv"
trf_path = f"test_data/{busline}/traffic-{direction}.csv"

print("="*80)
print("使用saved_models推理结果进行模拟（晚高峰提前数据）")
print("="*80)

# 读取数据
trf_con = pd.read_csv(trf_path)
passenger_df_temp = pd.read_csv(passenger_info_path)
station_num = passenger_df_temp['Boarding station'].max() + 1

# 时间设置
first_minute_th = 360
last_minute_th = 1260

print(f"\n配置:")
print(f"  线路: {busline}")
print(f"  方向: {direction}")
print(f"  站点数: {station_num}")
print(f"  最大容量: {max_capacity}")
print(f"  模拟时间: {first_minute_th//60}:00 到 {last_minute_th//60}:00")
print(f"  数据类型: 晚高峰提前1小时")

# 解析发车时刻表
def parse_departure_times(result_file):
    """解析发车时刻表"""
    with open(result_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pattern = r'上行发车时间:\n(.*?)\n\n下行发车时间:'
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        raise ValueError("无法找到上行发车时间")
    
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

# 读取发车时刻表 - 使用saved_models的shifted结果
print("\n读取发车时刻表...")
result_file = f"saved_models/{busline}_omega{omega_factor}_shifted.txt"
departure_times = parse_departure_times(result_file)
print(f"  发车次数: {len(departure_times)}")
print(f"  首班车: {departure_times[0]//60}:{departure_times[0]%60:02d}")
print(f"  末班车: {departure_times[-1]//60}:{departure_times[-1]%60:02d}")

# 初始化环境
print("\n初始化环境...")
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
    if departure_index < len(departure_times) and right_minute_th == departure_times[departure_index]:
        system.Departure(right_minute_th)
        departure_index += 1
    
    bus_on_line_now = []
    for bus in system.bus_online:
        if bus.arrv_mark != 1:
            bus_on_line_now.append(len(bus.pn_on))
    
    Cap_half_hour += system.All_cap_take / (station_num - 1)
    real_need_half_hour += sum(bus_on_line_now)
    
    if right_minute_th <= last_minute_th and right_minute_th % 30 == 0:
        Cap_half_hour = Cap_half_hour / 30
        Cap_list.append(Cap_half_hour)
        Cap_half_hour = 0
        
        real_need_half_hour = real_need_half_hour / 30
        real_need_list.append(real_need_half_hour)
        real_need_half_hour = 0
        
        hour = right_minute_th // 60
        minute = right_minute_th % 60
        active_buses = len([b for b in system.bus_online if b.arrv_mark != 1])
        print(f"  {hour:02d}:{minute:02d} - 活跃车辆: {active_buses:2d}, "
              f"真实需求: {real_need_list[-1]:6.2f}, 容量: {Cap_list[-1]:6.2f}")
    
    system.step_forward(right_minute_th)
    right_minute_th += 1

print("-"*80)
print("\n模拟完成！")
print("="*80)

# 提取从7:00开始的数据
real_need_list_from_7 = real_need_list[2:31]
Cap_list_from_7 = Cap_list[2:31]

print(f"\n真实需求（晚高峰提前，saved_model） (从7:00开始，共{len(real_need_list_from_7)}个点):")
print([f'{x:.1f}' for x in real_need_list_from_7])

print(f"\nDRL-TSBC容量（晚高峰提前，saved_model） (从7:00开始，共{len(Cap_list_from_7)}个点):")
print([f'{c:.1f}' for c in Cap_list_from_7])

# 保存结果
output_file = f"test_data/{busline}/simulated_demand_capacity_{busline}_shifted_savedmodel.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("真实需求（晚高峰提前，saved_model）:\n")
    f.write(str(real_need_list_from_7) + "\n\n")
    f.write("DRL-TSBC容量（晚高峰提前，saved_model）:\n")
    f.write(str(Cap_list_from_7) + "\n\n")

print(f"\n结果已保存到: {output_file}")
print("="*80)
