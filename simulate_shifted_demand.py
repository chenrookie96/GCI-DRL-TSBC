"""
使用晚高峰提前的数据进行模拟
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
print("使用晚高峰提前数据进行模拟")
print("="*80)

# 读取数据
trf_con = pd.read_csv(trf_path)
passenger_df_temp = pd.read_csv(passenger_info_path)
station_num = passenger_df_temp['Boarding station'].max() + 1

# 时间设置（与原始数据相同）
first_minute_th = 360  # 从6:00开始（从午夜算起360分钟）
last_minute_th = 1260  # 到21:00（从午夜算起1260分钟）

print(f"\n配置:")
print(f"  线路: {busline}")
print(f"  方向: {direction}")
print(f"  站点数: {station_num}")
print(f"  最大容量: {max_capacity}")
print(f"  模拟时间: {first_minute_th//60}:00 到 {last_minute_th//60}:00")
print(f"  数据类型: 晚高峰提前1小时")

# 解析发车时刻表 - 使用shifted结果
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
            # 转换为从0:00（午夜）起算的分钟数
            total_minutes = hour * 60 + minute
            departure_times.append(total_minutes)
    
    return departure_times


# 读取发车时刻表 - 使用shifted结果
print("\n读取发车时刻表...")
result_file = f"test_data/{busline}/drl_tsbc_result_{busline}_{omega_factor}_shifted.txt"
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
print("\n模拟完成！")
print("="*80)

# 提取从7:00开始的数据
real_need_list_from_7 = real_need_list[2:31]  # 取29个点
Cap_list_from_7 = Cap_list[2:31]

print(f"\n真实需求（晚高峰提前） (从7:00开始，共{len(real_need_list_from_7)}个点):")
print([f'{x:.1f}' for x in real_need_list_from_7])

print(f"\nDRL-TSBC容量（晚高峰提前） (从7:00开始，共{len(Cap_list_from_7)}个点):")
print([f'{c:.1f}' for c in Cap_list_from_7])

# 论文中的晚高峰提前数据（图2-6下图）
paper_demand_shifted = [75, 95, 145, 140, 80, 65, 40, 45, 40, 37, 35, 30, 38, 42, 
                        40, 30, 25, 37, 120, 165, 175, 160, 120, 25, 10, 15, 25, 30, 25]
print(f"\n论文真实需求（晚高峰提前） (共{len(paper_demand_shifted)}个点):")
print(paper_demand_shifted)

# 对比分析
print("\n" + "="*80)
print("数据对比分析")
print("="*80)

# 计算相关性
import numpy as np
correlation = np.corrcoef(real_need_list_from_7, paper_demand_shifted)[0, 1]
print(f"\n与论文数据的相关系数: {correlation:.4f}")

# 找出高峰时段
sim_peak_idx = real_need_list_from_7.index(max(real_need_list_from_7))
paper_peak_idx = paper_demand_shifted.index(max(paper_demand_shifted))
print(f"\n模拟数据高峰: 索引{sim_peak_idx} (时间: {7 + sim_peak_idx*0.5:.1f}:00), 需求: {real_need_list_from_7[sim_peak_idx]:.1f}")
print(f"论文数据高峰: 索引{paper_peak_idx} (时间: {7 + paper_peak_idx*0.5:.1f}:00), 需求: {paper_demand_shifted[paper_peak_idx]:.1f}")

# 验证晚高峰是否提前
print("\n晚高峰提前验证:")
print("  原始晚高峰应在: 17:00-18:30 (索引20-23)")
print("  提前后应在: 16:00-17:30 (索引18-21)")
print(f"  模拟数据高峰在索引{sim_peak_idx}: {'✓ 正确' if 18 <= sim_peak_idx <= 21 else '✗ 错误'}")
print(f"  论文数据高峰在索引{paper_peak_idx}: {'✓ 正确' if 18 <= paper_peak_idx <= 21 else '✗ 错误'}")

# 保存结果
output_file = f"test_data/{busline}/simulated_demand_capacity_{busline}_shifted.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("真实需求（晚高峰提前）:\n")
    f.write(str(real_need_list_from_7) + "\n\n")
    f.write("DRL-TSBC容量（晚高峰提前）:\n")
    f.write(str(Cap_list_from_7) + "\n\n")
    f.write("论文真实需求（晚高峰提前）:\n")
    f.write(str(paper_demand_shifted) + "\n\n")
    f.write(f"相关系数: {correlation:.4f}\n")

print(f"\n结果已保存到: {output_file}")
print("="*80)
