"""
图2-4仿真脚本
DRL-TSBC在208线上下行方向生成的公交时刻表所提供的总客运容量与真实需求的对比
"""

import sys
sys.path.insert(0, 'drl_tsbc_code')

import pandas as pd
import numpy as np
import re
import os
from drl_tsbc_environment import Station, DirectionSystem

# 配置参数
busline = 208
max_capacity = 48
omega_factor = 1000

# 时间设置
first_minute_th = 360  # 6:00
last_minute_th = 1260  # 21:00

print("="*80)
print("图2-4仿真：208线上下行总客运容量与真实需求对比")
print("="*80)

def parse_departure_times_from_original(result_file, direction_name):
    """从原始推理结果解析发车时刻表"""
    with open(result_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 找到调整前部分
    section_pattern = r'【调整前（原始数据）】(.*?)【调整后'
    section_match = re.search(section_pattern, content, re.DOTALL)
    if not section_match:
        # 尝试其他格式
        section_pattern = r'【调整前（原始数据）】(.*?)$'
        section_match = re.search(section_pattern, content, re.DOTALL)
    
    if not section_match:
        raise ValueError("无法找到调整前部分")
    
    section_content = section_match.group(1)
    
    # 找到对应方向的发车时间
    if direction_name == "上行":
        pattern = r'上行发车时间:\n(.*?)\n\n下行发车时间:'
    else:
        pattern = r'下行发车时间:\n(.*?)(?:\n\n|$)'
    
    match = re.search(pattern, section_content, re.DOTALL)
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

def simulate_direction(direction, departure_times):
    """模拟单个方向，计算总客运容量和真实需求"""
    direction_name = '上行' if direction == 0 else '下行'
    
    print(f"\n{'='*80}")
    print(f"模拟: {direction_name}")
    print(f"{'='*80}")
    
    # 数据路径
    passenger_info_path = f"test_data/{busline}/passenger_dataframe_direction{direction}.csv"
    trf_path = f"test_data/{busline}/traffic-{direction}.csv"
    
    # 读取数据
    trf_con = pd.read_csv(trf_path)
    passenger_df_temp = pd.read_csv(passenger_info_path)
    station_num = passenger_df_temp['Boarding station'].max() + 1
    
    print(f"  站点数: {station_num}")
    print(f"  发车次数: {len(departure_times)}")
    print(f"  首班车: {departure_times[0]//60:02d}:{departure_times[0]%60:02d}")
    print(f"  末班车: {departure_times[-1]//60:02d}:{departure_times[-1]%60:02d}")
    
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
        right_minute_th += 1
    
    print("-"*80)
    
    # 提取从7:00开始的数据（跳过6:00-7:00的2个点）
    real_need_list_from_7 = real_need_list[2:31]
    Cap_list_from_7 = Cap_list[2:31]
    
    return real_need_list_from_7, Cap_list_from_7


if __name__ == "__main__":
    result_file = f"saved_models/{busline}_omega{omega_factor}_surge.txt"
    
    if not os.path.exists(result_file):
        print(f"错误: 找不到推理结果文件: {result_file}")
        exit(1)
    
    # 解析上下行发车时刻表（使用调整前的原始数据结果）
    print("\n读取发车时刻表（原始数据推理结果）...")
    up_times = parse_departure_times_from_original(result_file, "上行")
    down_times = parse_departure_times_from_original(result_file, "下行")
    print(f"上行: {len(up_times)}次, 下行: {len(down_times)}次")
    
    # 仿真上行
    up_real, up_cap = simulate_direction(0, up_times)
    
    # 仿真下行
    down_real, down_cap = simulate_direction(1, down_times)
    
    # 打印结果汇总
    print("\n" + "="*80)
    print("仿真结果汇总")
    print("="*80)
    
    print(f"\n上行容量 (共{len(up_cap)}个点):")
    print([f'{x:.1f}' for x in up_cap])
    
    print(f"\n上行真实需求:")
    print([f'{x:.1f}' for x in up_real])
    
    print(f"\n下行容量 (共{len(down_cap)}个点):")
    print([f'{x:.1f}' for x in down_cap])
    
    print(f"\n下行真实需求:")
    print([f'{x:.1f}' for x in down_real])
    
    # 保存结果
    output_file = f"test_data/{busline}/figure_2_4_data.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("图2-4数据\n")
        f.write("="*60 + "\n\n")
        f.write("上行容量:\n")
        f.write(str([float(x) for x in up_cap]) + "\n\n")
        f.write("上行真实需求:\n")
        f.write(str([float(x) for x in up_real]) + "\n\n")
        f.write("下行容量:\n")
        f.write(str([float(x) for x in down_cap]) + "\n\n")
        f.write("下行真实需求:\n")
        f.write(str([float(x) for x in down_real]) + "\n\n")
    
    print(f"\n数据已保存: {output_file}")
    print("="*80)
