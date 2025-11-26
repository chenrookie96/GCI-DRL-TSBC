"""
图2-5推理脚本
对比调整前（原始数据）和调整后（突发客流数据）的DRL-TSBC表现
"""

import pandas as pd
import numpy as np
import torch
import os
import sys
sys.path.insert(0, 'drl_tsbc_code')

from drl_tsbc_brain import DQN
from drl_tsbc_environment import (
    Station, BidirectionalBusSystem,
    first_time, last_time, min_Interval, max_Interval,
    device,
    start_time, end_time, ideal_interval, avg_flag
)
from data_loader import BusDataLoader

# 线路配置
busline = 208
direction_up = 0
direction_down = 1
omega_factor = 1000
omega = 1 / omega_factor

# 数据路径
data_dir = f"./test_data/{busline}"
model_load_path = f"./saved_models/{busline}_omega{omega_factor}.pth"

# 最大载客量
pn_on_max = 48


def adjust_schedule(departure_times_up, departure_times_down, Tmax):
    """后处理算法：调整时刻表使上下行发车次数相等"""
    up_count = len(departure_times_up)
    down_count = len(departure_times_down)
    
    if up_count == down_count:
        print("上下行发车次数已相等，无需调整")
        return departure_times_up, departure_times_down
    
    if up_count > down_count:
        print(f"调整上行时刻表（从 {up_count} 调整到 {down_count} 次）")
        times_to_adjust = departure_times_up.copy()
        direction = "upward"
    else:
        print(f"调整下行时刻表（从 {down_count} 调整到 {up_count} 次）")
        times_to_adjust = departure_times_down.copy()
        direction = "downward"
    
    if len(times_to_adjust) >= 2:
        del times_to_adjust[-2]
    
    k = len(times_to_adjust) - 1
    while k > 0:
        interval = times_to_adjust[k] - times_to_adjust[k-1]
        if interval > Tmax:
            times_to_adjust[k-1] = times_to_adjust[k] - Tmax
            k -= 1
        else:
            break
    
    if direction == "upward":
        return times_to_adjust, departure_times_down
    else:
        return departure_times_up, times_to_adjust


def run_inference(use_surge_data=False):
    """
    运行推理
    use_surge_data: True使用突发客流数据，False使用原始数据
    """
    data_type = "调整后（突发客流）" if use_surge_data else "调整前（原始数据）"
    
    print("\n" + "="*80)
    print(f"DRL-TSBC 推理 - {data_type}")
    print("="*80)
    
    # 选择数据文件
    if use_surge_data:
        passenger_info_path_up = f"{data_dir}/passenger_dataframe_direction{direction_up}_surge.csv"
    else:
        passenger_info_path_up = f"{data_dir}/passenger_dataframe_direction{direction_up}.csv"
    
    passenger_info_path_down = f"{data_dir}/passenger_dataframe_direction{direction_down}.csv"
    
    # 检查文件
    if not os.path.exists(passenger_info_path_up):
        print(f"错误: 找不到数据文件: {passenger_info_path_up}")
        return None
    
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
    start_minute_th = (int(start_time[:-3]) - int(trf_con_up.iloc[0, 0])) * 60 + (
        int(start_time[-2:]) - int(trf_con_up.iloc[0, 1])
    )
    end_minute_th = (int(end_time[:-3]) - int(trf_con_up.iloc[0, 0])) * 60 + (
        int(end_time[-2:]) - int(trf_con_up.iloc[0, 1])
    )
    
    # 加载模型
    print(f"加载模型: {model_load_path}")
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
    
    print("开始推理...")
    
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
    
    # 后处理：确保上下行发车次数相等
    print("\n后处理：调整时刻表")
    departure_times_up, departure_times_down = adjust_schedule(
        departure_times_up, departure_times_down, max_Interval
    )
    
    # 计算统计
    total_minutes = last_minute_th - first_minute_th
    awt_up = total_wait_up / total_minutes / station_num_up if total_minutes > 0 else 0
    awt_down = total_wait_down / total_minutes / station_num_down if total_minutes > 0 else 0
    
    print(f"推理完成 - 上行{len(departure_times_up)}次, 下行{len(departure_times_down)}次")
    
    return {
        'departure_times_up': departure_times_up,
        'departure_times_down': departure_times_down,
        'awt_up': awt_up,
        'awt_down': awt_down,
        'cant_taken_up': int(cant_taken_up),
        'cant_taken_down': int(cant_taken_down),
        'station_num_up': station_num_up,
        'station_num_down': station_num_down
    }


if __name__ == "__main__":
    # 运行原始数据推理
    result_original = run_inference(use_surge_data=False)
    
    # 运行突发客流数据推理
    result_surge = run_inference(use_surge_data=True)
    
    if result_original and result_surge:
        # 输出结果
        print("\n" + "="*80)
        print("推理结果汇总")
        print("="*80)
        
        print(f"\n{'场景':<15} {'方向':<10} {'指标':<25} {'结果':<15}")
        print("-" * 70)
        
        # 调整前
        print(f"{'调整前':<15} {'上行':<10} {'发车次数':<25} {len(result_original['departure_times_up']):<15}")
        print(f"{'':<15} {'':<10} {'乘客平均等待时间 (m)':<25} {result_original['awt_up']:<15.2f}")
        print(f"{'':<15} {'':<10} {'被滞留乘客数量':<25} {result_original['cant_taken_up']:<15}")
        print(f"{'':<15} {'下行':<10} {'发车次数':<25} {len(result_original['departure_times_down']):<15}")
        print(f"{'':<15} {'':<10} {'乘客平均等待时间 (m)':<25} {result_original['awt_down']:<15.2f}")
        print(f"{'':<15} {'':<10} {'被滞留乘客数量':<25} {result_original['cant_taken_down']:<15}")
        print("-" * 70)
        
        # 调整后
        print(f"{'调整后':<15} {'上行':<10} {'发车次数':<25} {len(result_surge['departure_times_up']):<15}")
        print(f"{'':<15} {'':<10} {'乘客平均等待时间 (m)':<25} {result_surge['awt_up']:<15.2f}")
        print(f"{'':<15} {'':<10} {'被滞留乘客数量':<25} {result_surge['cant_taken_up']:<15}")
        print(f"{'':<15} {'下行':<10} {'发车次数':<25} {len(result_surge['departure_times_down']):<15}")
        print(f"{'':<15} {'':<10} {'乘客平均等待时间 (m)':<25} {result_surge['awt_down']:<15.2f}")
        print(f"{'':<15} {'':<10} {'被滞留乘客数量':<25} {result_surge['cant_taken_down']:<15}")
        print("=" * 70)
        
        # 保存结果
        output_file = f"saved_models/{busline}_omega{omega_factor}_surge.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("图2-5推理结果\n")
            f.write("="*80 + "\n\n")
            
            # 调整前
            f.write("【调整前（原始数据）】\n")
            f.write(f"上行发车次数: {len(result_original['departure_times_up'])}\n")
            f.write(f"上行乘客平均等待时间: {result_original['awt_up']:.2f} 分钟\n")
            f.write(f"上行被滞留乘客数量: {result_original['cant_taken_up']}\n")
            f.write(f"下行发车次数: {len(result_original['departure_times_down'])}\n")
            f.write(f"下行乘客平均等待时间: {result_original['awt_down']:.2f} 分钟\n")
            f.write(f"下行被滞留乘客数量: {result_original['cant_taken_down']}\n")
            f.write("\n上行发车时间:\n")
            for i, t in enumerate(result_original['departure_times_up']):
                f.write(f"  {i+1}. {t//60:02d}:{t%60:02d}\n")
            f.write("\n下行发车时间:\n")
            for i, t in enumerate(result_original['departure_times_down']):
                f.write(f"  {i+1}. {t//60:02d}:{t%60:02d}\n")
            
            f.write("\n" + "="*80 + "\n\n")
            
            # 调整后
            f.write("【调整后（突发客流）】\n")
            f.write(f"上行发车次数: {len(result_surge['departure_times_up'])}\n")
            f.write(f"上行乘客平均等待时间: {result_surge['awt_up']:.2f} 分钟\n")
            f.write(f"上行被滞留乘客数量: {result_surge['cant_taken_up']}\n")
            f.write(f"下行发车次数: {len(result_surge['departure_times_down'])}\n")
            f.write(f"下行乘客平均等待时间: {result_surge['awt_down']:.2f} 分钟\n")
            f.write(f"下行被滞留乘客数量: {result_surge['cant_taken_down']}\n")
            f.write("\n上行发车时间:\n")
            for i, t in enumerate(result_surge['departure_times_up']):
                f.write(f"  {i+1}. {t//60:02d}:{t%60:02d}\n")
            f.write("\n下行发车时间:\n")
            for i, t in enumerate(result_surge['departure_times_down']):
                f.write(f"  {i+1}. {t//60:02d}:{t%60:02d}\n")
        
        print(f"\n结果已保存: {output_file}")
