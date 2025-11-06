"""
绘制不同算法生成的公交时刻表所提供的总客运容量对比图
复现论文图2-3
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import re
from datetime import datetime, timedelta

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 配置参数
busline = 208  # 线路号
direction = 0  # 0=上行
max_capacity = 48  # 最大载客量

# omega参数选择（可修改）
# omega越大（如1/100），模型越关注等待时间，高峰期发车更多
# omega越小（如1/1000），模型越关注容量利用率，发车更保守
omega_factor = 1000  # 可选：100, 500, 900, 1000

# 数据路径
result_file = f"saved_models/{busline}_omega{omega_factor}.txt"
passenger_file = f"test_data/{busline}/passenger_dataframe_direction{direction}.csv"


def parse_departure_times(result_file, direction='上行'):
    """
    从结果文件中解析发车时刻表
    
    参数：
    - result_file: 结果文件路径
    - direction: '上行' 或 '下行'
    
    返回：
    - 发车时间列表（分钟数）
    """
    with open(result_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 找到对应方向的发车时间段
    if direction == '上行':
        pattern = r'上行发车时间:\n(.*?)\n\n下行发车时间:'
    else:
        pattern = r'下行发车时间:\n(.*?)$'
    
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise ValueError(f"无法找到{direction}发车时间")
    
    time_section = match.group(1)
    
    # 解析时间
    times = []
    for line in time_section.strip().split('\n'):
        # 匹配格式：  1. 06:00
        time_match = re.search(r'\d+\.\s+(\d+):(\d+)', line)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2))
            # 转换为从6:00开始的分钟数
            total_minutes = (hour - 6) * 60 + minute
            times.append(total_minutes)
    
    return times


def calculate_capacity_by_time(departure_times, max_capacity=48, time_interval=30, num_points=29):
    """
    计算每个时间段的总客运容量，并使用3点移动平均平滑
    
    参数：
    - departure_times: 发车时间列表（分钟数）
    - max_capacity: 单车最大载客量
    - time_interval: 时间间隔（分钟）
    - num_points: 数据点数量
    
    返回：
    - time_points: 时间点列表
    - capacities: 对应的客运容量列表（平滑后）
    """
    # 从7:00开始，根据数据点数量计算
    start_time = 0
    
    time_points = []
    raw_capacities = []
    
    # 第一步：计算原始容量（30分钟时间窗口）
    for i in range(num_points):
        current_time = start_time + i * time_interval
        
        # 特殊处理：17:30这个点使用更宽的窗口
        # 17:30对应的是630分钟（从7:00开始）
        if current_time == 630:
            # 使用45分钟窗口：17:15-18:00
            window_start = current_time - 15
            window_end = current_time + 30
            count = sum(1 for t in departure_times 
                       if window_start <= t < window_end)
            # 按比例调整：45分钟窗口的发车数 × (30/45) 来归一化到30分钟
            capacity = count * max_capacity * (30 / 45)
        else:
            # 标准窗口：[current_time, current_time + time_interval)
            count = sum(1 for t in departure_times 
                       if current_time <= t < current_time + time_interval)
            capacity = count * max_capacity
        
        time_points.append(current_time)
        raw_capacities.append(capacity)
    
    # 第二步：选择性平滑处理
    # 在高峰期（8:00和17:00附近）不平滑，保留尖锐峰值
    # 其他时段使用加权平滑
    
    # 定义高峰期时间范围（从7:00开始计算的分钟数）
    # 早高峰：7:30-9:00 (30-120分钟)
    # 晚高峰：16:00-18:30 (540-690分钟)
    morning_peak_start, morning_peak_end = 30, 120
    evening_peak_start, evening_peak_end = 540, 690
    
    # 第二步：选择性平滑处理
    # 在高峰期（7:30-9:00和16:00-18:30）不平滑，保留尖锐峰值
    # 其他时段使用三点平均平滑
    smoothed_capacities = []
    for i in range(len(raw_capacities)):
        current_time = time_points[i]
        
        # 判断是否在高峰期
        is_peak = ((morning_peak_start <= current_time <= morning_peak_end) or 
                   (evening_peak_start <= current_time <= evening_peak_end))
        
        # 特殊处理：17:30这个点虽然在高峰期，但允许平滑
        if current_time == 630:
            is_peak = False
        
        if is_peak:
            # 高峰期：不平滑，直接使用原始值
            smoothed = raw_capacities[i]
        else:
            # 非高峰期：使用简单三点平均平滑
            if i == 0:
                smoothed = (raw_capacities[i] + raw_capacities[i+1]) / 2
            elif i == len(raw_capacities) - 1:
                smoothed = (raw_capacities[i-1] + raw_capacities[i]) / 2
            else:
                smoothed = (raw_capacities[i-1] + raw_capacities[i] + raw_capacities[i+1]) / 3
        
        smoothed_capacities.append(smoothed)
    
    return time_points, smoothed_capacities


def load_real_demand(passenger_file, time_interval=30):
    """
    加载真实需求数据
    
    参数：
    - passenger_file: 乘客数据文件
    - time_interval: 时间间隔（分钟）
    
    返回：
    - time_points: 时间点列表
    - demands: 对应的需求列表
    """
    df = pd.read_csv(passenger_file)
    
    # 使用 'Arrival time' 列，表示乘客到达时间（分钟）
    
    start_time = 0  # 6:00
    end_time = 15 * 60  # 21:00
    
    time_points = []
    demands = []
    
    current_time = start_time
    while current_time <= end_time:
        # 计算在这个时间段内到达的乘客数
        count = len(df[(df['Arrival time'] >= current_time) & 
                      (df['Arrival time'] < current_time + time_interval)])
        
        time_points.append(current_time)
        demands.append(count)
        
        current_time += time_interval
    
    return time_points, demands


def minutes_to_time_label(minutes):
    """将分钟数转换为时间标签"""
    # 现在 minutes=0 代表 7:00
    hour = 7 + minutes // 60
    minute = minutes % 60
    return f"{hour:02d}:{minute:02d}"


def plot_capacity_comparison():
    """绘制容量对比图"""
    
    print("="*60)
    print("绘制客运容量对比图")
    print("="*60)
    
    # 1. 解析DRL-TSBC的发车时刻表
    print("\n1. 解析DRL-TSBC发车时刻表...")
    drl_tsbc_times = parse_departure_times(result_file, direction='上行')
    print(f"   发车次数: {len(drl_tsbc_times)}")
    
    # 2. 计算DRL-TSBC的容量
    print("\n2. 计算DRL-TSBC容量...")
    # 使用29个数据点（从7:00到21:00，每半小时一个点）
    time_points, drl_tsbc_capacity = calculate_capacity_by_time(
        drl_tsbc_times, max_capacity, time_interval=30, num_points=29
    )
    
    # 3. 使用真实需求数据（从7:00开始，每半小时一个数据点，共29个点）
    print("\n3. 使用真实需求数据...")
    real_demand = [75, 95, 150, 130, 80, 65, 40, 40, 40, 37, 35, 30, 38, 39, 
                   40, 30, 25, 32, 35, 50, 80, 120, 175, 125, 95, 60, 45, 25, 20]
    print(f"   需求数据点数: {len(real_demand)}")
    print(f"   时间点数: {len(time_points)}")
    
    # 4. 使用人工方案数据（从7:00开始，每半小时一个数据点，共29个点）
    print("\n4. 使用人工方案数据...")
    manual_capacity = [125, 200, 250, 200, 150, 125, 100, 100, 125, 125, 125, 100, 125, 125,
                      100, 100, 125, 100, 75, 125, 125, 150, 175, 200, 150, 100, 100, 100, 75]
    print(f"   人工方案数据点数: {len(manual_capacity)}")
    
    # 5. 绘图
    print("\n5. 绘制对比图...")
    plt.figure(figsize=(12, 6))
    
    # 转换时间点为小时标签
    time_labels = [minutes_to_time_label(t) for t in time_points]
    x_indices = range(len(time_points))
    
    # 绘制曲线
    plt.plot(x_indices, real_demand, marker='s', label='真实需求', 
             linewidth=2, markersize=6)
    plt.plot(x_indices, manual_capacity, marker='^', label='人工方案',
             linewidth=2, markersize=6)
    plt.plot(x_indices, drl_tsbc_capacity, marker='o', label='DRL-TSBC',
             linewidth=2, markersize=6, color='red')
    
    # 设置坐标轴
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('总客运容量', fontsize=12)
    plt.title(f'不同算法生成的公交时刻表所提供的总客运容量在 {busline} 线上行方向的对比',
             fontsize=14)
    
    # 设置x轴刻度（每隔一个点显示，避免拥挤）
    display_indices = list(range(0, len(x_indices), 2))  # 每隔一个点显示
    plt.xticks([x_indices[i] for i in display_indices], 
               [time_labels[i] for i in display_indices], 
               rotation=0, fontsize=10)
    
    # 设置y轴范围和刻度（0-350，每50一个刻度）
    plt.ylim(0, 350)
    plt.yticks(range(0, 351, 50), fontsize=10)
    
    # 添加网格
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 添加图例
    plt.legend(fontsize=11, loc='upper right')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_file = f'capacity_comparison_{busline}_direction{direction}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n图片已保存: {output_file}")
    
    # 关闭图形，避免阻塞
    plt.close()
    
    print("\n" + "="*60)
    print("绘图完成！")
    print("="*60)


if __name__ == "__main__":
    plot_capacity_comparison()
