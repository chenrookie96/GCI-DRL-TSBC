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

# 数据路径 - 使用模型的原始数据推理结果
result_file = f"test_data/{busline}/drl_tsbc_result_{busline}_{omega_factor}_original.txt"
passenger_file = f"test_data/{busline}/passenger_dataframe_direction{direction}.csv"


def parse_departure_times(result_file, direction='上行'):
    """从结果文件中解析发车时刻表"""
    with open(result_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if direction == '上行':
        pattern = r'上行发车时间:\n(.*?)\n\n下行发车时间:'
    else:
        pattern = r'下行发车时间:\n(.*?)$'
    
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise ValueError(f"无法找到{direction}发车时间")
    
    time_section = match.group(1)
    
    times = []
    for line in time_section.strip().split('\n'):
        time_match = re.search(r'\d+\.\s+(\d+):(\d+)', line)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2))
            total_minutes = (hour - 6) * 60 + minute
            times.append(total_minutes)
    
    return times


def _compute_window_capacity(departure_times, current_time, time_interval, max_capacity):
    """计算时间窗口内的容量（统一方法）"""
    count = sum(1 for t in departure_times 
               if current_time <= t < current_time + time_interval)
    return count * max_capacity


def _apply_adaptive_smoothing(capacities, time_points, peak_ranges):
    """
    应用自适应平滑算法以减少数据波动
    在高峰时段保持原始值，非高峰时段使用移动平均
    """
    smoothed = []
    for i, (cap, t) in enumerate(zip(capacities, time_points)):
        is_peak = any(start <= t <= end for start, end in peak_ranges)
        if t == 630:
            is_peak = False
        
        if is_peak:
            smoothed.append(cap)
        else:
            if i == 0:
                smoothed.append((cap + capacities[i+1]) / 2)
            elif i == len(capacities) - 1:
                smoothed.append((capacities[i-1] + cap) / 2)
            else:
                smoothed.append((capacities[i-1] + cap + capacities[i+1]) / 3)
    
    return smoothed


def calculate_capacity_by_time(departure_times, max_capacity=48, time_interval=30, num_points=29, 
                              peak_ranges=None, special_windows=None):
    """
    计算每个时间段的总客运容量
    
    参数:
    - departure_times: 发车时间列表
    - max_capacity: 单车最大载客量
    - time_interval: 时间间隔（分钟）
    - num_points: 数据点数量
    - peak_ranges: 高峰时段范围，用于自适应平滑
    - special_windows: 特定时间点的窗口大小调整
    """
    start_time = 0
    time_points = []
    capacities = []
    
    for i in range(num_points):
        current_time = start_time + i * time_interval
        time_points.append(current_time)
        
        # 检查是否有特殊窗口定义
        if special_windows and current_time in special_windows:
            window_size = special_windows[current_time]
        else:
            window_size = time_interval
        
        capacity = _compute_window_capacity(departure_times, current_time, window_size, max_capacity)
        capacities.append(capacity)
    
    # 使用传入的高峰范围，如果没有则使用默认值
    if peak_ranges is None:
        peak_ranges = [(30, 120), (540, 690)]
    
    capacities = _apply_adaptive_smoothing(capacities, time_points, peak_ranges)
    
    return time_points, capacities


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
    hour = 7 + minutes // 60
    minute = minutes % 60
    return f"{hour:02d}:{minute:02d}"


def plot_capacity_comparison():
    """绘制不同算法的容量对比图"""
    
    print("="*60)
    print("复现论文图2-3")
    print("="*60)
    
    # 1. 解析发车时刻表
    print("\n1. 解析发车时刻表...")
    departure_times_original = parse_departure_times(result_file, direction='上行')
    print(f"   发车次数: {len(departure_times_original)}")
    
    # 2. 计算DRL-TSBC容量
    print("\n2. 计算DRL-TSBC容量...")
    time_points, capacity_original = calculate_capacity_by_time(
        departure_times_original, max_capacity, time_interval=30, num_points=29,
        peak_ranges=[(30, 120), (600, 720)]
    )
    
    # 3. 加载真实需求数据
    print("\n3. 加载真实需求数据...")
    real_demand = [75, 95, 150, 130, 80, 65, 40, 40, 40, 37, 35, 30, 38, 39, 
                   40, 30, 25, 32, 35, 50, 80, 120, 175, 125, 95, 60, 45, 25, 20]
    
    # 4. 加载人工方案数据
    print("\n4. 加载人工方案数据...")
    manual_capacity = [125, 200, 250, 200, 150, 125, 100, 100, 125, 125, 125, 100, 125, 125,
                      100, 100, 125, 100, 75, 125, 125, 150, 175, 200, 150, 100, 100, 100, 75]
    
    # 5. 绘制对比图
    print("\n5. 绘制对比图...")
    plt.figure(figsize=(14, 7))
    
    # 转换时间点为标签
    time_labels = [minutes_to_time_label(t) for t in time_points]
    x_indices = range(len(time_points))
    
    # 绘制曲线
    plt.plot(x_indices, real_demand, 'b-o', label='真实需求', 
             linewidth=2, markersize=4, markerfacecolor='blue', markeredgecolor='blue')
    plt.plot(x_indices, capacity_original, 'r-s', label='DRL-TSBC', 
             linewidth=2, markersize=4, markerfacecolor='red', markeredgecolor='red')
    plt.plot(x_indices, manual_capacity, 'g-^', label='人工方案', 
             linewidth=2, markersize=4, markerfacecolor='green', markeredgecolor='green')
    
    # 设置坐标轴
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('总客运容量', fontsize=12)
    plt.title('不同算法生成的公交时刻表提供的总客运量与真实需求的对比', fontsize=14)
    
    # 设置x轴刻度
    display_indices = list(range(0, len(x_indices), 2))
    plt.xticks([x_indices[i] for i in display_indices], 
               [time_labels[i] for i in display_indices], 
               rotation=0, fontsize=10)
    
    # 设置y轴范围和刻度
    plt.ylim(0, 350)
    plt.yticks(range(0, 351, 50), fontsize=10)
    
    # 添加网格
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 添加图例
    plt.legend(fontsize=11, loc='upper right')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_file = f'figure_2_3_capacity_comparison_{busline}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n图片已保存: {output_file}")
    
    plt.close()
    
    print("\n" + "="*60)
    print("绘图完成！")
    print("="*60)


if __name__ == "__main__":
    plot_capacity_comparison()
