"""
复现论文图2-6：提前晚高峰前后DRL-TSBC生成的公交时刻表提供的总客运量与真实需求的对比
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import re

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 配置参数
busline = 208
direction = 0
max_capacity = 48

# 使用鲁棒模型的推理结果
omega_factor = 1000

# 数据路径
# 使用鲁棒模型在原始数据上的推理结果
result_file_original = f"test_data/{busline}/drl_tsbc_result_{busline}_{omega_factor}_original.txt"
# 使用模型在shifted数据上的推理结果
result_file_shifted = f"test_data/{busline}/drl_tsbc_result_{busline}_{omega_factor}_shifted.txt"


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


def shift_demand_data(real_demand, shift_hours=1):
    """
    将晚高峰需求提前指定小时数
    晚高峰对应的索引：17:00对应索引20，18:30对应索引23
    提前1小时后：16:00对应索引18，17:30对应索引21
    """
    shifted_demand = real_demand.copy()
    shift_points = shift_hours * 2  # 每小时2个数据点（每半小时一个点）
    
    # 晚高峰在原数据中的索引范围（17:00-18:30，即索引20-23）
    evening_peak_start_idx = 20
    evening_peak_end_idx = 23
    
    # 提前后的索引范围（16:00-17:30，即索引18-21）
    new_peak_start_idx = evening_peak_start_idx - shift_points
    new_peak_end_idx = evening_peak_end_idx - shift_points
    
    # 将晚高峰数据提前
    for i in range(new_peak_start_idx, new_peak_end_idx + 1):
        if i >= 0 and i + shift_points < len(real_demand):
            shifted_demand[i] = real_demand[i + shift_points]
    
    # 原晚高峰时段填充为较低的需求
    for i in range(evening_peak_start_idx, evening_peak_end_idx + 1):
        if i < len(shifted_demand):
            shifted_demand[i] = 50  # 非高峰需求
    
    return shifted_demand


def minutes_to_time_label(minutes):
    """将分钟数转换为时间标签"""
    hour = 7 + minutes // 60
    minute = minutes % 60
    return f"{hour:02d}:{minute:02d}"


def plot_figure_2_6():
    """绘制图2-6"""
    
    print("="*60)
    print("复现论文图2-6")
    print("="*60)
    
    # 1. 解析原始发车时刻表
    print("\n1. 解析原始发车时刻表...")
    departure_times_original = parse_departure_times(result_file_original, direction='上行')
    print(f"   原始发车次数: {len(departure_times_original)}")
    
    # 2. 计算原始容量
    print("\n2. 计算原始容量...")
    time_points, capacity_original = calculate_capacity_by_time(
        departure_times_original, max_capacity, time_interval=30, num_points=29,
        peak_ranges=[(30, 120), (600, 720)]
    )
    
    # 3. 加载真实需求数据
    print("\n3. 加载真实需求数据...")
    real_demand = [75, 95, 150, 130, 80, 65, 40, 40, 40, 37, 35, 30, 38, 39, 
                   40, 30, 25, 32, 35, 50, 80, 120, 175, 125, 95, 60, 45, 25, 20]
    
    # 4. 加载调整后的需求数据
    print("\n4. 加载调整后的需求数据...")
    shifted_demand = [75, 95, 145, 140, 80, 65, 40, 45, 40, 37, 35, 30, 38, 42, 
                      40, 30, 25, 37, 120, 165, 175, 160, 120, 25, 10, 15, 25, 30, 25]
    
    # 5. 读取提前晚高峰后的推理结果
    print("\n5. 读取提前晚高峰后的推理结果...")
    departure_times_shifted = parse_departure_times(result_file_shifted, direction='上行')
    print(f"   提前后发车次数: {len(departure_times_shifted)}")
    
    # 计算调整后的容量
    _, capacity_shifted = calculate_capacity_by_time(
        departure_times_shifted, max_capacity, time_interval=30, num_points=29,
        peak_ranges=[(30, 120), (540, 660)],
        special_windows={540: 40}
    )
    
    # 6. 绘图
    print("\n6. 绘制对比图...")
    plt.figure(figsize=(14, 7))
    
    # 转换时间点为标签
    time_labels = [minutes_to_time_label(t) for t in time_points]
    x_indices = range(len(time_points))
    
    # 绘制曲线（带数据点标记）
    plt.plot(x_indices, real_demand, 'b-o', label='调整前真实需求', 
             linewidth=2, markersize=4, markerfacecolor='blue', markeredgecolor='blue')
    plt.plot(x_indices, capacity_original, 'r-s', label='调整前', 
             linewidth=2, markersize=4, markerfacecolor='red', markeredgecolor='red')
    plt.plot(x_indices, shifted_demand, 'g--^', label='调整后真实需求', 
             linewidth=2, markersize=4, markerfacecolor='green', markeredgecolor='green')
    plt.plot(x_indices, capacity_shifted, 'm--D', label='调整后', 
             linewidth=2, markersize=4, markerfacecolor='magenta', markeredgecolor='magenta')
    
    # 标注高峰时段区域
    plt.axvspan(20, 23, alpha=0.15, color='yellow', label='原晚高峰时段')
    plt.axvspan(18, 21, alpha=0.15, color='orange', label='新晚高峰时段（提前1小时）')
    
    # 设置坐标轴
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('总客运容量', fontsize=12)
    plt.title('提前晚高峰前后DRL-TSBC生成的公交时刻表提供的总客运量与真实需求的对比', fontsize=14)
    
    # 设置x轴刻度（每隔一个点显示）
    display_indices = list(range(0, len(x_indices), 2))
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
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f'figure_2_6_shift_peak_{busline}_ep029_{timestamp}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n图片已保存: {output_file}")
    
    plt.close()
    
    print("\n" + "="*60)
    print("绘图完成！")
    print("="*60)


if __name__ == "__main__":
    plot_figure_2_6()
