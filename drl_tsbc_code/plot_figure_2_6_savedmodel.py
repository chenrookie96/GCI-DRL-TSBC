"""
使用saved_model的数据绘制图2-6：晚高峰提前实验（调整前后对比）
"""

import matplotlib.pyplot as plt
from matplotlib import rcParams
import re

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 配置参数
busline = 208

# 数据文件路径
original_data_file = f"test_data/{busline}/simulated_demand_capacity_{busline}_savedmodel.txt"
shifted_data_file = f"test_data/{busline}/simulated_demand_capacity_{busline}_shifted_savedmodel.txt"


def load_simulated_data(file_path):
    """从模拟结果文件中加载数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 解析真实需求
    demand_match = re.search(r'真实需求.*?:\n\[(.*?)\]', content, re.DOTALL)
    if not demand_match:
        raise ValueError(f"无法找到真实需求数据: {file_path}")
    
    demand_str = demand_match.group(1)
    real_demand = [float(x.strip().strip("'")) for x in demand_str.split(',')]
    
    # 解析DRL-TSBC容量
    capacity_match = re.search(r'DRL-TSBC容量.*?:\n\[(.*?)\]', content, re.DOTALL)
    if not capacity_match:
        raise ValueError(f"无法找到DRL-TSBC容量数据: {file_path}")
    
    capacity_str = capacity_match.group(1)
    capacity_values = []
    matches = re.findall(r'np\.float64\(([\d.]+)\)', capacity_str)
    if matches:
        capacity_values = [float(x) for x in matches]
    else:
        for item in capacity_str.split(','):
            item = item.strip().strip("'")
            num_match = re.search(r'\d+\.?\d*', item)
            if num_match:
                capacity_values.append(float(num_match.group()))
    
    return real_demand, capacity_values


def minutes_to_time_label(minutes):
    """将分钟数转换为时间标签（从7:00开始）"""
    hour = 7 + minutes // 60
    minute = minutes % 60
    return f"{hour:02d}:{minute:02d}"


def plot_figure_2_6():
    """绘制图2-6：晚高峰提前实验（调整前后对比，saved_model）"""
    
    print("="*80)
    print("绘制图2-6：晚高峰提前实验（saved_model）")
    print("="*80)
    
    # 1. 加载调整前数据
    print("\n1. 加载调整前数据...")
    real_demand_original, capacity_original = load_simulated_data(original_data_file)
    print(f"   数据点: {len(real_demand_original)}")
    
    # 2. 加载调整后数据
    print("\n2. 加载调整后数据...")
    real_demand_shifted, capacity_shifted = load_simulated_data(shifted_data_file)
    print(f"   数据点: {len(real_demand_shifted)}")
    
    # 3. 生成时间标签
    time_points = [i * 30 for i in range(len(real_demand_shifted))]
    time_labels = [minutes_to_time_label(t) for t in time_points]
    x_indices = range(len(time_points))
    
    # 4. 绘制图表
    print("\n3. 绘制图表...")
    plt.figure(figsize=(14, 7))
    
    # 绘制调整前的真实需求和DRL-TSBC容量
    plt.plot(x_indices, real_demand_original, color='#1f77b4', linestyle='-', marker='o', 
             label='调整前真实需求', linewidth=2, markersize=5)
    plt.plot(x_indices, capacity_original, color='#ff7f0e', linestyle='-', marker='s', 
             label='调整前', linewidth=2, markersize=5)
    
    # 绘制调整后的真实需求和DRL-TSBC容量
    plt.plot(x_indices, real_demand_shifted, color='#2ca02c', linestyle='--', marker='^', 
             label='调整后真实需求', linewidth=2, markersize=5)
    plt.plot(x_indices, capacity_shifted, color='#d62728', linestyle='--', marker='d', 
             label='调整后', linewidth=2, markersize=5)
    
    # 设置坐标轴
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('总客运容量', fontsize=12)
    plt.title('提前晚高峰前后DRL-TSBC生成的公交时刻表提供的总客运量与真实需求的对比（saved_model）', fontsize=12)
    
    # 设置x轴刻度
    display_indices = list(range(0, len(x_indices), 2))
    plt.xticks([x_indices[i] for i in display_indices], 
               [time_labels[i] for i in display_indices], 
               rotation=0, fontsize=10)
    
    # 设置y轴
    max_value = max(max(real_demand_original), max(capacity_original),
                    max(real_demand_shifted), max(capacity_shifted))
    y_max = int((max_value + 50) // 50 * 50)
    plt.ylim(0, y_max)
    plt.yticks(range(0, y_max + 1, 50), fontsize=10)
    
    # 添加网格
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 添加晚高峰时段标注
    plt.axvspan(20, 24, alpha=0.15, color='orange', label='原晚高峰时段')
    plt.axvspan(18, 22, alpha=0.15, color='yellow', label='新晚高峰时段（提前1小时）')
    
    # 添加图例
    plt.legend(fontsize=10, loc='upper right')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_file = f'figure_2_6_savedmodel_{busline}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n图片已保存: {output_file}")
    
    # 显示统计信息
    print("\n" + "="*80)
    print("数据统计:")
    print("-"*80)
    print("调整前:")
    print(f"  真实需求 - 平均: {sum(real_demand_original)/len(real_demand_original):.2f}, "
          f"最大: {max(real_demand_original):.2f}, 最小: {min(real_demand_original):.2f}")
    print(f"  DRL-TSBC - 平均: {sum(capacity_original)/len(capacity_original):.2f}, "
          f"最大: {max(capacity_original):.2f}, 最小: {min(capacity_original):.2f}")
    
    print("\n调整后:")
    print(f"  真实需求 - 平均: {sum(real_demand_shifted)/len(real_demand_shifted):.2f}, "
          f"最大: {max(real_demand_shifted):.2f}, 最小: {min(real_demand_shifted):.2f}")
    print(f"  DRL-TSBC - 平均: {sum(capacity_shifted)/len(capacity_shifted):.2f}, "
          f"最大: {max(capacity_shifted):.2f}, 最小: {min(capacity_shifted):.2f}")
    
    # 找出高峰时段
    original_peak_idx = real_demand_original.index(max(real_demand_original))
    shifted_peak_idx = real_demand_shifted.index(max(real_demand_shifted))
    
    print("\n高峰时段:")
    print(f"  调整前高峰: {time_labels[original_peak_idx]} (需求: {real_demand_original[original_peak_idx]:.1f})")
    print(f"  调整后高峰: {time_labels[shifted_peak_idx]} (需求: {real_demand_shifted[shifted_peak_idx]:.1f})")
    print(f"  高峰提前: {(original_peak_idx - shifted_peak_idx) * 30} 分钟")
    
    print("="*80)
    print("绘图完成！")
    print("="*80)
    
    plt.close()


if __name__ == "__main__":
    plot_figure_2_6()
