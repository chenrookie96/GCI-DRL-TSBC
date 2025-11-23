"""
使用saved_model的数据绘制图2-3：原始数据的真实需求与DRL-TSBC容量对比
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
data_file = f"test_data/{busline}/simulated_demand_capacity_{busline}_savedmodel.txt"


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


def plot_figure_2_3():
    """绘制图2-3：原始数据（saved_model）"""
    
    print("="*80)
    print("绘制图2-3：原始数据的真实需求与DRL-TSBC容量对比（saved_model）")
    print("="*80)
    
    # 加载数据
    print("\n加载数据...")
    real_demand, capacity = load_simulated_data(data_file)
    print(f"   数据点: {len(real_demand)}")
    
    # 加载人工方案数据（论文中的数据）
    print("\n加载人工方案数据...")
    manual_capacity = [125, 200, 250, 200, 150, 125, 100, 100, 125, 125, 125, 100, 125, 125,
                      100, 100, 125, 100, 75, 125, 125, 150, 175, 200, 150, 100, 100, 100, 75]
    
    # 生成时间标签
    time_points = [i * 30 for i in range(len(real_demand))]
    time_labels = [minutes_to_time_label(t) for t in time_points]
    x_indices = range(len(time_points))
    
    # 绘制图表
    print("\n绘制图表...")
    plt.figure(figsize=(14, 7))
    
    # 绘制真实需求、DRL-TSBC容量和人工方案
    plt.plot(x_indices, real_demand, color='#1f77b4', linestyle='-', marker='o', 
             label='真实需求', linewidth=2, markersize=5)
    plt.plot(x_indices, capacity, color='#ff7f0e', linestyle='-', marker='s', 
             label='DRL-TSBC', linewidth=2, markersize=5)
    plt.plot(x_indices, manual_capacity, color='#2ca02c', linestyle='-', marker='^', 
             label='人工方案', linewidth=2, markersize=5)
    
    # 设置坐标轴
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('总客运容量', fontsize=12)
    plt.title('不同算法生成的公交时刻表提供的总客运量与真实需求的对比（saved_model）', fontsize=13)
    
    # 设置x轴刻度
    display_indices = list(range(0, len(x_indices), 2))
    plt.xticks([x_indices[i] for i in display_indices], 
               [time_labels[i] for i in display_indices], 
               rotation=0, fontsize=10)
    
    # 设置y轴
    max_value = max(max(real_demand), max(capacity), max(manual_capacity))
    y_max = int((max_value + 50) // 50 * 50)
    plt.ylim(0, y_max)
    plt.yticks(range(0, y_max + 1, 50), fontsize=10)
    
    # 添加网格
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 添加图例
    plt.legend(fontsize=11, loc='upper right')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_file = f'figure_2_3_savedmodel_{busline}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n图片已保存: {output_file}")
    
    # 显示统计信息
    print("\n" + "="*80)
    print("数据统计:")
    print("-"*80)
    print(f"真实需求 - 平均: {sum(real_demand)/len(real_demand):.2f}, "
          f"最大: {max(real_demand):.2f}, 最小: {min(real_demand):.2f}")
    print(f"DRL-TSBC - 平均: {sum(capacity)/len(capacity):.2f}, "
          f"最大: {max(capacity):.2f}, 最小: {min(capacity):.2f}")
    print(f"人工方案 - 平均: {sum(manual_capacity)/len(manual_capacity):.2f}, "
          f"最大: {max(manual_capacity):.2f}, 最小: {min(manual_capacity):.2f}")
    
    # 找出高峰时段
    peak_idx = real_demand.index(max(real_demand))
    print(f"\n高峰时段: {time_labels[peak_idx]} (需求: {real_demand[peak_idx]:.1f})")
    
    print("="*80)
    print("绘图完成！")
    print("="*80)
    
    plt.close()


if __name__ == "__main__":
    plot_figure_2_3()
