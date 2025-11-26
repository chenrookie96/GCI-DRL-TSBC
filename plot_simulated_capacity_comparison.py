"""
使用模拟数据绘制容量对比图（图2-3）
直接使用 simulate_with_correct_env.py 生成的真实需求和DRL-TSBC容量数据
"""

import matplotlib.pyplot as plt
from matplotlib import rcParams
import re

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 配置参数
busline = 208
omega_factor = 1000

# 数据文件路径
simulated_data_file = f"test_data/{busline}/simulated_demand_capacity_{busline}.txt"


def load_simulated_data(file_path):
    """
    从模拟结果文件中加载真实需求和DRL-TSBC容量数据
    
    返回:
    - real_demand: 真实需求列表
    - drl_tsbc_capacity: DRL-TSBC容量列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 解析真实需求
    demand_match = re.search(r'真实需求:\n\[(.*?)\]', content, re.DOTALL)
    if not demand_match:
        raise ValueError("无法找到真实需求数据")
    
    demand_str = demand_match.group(1)
    real_demand = [float(x.strip()) for x in demand_str.split(',')]
    
    # 解析DRL-TSBC容量
    capacity_match = re.search(r'DRL-TSBC容量:\n\[(.*?)\]', content, re.DOTALL)
    if not capacity_match:
        raise ValueError("无法找到DRL-TSBC容量数据")
    
    capacity_str = capacity_match.group(1)
    # 处理可能包含 np.float64() 的格式
    # 例如: np.float64(195.2)
    capacity_values = []
    # 使用正则提取所有 np.float64(数字) 格式
    matches = re.findall(r'np\.float64\(([\d.]+)\)', capacity_str)
    if matches:
        capacity_values = [float(x) for x in matches]
    else:
        # 如果没有 np.float64 格式，直接按逗号分割
        for item in capacity_str.split(','):
            item = item.strip()
            num_match = re.search(r'\d+\.?\d*', item)
            if num_match:
                capacity_values.append(float(num_match.group()))
    
    return real_demand, capacity_values


def minutes_to_time_label(minutes):
    """将分钟数转换为时间标签（从7:00开始）"""
    hour = 7 + minutes // 60
    minute = minutes % 60
    return f"{hour:02d}:{minute:02d}"


def plot_capacity_comparison():
    """绘制容量对比图"""
    
    print("="*80)
    print("使用模拟数据绘制图2-3：容量对比图")
    print("="*80)
    
    # 1. 加载模拟数据
    print("\n1. 加载模拟数据...")
    real_demand, drl_tsbc_capacity = load_simulated_data(simulated_data_file)
    print(f"   真实需求数据点: {len(real_demand)}")
    print(f"   DRL-TSBC容量数据点: {len(drl_tsbc_capacity)}")
    
    # 2. 加载人工方案数据（论文中的数据）
    print("\n2. 加载人工方案数据...")
    manual_capacity = [125, 200, 250, 200, 150, 125, 100, 100, 125, 125, 125, 100, 125, 125,
                      100, 100, 125, 100, 75, 125, 125, 150, 175, 200, 150, 100, 100, 100, 75]
    
    # 3. 生成时间标签（从7:00开始，每半小时一个点）
    time_points = [i * 30 for i in range(len(real_demand))]  # 0, 30, 60, ...
    time_labels = [minutes_to_time_label(t) for t in time_points]
    x_indices = range(len(time_points))
    
    # 4. 绘制对比图
    print("\n3. 绘制对比图...")
    plt.figure(figsize=(14, 7))
    
    # 绘制三条曲线
    plt.plot(x_indices, real_demand, 'b-o', label='真实需求', 
             linewidth=2, markersize=4, markerfacecolor='blue', markeredgecolor='blue')
    plt.plot(x_indices, drl_tsbc_capacity, 'r-s', label='DRL-TSBC', 
             linewidth=2, markersize=4, markerfacecolor='red', markeredgecolor='red')
    plt.plot(x_indices, manual_capacity, 'g-^', label='人工方案', 
             linewidth=2, markersize=4, markerfacecolor='green', markeredgecolor='green')
    
    # 设置坐标轴
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('总客运容量', fontsize=12)
    plt.title('不同算法生成的公交时刻表提供的总客运量与真实需求的对比', fontsize=14)
    
    # 设置x轴刻度（每隔2个点显示一个标签）
    display_indices = list(range(0, len(x_indices), 2))
    plt.xticks([x_indices[i] for i in display_indices], 
               [time_labels[i] for i in display_indices], 
               rotation=0, fontsize=10)
    
    # 设置y轴范围和刻度
    max_value = max(max(real_demand), max(drl_tsbc_capacity), max(manual_capacity))
    y_max = int((max_value + 50) // 50 * 50)  # 向上取整到50的倍数
    plt.ylim(0, y_max)
    plt.yticks(range(0, y_max + 1, 50), fontsize=10)
    
    # 添加网格
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 添加图例
    plt.legend(fontsize=11, loc='upper right')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_file = f'figure_2_3_simulated_capacity_comparison_{busline}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n图片已保存: {output_file}")
    
    # 显示统计信息
    print("\n" + "="*80)
    print("数据统计:")
    print("-"*80)
    print(f"真实需求 - 平均值: {sum(real_demand)/len(real_demand):.2f}, "
          f"最大值: {max(real_demand):.2f}, 最小值: {min(real_demand):.2f}")
    print(f"DRL-TSBC - 平均值: {sum(drl_tsbc_capacity)/len(drl_tsbc_capacity):.2f}, "
          f"最大值: {max(drl_tsbc_capacity):.2f}, 最小值: {min(drl_tsbc_capacity):.2f}")
    print(f"人工方案 - 平均值: {sum(manual_capacity)/len(manual_capacity):.2f}, "
          f"最大值: {max(manual_capacity):.2f}, 最小值: {min(manual_capacity):.2f}")
    print("="*80)
    print("绘图完成！")
    print("="*80)
    
    plt.close()


if __name__ == "__main__":
    plot_capacity_comparison()
