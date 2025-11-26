"""
绘制图2-5：增加晚高峰客流前后DRL-TSBC生成的公交时刻表提供的总客运量与真实需求的对比
"""

import matplotlib.pyplot as plt
import numpy as np
import ast
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

busline = 208

def load_data():
    """加载仿真数据"""
    data_file = f"test_data/{busline}/figure_2_5_data.txt"
    
    if not os.path.exists(data_file):
        print(f"错误: 找不到数据文件: {data_file}")
        print("请先运行仿真脚本")
        return None
    
    with open(data_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 解析数据
    lines = content.split('\n')
    data = {}
    current_key = None
    
    for line in lines:
        if '调整前容量' in line:
            current_key = 'cap_before'
        elif '调整前真实需求' in line:
            current_key = 'real_before'
        elif '调整后容量' in line:
            current_key = 'cap_after'
        elif '调整后真实需求' in line:
            current_key = 'real_after'
        elif line.startswith('[') and current_key:
            data[current_key] = ast.literal_eval(line)
            current_key = None
    
    return data

def plot_figure_2_5():
    """绘制图2-5"""
    data = load_data()
    if data is None:
        return
    
    cap_before = data['cap_before']
    real_before = data['real_before']
    cap_after = data['cap_after']
    real_after = data['real_after']
    
    # 时间轴：从7:00到21:00，每半小时一个点
    time_labels = []
    for hour in range(7, 22):
        time_labels.append(f"{hour}:00")
        if hour < 21:
            time_labels.append(f"{hour}:30")
    
    # 确保数据长度匹配
    n_points = min(len(cap_before), len(real_before), len(cap_after), len(real_after), len(time_labels))
    
    x = np.arange(n_points)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制四条线（方块=调整前，星形=调整后）
    ax.plot(x, cap_before[:n_points], 's-', color='blue', label='调整前', markersize=5, linewidth=1.5)
    ax.plot(x, real_before[:n_points], 's-', color='cyan', label='调整前的真实需求', markersize=5, linewidth=1.5)
    ax.plot(x, cap_after[:n_points], '*-', color='purple', label='调整后', markersize=7, linewidth=1.5)
    ax.plot(x, real_after[:n_points], '*-', color='red', label='调整后的真实需求', markersize=7, linewidth=1.5)
    
    # 设置坐标轴
    ax.set_xlabel('时间', fontsize=12)
    ax.set_ylabel('总客运容量', fontsize=12)
    
    # 设置x轴刻度（每隔几个点显示一个标签）
    tick_positions = [0, 5, 10, 15, 20, 25]  # 7:00, 9:30, 12:00, 14:30, 17:00, 19:30
    tick_labels = ['7:00', '9:30', '12:00', '14:30', '17:00', '19:30']
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    
    # 设置y轴范围
    ax.set_ylim(0, 400)
    
    # 添加图例
    ax.legend(loc='upper right', fontsize=10)
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 添加标题
    ax.set_title('图2-5 增加晚高峰客流前后DRL-TSBC生成的公交时刻表提供的总客运量与真实需求的对比', fontsize=11)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = f"test_data/{busline}/figure_2_5.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"图片已保存: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    plot_figure_2_5()
