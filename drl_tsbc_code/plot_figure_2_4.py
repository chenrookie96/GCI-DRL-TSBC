"""
绘制图2-4：DRL-TSBC在208线上下行方向生成的公交时刻表所提供的总客运容量与真实需求的对比
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
    data_file = f"test_data/{busline}/figure_2_4_data.txt"
    
    if not os.path.exists(data_file):
        print(f"错误: 找不到数据文件: {data_file}")
        return None
    
    with open(data_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 解析数据
    lines = content.split('\n')
    data = {}
    current_key = None
    
    for line in lines:
        if '上行容量' in line and '真实' not in line:
            current_key = 'up_cap'
        elif '上行真实需求' in line:
            current_key = 'up_real'
        elif '下行容量' in line and '真实' not in line:
            current_key = 'down_cap'
        elif '下行真实需求' in line:
            current_key = 'down_real'
        elif line.startswith('[') and current_key:
            data[current_key] = ast.literal_eval(line)
            current_key = None
    
    return data

def plot_figure_2_4():
    """绘制图2-4"""
    data = load_data()
    if data is None:
        return
    
    up_cap = data['up_cap']
    up_real = data['up_real']
    down_cap = data['down_cap']
    down_real = data['down_real']
    
    # 确保数据长度匹配
    n_points = min(len(up_cap), len(up_real), len(down_cap), len(down_real))
    
    x = np.arange(n_points)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制四条线（按照论文图2-4的颜色）
    # 上行：蓝色系
    ax.plot(x, up_cap[:n_points], 'b-s', label='上行', markersize=5, linewidth=1.5)
    ax.plot(x, up_real[:n_points], 'c-s', label='上行真实需求', markersize=5, linewidth=1.5)
    # 下行：绿色系
    ax.plot(x, down_cap[:n_points], 'g-o', label='下行', markersize=5, linewidth=1.5)
    ax.plot(x, down_real[:n_points], 'lime', marker='o', label='下行真实需求', markersize=5, linewidth=1.5)
    
    # 设置坐标轴
    ax.set_xlabel('时间', fontsize=12)
    ax.set_ylabel('总客运容量', fontsize=12)
    
    # 设置x轴刻度
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
    ax.set_title('图2-4 DRL-TSBC在208线上下行方向生成的公交时刻表所提供的总客运容量与真实需求的对比', fontsize=11)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = f"test_data/{busline}/figure_2_4.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"图片已保存: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    plot_figure_2_4()
