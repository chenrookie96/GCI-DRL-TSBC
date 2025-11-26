"""
绘制图2-7：提前晚高峰前后DRL-TSBC生成的公交时刻表提供的上下行容运容量对比
实验设计：
- 上行：晚高峰提前1小时（shifted数据）
- 下行：保持原始数据（不提前）
- 展示：即使需求不对称，DRL-TSBC仍保持上下行发车次数一致
"""

import matplotlib.pyplot as plt
import numpy as np
import re

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 配置参数
busline = 208

# 读取模拟结果
input_file = f"test_data/{busline}/bidirectional_demand_capacity_{busline}_savedmodel.txt"

print("="*80)
print("绘制图2-7：上下行容量对比（上行shifted + 下行原始）")
print("="*80)

with open(input_file, 'r', encoding='utf-8') as f:
    content = f.read()

# 解析数据
def parse_list(content, label):
    pattern = f"{label}:\n\\[(.*?)\\]"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        values_str = match.group(1)
        values = []
        for x in values_str.split(','):
            x = x.strip().strip("'")
            # 处理 np.float64() 格式
            if 'np.float64(' in x:
                x = x.replace('np.float64(', '').replace(')', '')
            values.append(float(x))
        return values
    return []

upward_real = parse_list(content, "上行真实需求")
upward_cap = parse_list(content, "上行DRL-TSBC容量")
downward_real = parse_list(content, "下行真实需求")
downward_cap = parse_list(content, "下行DRL-TSBC容量")

print(f"\n数据点数:")
print(f"  上行真实需求（shifted）: {len(upward_real)}")
print(f"  上行容量: {len(upward_cap)}")
print(f"  下行真实需求（原始）: {len(downward_real)}")
print(f"  下行容量: {len(downward_cap)}")

print(f"\n实验说明:")
print(f"  上行：使用shifted数据（晚高峰提前1小时）")
print(f"  下行：使用原始数据（不提前）")
print(f"  目的：展示DRL-TSBC在不对称需求下保持发车次数一致")

# 生成时间标签（从7:00开始，每半小时一个点）
time_labels = []
for i in range(len(upward_real)):
    hour = 7 + i // 2
    minute = 0 if i % 2 == 0 else 30
    time_labels.append(f"{hour}:{minute:02d}")

# 创建图表
fig, ax = plt.subplots(figsize=(12, 6))

# 绘制4条曲线
x = np.arange(len(upward_real))

# 上行容量（蓝色实线，圆形标记）
ax.plot(x, upward_cap, 'o-', color='#1f77b4', linewidth=2, 
        markersize=5, label='上行', markerfacecolor='#1f77b4')

# 上行真实需求（蓝色虚线，方块标记）
ax.plot(x, upward_real, 's--', color='#1f77b4', linewidth=2, 
        markersize=5, label='上行真实需求', markerfacecolor='#1f77b4')

# 下行容量（绿色实线，三角标记）
ax.plot(x, downward_cap, '^-', color='#2ca02c', linewidth=2, 
        markersize=5, label='下行', markerfacecolor='#2ca02c')

# 下行真实需求（绿色虚线，倒三角标记）
ax.plot(x, downward_real, 'v--', color='#2ca02c', linewidth=2, 
        markersize=5, label='下行真实需求', markerfacecolor='#2ca02c')

# 设置坐标轴
ax.set_xlabel('时间', fontsize=14, fontweight='bold')
ax.set_ylabel('运送乘客数量', fontsize=14, fontweight='bold')

# 设置x轴刻度（每2个点显示一次，即每小时）
tick_positions = list(range(0, len(time_labels), 2))
tick_labels = [time_labels[i] for i in tick_positions]
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, fontsize=11)

# 设置y轴
ax.set_ylim(0, max(max(upward_cap), max(downward_cap)) * 1.1)
ax.tick_params(axis='y', labelsize=11)

# 添加网格
ax.grid(True, linestyle='--', alpha=0.3)

# 添加图例
ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

# 调整布局
plt.tight_layout()

# 保存图片
output_file = f"drl_tsbc_result/figure_2_7_savedmodel_{busline}.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n图表已保存到: {output_file}")

# 显示图表
plt.show()

print("="*80)
