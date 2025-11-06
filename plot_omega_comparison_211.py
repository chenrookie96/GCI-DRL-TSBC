"""
复现论文图2-8：DRL-TSBC在不同ω下的211线发车次数与乘客平均等待时间的对比
使用表格数据绘制
"""

import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 211线数据（从表格中提取）
omega_values = [500, 1000, 2000, 3000, 4000]
omega_labels = ['1/500', '1/1000', '1/2000', '1/3000', '1/4000']

# 发车次数
ndt = [80, 72, 64, 57, 53]

# 上行AWT
awt_upward = [3.29, 4.77, 5.84, 7.61, 8.45]

# 下行AWT
awt_downward = [3.04, 3.61, 4.34, 6.09, 6.85]

# 创建图形
fig, ax1 = plt.subplots(figsize=(10, 6))

# 左侧Y轴：发车次数
color_ndt = 'tab:blue'
ax1.set_xlabel('ω', fontsize=14)
ax1.set_ylabel('发车次数', fontsize=14, color=color_ndt)
line1 = ax1.plot(omega_labels, ndt, marker='o', color=color_ndt, 
                 linewidth=1.5, markersize=6, label='NDT')
ax1.tick_params(axis='y', labelcolor=color_ndt)
ax1.set_ylim(50, 85)
ax1.grid(True, alpha=0.3, linestyle='--')

# 右侧Y轴：平均等待时间
ax2 = ax1.twinx()
color_awt = 'tab:orange'
ax2.set_ylabel('乘客平均等待时间 (min)', fontsize=14, color=color_awt)
line2 = ax2.plot(omega_labels, awt_upward, marker='s', color=color_awt, 
                 linewidth=1.5, markersize=6, label='AWT_upward')
line3 = ax2.plot(omega_labels, awt_downward, marker='^', color='tab:purple', 
                 linewidth=1.5, markersize=6, label='AWT_downward')
ax2.tick_params(axis='y', labelcolor=color_awt)
ax2.set_ylim(2.5, 9)

# 添加标题
plt.title('DRL-TSBC 在不同 ω 下的 211 线发车次数与乘客平均等待时间的对比', 
          fontsize=15, pad=20)

# 合并图例
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center left', fontsize=11)

# 调整布局
fig.tight_layout()

# 保存图片
output_file = 'omega_comparison_211.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"图片已保存: {output_file}")

# 关闭图形
plt.close()

print("\n数据对比：")
print("="*70)
print(f"{'ω':<12} {'发车次数':<12} {'上行AWT':<12} {'下行AWT':<12}")
print("-"*70)
for i in range(len(omega_labels)):
    print(f"{omega_labels[i]:<12} {ndt[i]:<12} {awt_upward[i]:<12.2f} {awt_downward[i]:<12.2f}")
print("="*70)

print("\n观察：")
print("1. ω越小（如1/4000），发车次数越少（53次），但AWT越高（8.45分钟）")
print("2. ω越大（如1/500），发车次数越多（80次），但AWT越低（3.29分钟）")
print("3. 这说明ω控制了等待时间与发车成本的权衡")
