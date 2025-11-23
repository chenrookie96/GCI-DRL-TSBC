"""
创建提前晚高峰的乘客数据
将晚高峰时段（17:00-18:30）的乘客到达时间提前1小时
"""

import pandas as pd
import numpy as np

# 配置
busline = 208
direction = 0  # 上行
shift_hours = 1  # 提前1小时
shift_minutes = shift_hours * 60

# 原始数据路径
original_file = f"test_data/{busline}/passenger_dataframe_direction{direction}.csv"
# 新数据路径
shifted_file = f"test_data/{busline}/passenger_dataframe_direction{direction}_shifted.csv"

print("="*60)
print("创建提前晚高峰的乘客数据")
print("晚高峰定义: 17:00-19:00 提前到 16:00-18:00")
print("="*60)

# 读取原始数据
print(f"\n读取原始数据: {original_file}")
df = pd.read_csv(original_file)
print(f"原始数据行数: {len(df)}")
print(f"列名: {df.columns.tolist()}")

# 复制数据
df_shifted = df.copy()

# 晚高峰时段定义（从0:00午夜开始计算的分钟数）
# 原始晚高峰：17:00-19:00 = 1020-1140分钟
# 提前后晚高峰：16:00-18:00 = 960-1080分钟
evening_peak_start = 1020  # 17:00
evening_peak_end = 1140    # 19:00

# 统计晚高峰乘客数量
evening_passengers = df[(df['Arrival time'] >= evening_peak_start) & 
                        (df['Arrival time'] <= evening_peak_end)]
print(f"\n晚高峰时段乘客数: {len(evening_passengers)}")

# 将晚高峰时段的乘客到达时间提前
mask = (df_shifted['Arrival time'] >= evening_peak_start) & \
       (df_shifted['Arrival time'] <= evening_peak_end)
df_shifted.loc[mask, 'Arrival time'] = df_shifted.loc[mask, 'Arrival time'] - shift_minutes

# 同样需要调整上车时间（如果有的话）
if 'Boarding time' in df_shifted.columns:
    mask_boarding = (df_shifted['Boarding time'] >= evening_peak_start) & \
                    (df_shifted['Boarding time'] <= evening_peak_end)
    df_shifted.loc[mask_boarding, 'Boarding time'] = \
        df_shifted.loc[mask_boarding, 'Boarding time'] - shift_minutes

print(f"\n提前后的晚高峰时段: {evening_peak_start - shift_minutes} - {evening_peak_end - shift_minutes} 分钟")
print(f"对应时间: {(evening_peak_start - shift_minutes)//60}:{(evening_peak_start - shift_minutes)%60:02d} - "
      f"{(evening_peak_end - shift_minutes)//60}:{(evening_peak_end - shift_minutes)%60:02d}")

# 保存新数据
df_shifted.to_csv(shifted_file, index=False)
print(f"\n提前晚高峰的数据已保存: {shifted_file}")

# 统计对比
print("\n" + "="*60)
print("数据统计对比")
print("="*60)
print(f"原始数据总行数: {len(df)}")
print(f"新数据总行数: {len(df_shifted)}")
print(f"晚高峰乘客数: {len(evening_passengers)}")

# 显示新的晚高峰时段乘客数
new_evening_passengers = df_shifted[(df_shifted['Arrival time'] >= evening_peak_start - shift_minutes) & 
                                     (df_shifted['Arrival time'] <= evening_peak_end - shift_minutes)]
print(f"新晚高峰时段乘客数: {len(new_evening_passengers)}")

print("\n完成！")
