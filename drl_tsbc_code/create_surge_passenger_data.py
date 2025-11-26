"""
创建突发客流数据（图2-5）
在208线上行方向的第9站（编号8）增加晚高峰期间的客流量
模拟突发事件导致的客流突增
"""

import pandas as pd
import numpy as np

np.random.seed(42)

# 读取原始上行乘客数据
df = pd.read_csv('test_data/208/passenger_dataframe_direction0.csv')

print("="*60)
print("创建突发客流数据（图2-5）")
print("="*60)
print(f"原始乘客数: {len(df)}")

# 晚高峰时段：17:00-19:30（1020-1170分钟）
evening_start = 1020  # 17:00
evening_end = 1170    # 19:30

# 第9站编号为8
target_station = 8

# 统计原始第9站晚高峰客流
original_station9_evening = df[
    (df['Boarding station'] == target_station) & 
    (df['Boarding time'] >= evening_start) & 
    (df['Boarding time'] <= evening_end)
]
print(f"原始第9站晚高峰乘客数: {len(original_station9_evening)}")

# 增加的乘客数量（大幅增加客流，使DRL-TSBC明显增加发车）
# 根据图2-5，调整后晚高峰需求明显高于调整前
num_new_passengers = 300  # 增加300人，使晚高峰客流显著增加

# 生成新乘客数据
new_passengers = []
for i in range(num_new_passengers):
    # 随机生成上车时间（晚高峰期间，集中在17:00-19:00）
    boarding_time = np.random.randint(1020, 1140)  # 17:00-19:00
    
    # 到站时间比上车时间早1-5分钟
    arrival_time = boarding_time - np.random.randint(1, 6)
    
    # 下车站点：从第9站上车，随机选择后面的站点下车
    # 站点范围0-25，从站点8上车，可以在9-25下车
    alighting_station = np.random.randint(target_station + 1, 26)
    
    # 生成唯一标签
    label = f"SURGE_{i:06d}"
    
    new_passengers.append({
        'Label': label,
        'Boarding time': boarding_time,
        'Boarding station': target_station,
        'Alighting station': alighting_station,
        'Arrival time': arrival_time
    })

# 创建新乘客DataFrame
new_df = pd.DataFrame(new_passengers)

# 合并原始数据和新增数据
surge_df = pd.concat([df, new_df], ignore_index=True)

# 按上车时间排序
surge_df = surge_df.sort_values('Boarding time').reset_index(drop=True)

print(f"新增乘客数: {len(new_df)}")
print(f"合并后总乘客数: {len(surge_df)}")

# 验证第9站晚高峰客流
new_station9_evening = surge_df[
    (surge_df['Boarding station'] == target_station) & 
    (surge_df['Boarding time'] >= evening_start) & 
    (surge_df['Boarding time'] <= evening_end)
]
print(f"调整后第9站晚高峰乘客数: {len(new_station9_evening)}")

# 保存突发客流数据
output_path = 'test_data/208/passenger_dataframe_direction0_surge.csv'
surge_df.to_csv(output_path, index=False)
print(f"\n突发客流数据已保存: {output_path}")

# 打印晚高峰对比
print("\n" + "="*60)
print("晚高峰客流对比（第9站）")
print("="*60)
print(f"{'时段':<15} {'原始':<10} {'调整后':<10} {'增加':<10}")
print("-"*45)

for hour in range(17, 20):
    for half in [0, 30]:
        start = hour * 60 + half
        end = start + 30
        
        orig_count = len(original_station9_evening[
            (original_station9_evening['Boarding time'] >= start) & 
            (original_station9_evening['Boarding time'] < end)
        ])
        
        new_count = len(new_station9_evening[
            (new_station9_evening['Boarding time'] >= start) & 
            (new_station9_evening['Boarding time'] < end)
        ])
        
        print(f"{hour:02d}:{half:02d}-{hour:02d}:{half+30:02d}    {orig_count:<10} {new_count:<10} +{new_count-orig_count:<10}")

print("="*60)
