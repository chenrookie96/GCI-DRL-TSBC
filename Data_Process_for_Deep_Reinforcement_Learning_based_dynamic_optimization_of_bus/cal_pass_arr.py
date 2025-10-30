import csv
from collections import namedtuple
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
date = '2023-08-14'

# 使用pandas读取csv文件，包含date,s,arrival_time,depart_time
df = pd.read_csv('.\\data_cq_with_date\\208\\arrive_depart_time_0.csv')
# 将df第一行中的下划线转化为空格

# 取出日期为2023-08-14的数据
df = df[df['date'] == date]
# 若s=1，使用depart_time表示arrival_time
df.loc[df['s'] == 1, 'arrival_time'] = df['depart_time']
# 去除depart_time信息
df = df.drop('depart_time', axis=1)
# 去除包含nan的信息
df = df.dropna()
# 将arrival_time转化为用分钟表示，第0分钟为00:00
df['arrival_time'] = df['arrival_time'].apply(lambda x: int(str(x).split(':')[0]) * 60 + int(str(x).split(':')[1]))
# 用s进行分类，并通过arrival_time进行排序
df = df.sort_values(by=['s', 'arrival_time'])

# 使用pandas读取另一个csv文件，包含Label,Boarding time,Boarding_station,Alighting_station,Arrival_time,Date
df2 = pd.read_csv('.\\data_cq_with_date\\208\\passenger_dataframe_direction-0.csv')
# 取出日期为2023-08-14的数据
df2 = df2[df2['Date'] == date]
# 将df2中时间在晚上18点到20点间的数据和16点到18点间的数据调换
# temp = df2
# df2.loc[(df2['Boarding time'] >= 960) & (df2['Boarding time'] < 1080), 'Boarding time'] += 120
# df2.loc[(temp['Boarding time'] >= 1080) & (temp['Boarding time'] < 1200), 'Boarding time'] -= 120
# 遍历df2，在df中找到df2中Boarding_station等于df中s-1的数据，将df2中的Arrival_time替换为小于df2的Boarding time的df中的arrival_time
for index, row in df2.iterrows():
    # 取出df中s等于df2中Boarding_station的数据
    df3 = df[(df['s'] - 1) == row['Boarding_station']]
    # 取出df3中最大的arrival_time但小于df2中Boarding time的数据
    df4 = df3[df3['arrival_time'] < row['Boarding time']].max()
    # 若df4为nan，则将直接将其记为无效数据
    if np.isnan(df4['arrival_time']):
        df4 = df3[df3['arrival_time'] > row['Boarding time']].min()
    # 在区间中，通过一个分布函数来确定df2中的Arrival_time，这里要求越靠近Boarding time的时间点的概率越大
    # print(row['Boarding time'],df4['arrival_time'] )
    # x = np.random.normal(row['Boarding time'] , (row['Boarding time']  - df4['arrival_time']) / 2 , 100000)
    # 去掉x中大于row['Boarding time']的值
    # x = x[x <= row['Boarding time']]
    # plt.figure(figsize=(20, 10), dpi=100)
    # 2）绘制直方图
    # plt.hist(x, 1000)
    # 3）显示图像
    # plt.show()

    x = np.random.normal(row['Boarding time'] , abs(row['Boarding time'] - df4['arrival_time']) / 2)
    x = abs(x - row['Boarding time'])
    estimate_time = int(row['Boarding time'] - x)
    # 将df2中的Arrival_time替换为通过一个分布计算的值
    df2.loc[index, 'Arrival_time'] = estimate_time
# 去掉date列
df2 = df2.drop('Date', axis=1)
# 将df2输出为csv文件
df2.columns = df2.columns.str.replace('_', ' ')
df2.to_csv('.\\data_cq_with_date\\208\\passenger_dataframe_direction0.csv', index=False)

# 将df中s为1的arrival_time以列表形式输出
print(df[df['s'] == 1]['arrival_time'].tolist())
