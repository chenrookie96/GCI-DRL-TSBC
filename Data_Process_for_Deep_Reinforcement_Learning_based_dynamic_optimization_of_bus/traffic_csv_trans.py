import csv
from collections import namedtuple
import sys
import numpy as np

output = open("traffic-1.csv", "w")
sys.stdout = output

now = 345
max = 0
with open(r'.\\data_cq\\data_cq\\278\\traffic-0.csv', encoding='utf-8')as f:
    reader = csv.reader(f)
    headers = next(reader)
    Row = namedtuple('Row', headers)
    for row in reader:
        row = Row(*row)
        if (int(row.order) > max):
            max = int(row.order)
    print('time_h1,time_h2,time_m1,time_m2,start_m,finish_m',end=',')
    for i in range(max + 1):
        print('s'+str(i),end=',')
        

print()
max_zeros = max + 1  # 根据你的需求设定max的值

for hour in range(7):
    for minute in range(0, 60, 15):
        if hour == 6 and minute > 15:
            break
        time_h1 = hour
        time_h2 = hour
        time_m1 = minute
        time_m2 = minute + 15
        start_m = hour * 60 + minute
        finish_m = start_m + 15
        if time_m2 == 60:
            time_h2 += 1
            time_m2 = 0
        zeros = [0] * max_zeros
        if hour == 6 and minute == 15:
            print(f"{time_h1},{time_h2},{time_m1},{time_m2},{start_m+1},{finish_m},{','.join(map(str, zeros))}",end = ',')
        else:
            print(f"{time_h1},{time_h2},{time_m1},{time_m2},{start_m+1},{finish_m},{','.join(map(str, zeros))},")

with open(r'.\\data_cq\\data_cq\\278\\traffic-0.csv', encoding='utf-8')as f:
    reader = csv.reader(f)
    headers = next(reader)
    Row = namedtuple('Row', headers)
    for row in reader:
        row = Row(*row)
        if row.start_m != now:
            print()
            now = row.start_m
            print(row.time_h1 + ',' + row.time_h2 + ',' + row.time_m1 + ',' + row.time_m2 + ',' + row.start_m + ',' + row.finish_m ,end=',')
        print(row.s,end=',')
print()
for hour in range(22,24):
    for minute in range(0, 60, 15):
        if hour == 23 and minute > 30:
            break
        time_h1 = hour
        time_h2 = hour
        time_m1 = minute
        time_m2 = minute + 15
        start_m = hour * 60 + minute
        finish_m = start_m + 15
        if time_m2 == 60:
            time_h2 += 1
            time_m2 = 0
        print(f"{time_h1},{time_h2},{time_m1},{time_m2},{start_m+1},{finish_m},")
output.close()

    