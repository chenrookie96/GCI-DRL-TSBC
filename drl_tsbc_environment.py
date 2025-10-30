"""
DRL-TSBC 双向公交仿真环境
基于work_env.py改造，支持双向同时仿真
修复了Pandas 2.x兼容性问题（使用pd.concat替代DataFrame.append）
"""

import pandas as pd
import numpy as np
import random
import torch
import copy
import time
import os
import sys
from queue import Queue
import warnings
warnings.filterwarnings("ignore")

np.random.seed(10086)

# ==================== 全局参数配置 ====================
# 训练频率
train_counter = 5

# 营运时间
first_time = "06:00"
last_time = "21:00"

# 均匀排班配置
start_time = "10:00"
end_time = "16:00"
ideal_interval = 12
avg_flag = 0  # 0=不使用，1=特定区间等间隔，2=全局方差最小化，3=特定区间方差最小化

# 发车间隔约束
min_Interval = 5
max_Interval = 22

# DQN参数
epsilon = 0.9  # 原始代码：90%使用网络，10%随机

# 奖励函数参数
mu = 5000  # 等待时间归一化
delta = 200  # 发车次数差异归一化
beta = 0.2  # 滞留乘客惩罚权重
zeta = 0.002  # 发车次数平衡权重
alpha = 1  # 注意：pn_on_max=48已包含站立系数(32×1.5)，所以这里alpha=1

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ==================== Station类 ====================
class Station:
    """车站类 - 管理乘客到达和等待"""
    
    def __init__(self, station_number, passenger_info_path, first_minute_th):
        self.station_number = station_number
        self.all_passenger_info_path = passenger_info_path
        self.all_passenger_info_dataframe = pd.read_csv(passenger_info_path)
        
        # 初始化乘客列表
        self.init_frist_minute_passenger = []
        self.right_minute_passenger = []
        self.next_minute_passenger = []
        
        # 按站点分组乘客数据
        self.all_station_all_minute_passenger = []
        for i in range(self.station_number):
            station_passengers = self.all_passenger_info_dataframe[
                self.all_passenger_info_dataframe["Boarding station"] == i
            ]
            self.all_station_all_minute_passenger.append(station_passengers)
        
        # 初始化第一分钟的乘客
        for i in range(self.station_number):
            first_min_passengers = self.all_station_all_minute_passenger[i][
                self.all_station_all_minute_passenger[i]["Arrival time"] <= first_minute_th
            ]
            self.init_frist_minute_passenger.append(first_min_passengers)
        
        # 初始化下一分钟的乘客
        for i in range(self.station_number):
            next_min_passengers = self.all_station_all_minute_passenger[i][
                self.all_station_all_minute_passenger[i]["Arrival time"] == (first_minute_th + 1)
            ]
            self.next_minute_passenger.append(next_min_passengers)
        
        self.right_minute_passenger = self.init_frist_minute_passenger.copy()
    
    def reset(self, first_minute_th):
        """重置车站状态"""
        self.all_passenger_info_dataframe = pd.read_csv(self.all_passenger_info_path)
        
        self.init_frist_minute_passenger = []
        self.right_minute_passenger = []
        self.next_minute_passenger = []
        self.all_station_all_minute_passenger = []
        
        for i in range(self.station_number):
            station_passengers = self.all_passenger_info_dataframe[
                self.all_passenger_info_dataframe["Boarding station"] == i
            ]
            self.all_station_all_minute_passenger.append(station_passengers)
        
        for i in range(self.station_number):
            first_min_passengers = self.all_station_all_minute_passenger[i][
                self.all_station_all_minute_passenger[i]["Arrival time"] <= first_minute_th
            ]
            self.init_frist_minute_passenger.append(first_min_passengers)
        
        for i in range(self.station_number):
            next_min_passengers = self.all_station_all_minute_passenger[i][
                self.all_station_all_minute_passenger[i]["Arrival time"] == (first_minute_th + 1)
            ]
            self.next_minute_passenger.append(next_min_passengers)
        
        self.right_minute_passenger = self.init_frist_minute_passenger.copy()
    
    def forward_one_step(self, right_minute_th):
        """时间前进一步"""
        for i in range(self.station_number):
            self.right_minute_passenger[i] = pd.concat([
                self.right_minute_passenger[i],
                self.next_minute_passenger[i]
            ], ignore_index=True)
        
        self.next_minute_passenger = []
        for i in range(self.station_number):
            next_min_passengers = self.all_station_all_minute_passenger[i][
                self.all_station_all_minute_passenger[i]["Arrival time"] == (right_minute_th + 2)
            ]
            self.next_minute_passenger.append(next_min_passengers)
    
    def Estimated_psger_procs(self, station_no, time_start, time_finish):
        """估算特定时间段内到达指定站点的乘客"""
        psger_procs1 = self.all_station_all_minute_passenger[station_no][
            self.all_station_all_minute_passenger[station_no]["Arrival time"] > time_start
        ]
        psger_procs2 = psger_procs1[psger_procs1["Arrival time"] <= time_finish]
        return psger_procs2



# ==================== Bus类 ====================
class Bus:
    """公交车类 - 管理车辆状态和乘客上下车"""
    
    def __init__(self, pn_on_max, station_number, start_time, trf_con, right_minute_th):
        self.start_time = start_time
        self.station_number = station_number
        self.pn_on_max = pn_on_max
        self.trf_con = trf_con
        
        # 车上乘客
        self.pn_on = pd.DataFrame(columns=[
            "Label", "Boarding time", "Boarding station", "Alighting station", "Arrival time"
        ])
        self.pn_on_num = 0
        
        # 车辆位置
        self.position = np.zeros(station_number)
        self.position[0] = 1
        
        self.pass_position = np.zeros(station_number)
        self.pass_position[0] = 1
        
        self.left_every_station_pn_on = np.zeros(station_number)
        
        self.there_label = 1
        self.pass_minute = 0
        self.arrv_mark = 0
        self.onlie_mark = 1
        self.bus_miage = 0
        self.cant_taken_once = 0
        
        # 预计到达时间
        self.Estimated_arrival_time = np.zeros(station_number)
        self.Estimated_arrival_time[0] = right_minute_th
        for i in range(1, station_number):
            self.Estimated_arrival_time[i] = right_minute_th + sum(
                trf_con.iloc[(right_minute_th // 15), 6:6+i]
            ) - self.pass_minute
        
        # 待处理乘客
        self.To_be_process_all_station = []
        self.Left_after_pass_the_station = []
        for i in range(station_number):
            empty_df = pd.DataFrame(columns=[
                "Label", "Boarding time", "Boarding station", "Alighting station", "Arrival time"
            ])
            self.To_be_process_all_station.append(empty_df.copy())
            self.Left_after_pass_the_station.append(empty_df.copy())
    
    def forward_one_step(self, right_minute_th):
        """车辆前进一步"""
        self.there_label = 0
        arrival_flag = 0
        
        for i in range(len(self.position) - 1):
            if self.position[i] == 1:
                self.bus_miage = self.bus_miage + (1 / self.trf_con.iloc[(right_minute_th // 15), 6 + i])
                self.pass_minute = self.pass_minute + 1
                
                if self.bus_miage >= 1:
                    self.pass_minute = 0
                    self.bus_miage = self.bus_miage - 1
                    self.position[i + 1] = 1
                    self.pass_position[i + 1] = 1
                    self.position[i] = 0
                    self.there_label = 1
                    
                    if sum(self.pass_position) == self.station_number:
                        self.arrv_mark = 1
                        arrival_flag = 1
                    break
                break
        
        return arrival_flag
    
    def up_down(self, station, right_minute_th):
        """处理乘客上下车"""
        for i in range(station.station_number):
            if self.there_label == 1 and self.position[i] == 1:
                # 乘客下车
                off_passengers = self.pn_on[self.pn_on["Alighting station"] == i]
                self.pn_on = pd.concat([self.pn_on, off_passengers], ignore_index=True)
                self.pn_on = self.pn_on.drop_duplicates(
                    subset=["Label", "Boarding time", "Boarding station", "Alighting station"],
                    keep=False
                )
                
                # 乘客上车
                if len(station.right_minute_passenger[i]) + len(self.pn_on) <= self.pn_on_max:
                    self.pn_on = pd.concat([
                        self.pn_on,
                        station.right_minute_passenger[i]
                    ], ignore_index=True)
                    station.right_minute_passenger[i] = pd.DataFrame(columns=[
                        "Label", "Boarding time", "Boarding station", "Alighting station", "Arrival time"
                    ])
                else:
                    append_mark = self.pn_on_max - len(self.pn_on)
                    self.pn_on = pd.concat([
                        self.pn_on,
                        station.right_minute_passenger[i].iloc[:append_mark, :]
                    ], ignore_index=True)
                    station.right_minute_passenger[i] = station.right_minute_passenger[i].iloc[append_mark:, :]
                break



# ==================== DirectionSystem类 ====================
class DirectionSystem:
    """单向公交系统 - 管理一个方向的所有公交车"""
    
    def __init__(self, station, pn_on_max, trf_con):
        self.station = station
        self.station_number = station.station_number
        self.pn_on_max = pn_on_max
        self.trf_con = trf_con
        
        self.bus_online = []
        self.Interval = 0
        self.end_label = 0
        
        # 统计变量
        self.All_psger_wait_time = 0
        self.All_cap_take = 0
        self.All_cap_uesd = 0
        self.Cant_taken_once = 0
        self.Cap_used = 0
        self.if_depart_wait_time = 0
        
        self.last_departure_cap_use = Queue()
    
    def Departure(self, right_minute_th):
        """发车"""
        new_bus = Bus(self.pn_on_max, self.station_number, right_minute_th, self.trf_con, right_minute_th)
        self.bus_online.append(new_bus)
        
        cap_use_now = self.All_cap_uesd
        self.All_cap_take += self.pn_on_max * (self.station_number - 1)
        
        # 处理第一辆车或后续车辆
        if len(self.bus_online) == 1:
            self._process_first_bus(right_minute_th)
        elif len(self.bus_online) > 1:
            self._process_subsequent_bus(right_minute_th)
        
        self.last_departure_cap_use.put(self.All_cap_uesd - cap_use_now)
    
    def _process_first_bus(self, right_minute_th):
        """处理第一辆车的乘客预测"""
        bus = self.bus_online[0]
        
        for i in range(self.station_number):
            # 计算待处理乘客
            bus.To_be_process_all_station[i] = pd.concat([
                self.station.right_minute_passenger[i],
                self.station.Estimated_psger_procs(i, right_minute_th, bus.Estimated_arrival_time[i])
            ], ignore_index=True)
            
            # 下车
            off_passengers = bus.pn_on[bus.pn_on["Alighting station"] == i]
            bus.pn_on = pd.concat([bus.pn_on, off_passengers], ignore_index=True)
            bus.pn_on = bus.pn_on.drop_duplicates(
                subset=["Label", "Boarding time", "Boarding station", "Alighting station"],
                keep=False
            )
            
            # 上车
            if len(bus.To_be_process_all_station[i]) + len(bus.pn_on) <= bus.pn_on_max:
                bus.pn_on = pd.concat([
                    bus.pn_on,
                    bus.To_be_process_all_station[i]
                ], ignore_index=True)
                bus.Left_after_pass_the_station[i] = pd.DataFrame(columns=[
                    "Label", "Boarding time", "Boarding station", "Alighting station", "Arrival time"
                ])
                self.All_psger_wait_time += (
                    bus.Estimated_arrival_time[i] * len(bus.To_be_process_all_station[i])
                    - np.sum(bus.To_be_process_all_station[i].iloc[:, 4])
                )
                self.All_cap_uesd += len(bus.pn_on)
                bus.left_every_station_pn_on[i] = len(bus.pn_on)
            else:
                append_mark = bus.pn_on_max - len(bus.pn_on)
                bus.pn_on = pd.concat([
                    bus.pn_on,
                    bus.To_be_process_all_station[i].iloc[:append_mark, :]
                ], ignore_index=True)
                bus.Left_after_pass_the_station[i] = bus.To_be_process_all_station[i].iloc[append_mark:, :]
                self.All_psger_wait_time += (
                    bus.Estimated_arrival_time[i] * append_mark
                    - np.sum(bus.To_be_process_all_station[i].iloc[:append_mark, 4])
                )
                self.All_cap_uesd += len(bus.pn_on)
                bus.left_every_station_pn_on[i] = len(bus.pn_on)
                bus.cant_taken_once += len(bus.To_be_process_all_station[i]) - append_mark
    
    def _process_subsequent_bus(self, right_minute_th):
        """处理后续车辆的乘客预测"""
        bus = self.bus_online[-1]
        prev_bus = self.bus_online[-2]
        
        for i in range(self.station_number):
            # 计算待处理乘客（包括上一辆车的滞留乘客）
            bus.To_be_process_all_station[i] = pd.concat([
                prev_bus.Left_after_pass_the_station[i],
                self.station.Estimated_psger_procs(
                    i, prev_bus.Estimated_arrival_time[i], bus.Estimated_arrival_time[i]
                )
            ], ignore_index=True)
            
            # 下车
            off_passengers = bus.pn_on[bus.pn_on["Alighting station"] == i]
            bus.pn_on = pd.concat([bus.pn_on, off_passengers], ignore_index=True)
            bus.pn_on = bus.pn_on.drop_duplicates(
                subset=["Label", "Boarding time", "Boarding station", "Alighting station"],
                keep=False
            )
            
            # 上车
            if len(bus.To_be_process_all_station[i]) + len(bus.pn_on) <= bus.pn_on_max:
                bus.pn_on = pd.concat([
                    bus.pn_on,
                    bus.To_be_process_all_station[i]
                ], ignore_index=True)
                bus.Left_after_pass_the_station[i] = pd.DataFrame(columns=[
                    "Label", "Boarding time", "Boarding station", "Alighting station", "Arrival time"
                ])
                self.All_psger_wait_time += (
                    bus.Estimated_arrival_time[i] * len(bus.To_be_process_all_station[i])
                    - np.sum(bus.To_be_process_all_station[i].iloc[:, 4])
                )
                self.All_cap_uesd += len(bus.pn_on)
                bus.left_every_station_pn_on[i] = len(bus.pn_on)
            else:
                append_mark = bus.pn_on_max - len(bus.pn_on)
                bus.pn_on = pd.concat([
                    bus.pn_on,
                    bus.To_be_process_all_station[i].iloc[:append_mark, :]
                ], ignore_index=True)
                bus.Left_after_pass_the_station[i] = bus.To_be_process_all_station[i].iloc[append_mark:, :]
                self.All_psger_wait_time += (
                    bus.Estimated_arrival_time[i] * append_mark
                    - np.sum(bus.To_be_process_all_station[i].iloc[:append_mark, 4])
                )
                self.All_cap_uesd += len(bus.pn_on)
                bus.left_every_station_pn_on[i] = len(bus.pn_on)
                bus.cant_taken_once += len(bus.To_be_process_all_station[i]) - append_mark

    
    def get_state_components(self, right_minute_th, other_direction_departures=0):
        """获取该方向的状态分量（4维）
        
        参数:
        - right_minute_th: 当前时间
        - other_direction_departures: 另一方向的发车次数（用于计算发车次数差）
        """
        if len(self.bus_online) == 0:
            return torch.zeros(4)
        
        # 计算预计到达时间
        right_depart_estimated_arrival_time = np.zeros(self.station_number)
        right_depart_estimated_arrival_time[0] = right_minute_th
        
        for i in range(1, self.station_number):
            right_depart_estimated_arrival_time[i] = right_minute_th + sum(
                self.trf_con.iloc[(right_minute_th // 15), 6:6+i]
            )
        
        # 计算待处理乘客
        Passenger_tobe_proc_per_station = []
        Passenger_num_on_board_leaving_station = pd.DataFrame(columns=[
            "Label", "Boarding time", "Boarding station", "Alighting station", "Arrival time"
        ])
        Passenger_num_on_board_leaving_station_num = []
        This_departure_pre_wait_time = 0
        Passenger_cant_takeon_once = 0
        
        for i in range(self.station_number):
            # 待处理乘客
            tobe_proc = pd.concat([
                self.bus_online[-1].Left_after_pass_the_station[i],
                self.station.Estimated_psger_procs(
                    i,
                    self.bus_online[-1].Estimated_arrival_time[i],
                    right_depart_estimated_arrival_time[i]
                )
            ], ignore_index=True)
            Passenger_tobe_proc_per_station.append(tobe_proc)
            
            # 下车
            off_passengers = Passenger_num_on_board_leaving_station[
                Passenger_num_on_board_leaving_station["Alighting station"] == i
            ]
            Passenger_num_on_board_leaving_station = pd.concat([
                Passenger_num_on_board_leaving_station,
                off_passengers
            ], ignore_index=True)
            Passenger_num_on_board_leaving_station = Passenger_num_on_board_leaving_station.drop_duplicates(
                subset=["Label", "Boarding time", "Boarding station", "Alighting station"],
                keep=False
            )
            
            # 上车
            if len(Passenger_tobe_proc_per_station[i]) + len(Passenger_num_on_board_leaving_station) <= self.pn_on_max:
                Passenger_num_on_board_leaving_station = pd.concat([
                    Passenger_num_on_board_leaving_station,
                    Passenger_tobe_proc_per_station[i]
                ], ignore_index=True)
                This_departure_pre_wait_time += (
                    right_depart_estimated_arrival_time[i] * len(Passenger_tobe_proc_per_station[i])
                    - np.sum(Passenger_tobe_proc_per_station[i].iloc[:, 4])
                )
            else:
                append_mark = self.pn_on_max - len(Passenger_num_on_board_leaving_station)
                Passenger_num_on_board_leaving_station = pd.concat([
                    Passenger_num_on_board_leaving_station,
                    Passenger_tobe_proc_per_station[i].iloc[:append_mark, :]
                ], ignore_index=True)
                This_departure_pre_wait_time += (
                    right_depart_estimated_arrival_time[i] * append_mark
                    - np.sum(Passenger_tobe_proc_per_station[i].iloc[:append_mark, 4])
                )
                Passenger_cant_takeon_once += len(Passenger_tobe_proc_per_station[i]) - append_mark
            
            Passenger_num_on_board_leaving_station_num.append(len(Passenger_num_on_board_leaving_station))
        
        # 保存用于奖励计算
        self.Cant_taken_once = Passenger_cant_takeon_once
        self.Cap_used = np.sum(Passenger_num_on_board_leaving_station_num)
        self.if_depart_wait_time = This_departure_pre_wait_time
        
        # 返回4维状态：[满载率, 等待时间, 容量利用率, 发车次数差]（论文公式2.7-2.10）
        # x_m^1: 满载率 C_max^m / C_max
        max_load_rate = np.max(Passenger_num_on_board_leaving_station_num) / self.pn_on_max if len(Passenger_num_on_board_leaving_station_num) > 0 else 0
        
        # x_m^2: 归一化等待时间 W_m / μ
        wait_time_norm = This_departure_pre_wait_time / mu
        
        # x_m^3: 客运容量利用率 o_m / e_m
        total_load_rate = np.sum(Passenger_num_on_board_leaving_station_num) / (alpha * self.pn_on_max * (self.station_number - 1))
        
        # x_m^4: 发车次数差 (c_m^up - c_m^down) / δ（论文公式2.10）
        departure_diff_norm = (len(self.bus_online) - other_direction_departures) / delta
        
        return torch.tensor([max_load_rate, wait_time_norm, total_load_rate, departure_diff_norm], dtype=torch.float32)
    
    def Action(self, departure_factor, right_minute_th, min_Interval, max_Interval,
               start_minute_th=None, end_minute_th=None, ideal_interval=None, avg_flag=0):
        """决定是否发车"""
        # 结束标志
        if right_minute_th > int((int(last_time[:-3]) - int(self.trf_con.iloc[0, 0])) * 60 + (int(last_time[-2:]) - int(self.trf_con.iloc[0, 1]))) + 50:
            self.end_label = 1
        
        # 均匀排班（特定区间等间隔）
        if avg_flag == 1 and start_minute_th and end_minute_th and ideal_interval:
            if start_minute_th <= right_minute_th <= end_minute_th and self.Interval == ideal_interval - 1:
                self.Departure(right_minute_th)
                self.Interval = 0
                return
        
        # 首班车、末班车或达到最大间隔强制发车
        first_minute_th = (int(first_time[:-3]) - int(self.trf_con.iloc[0, 0])) * 60 + (int(first_time[-2:]) - int(self.trf_con.iloc[0, 1]))
        last_minute_th = (int(last_time[:-3]) - int(self.trf_con.iloc[0, 0])) * 60 + (int(last_time[-2:]) - int(self.trf_con.iloc[0, 1]))
        
        if (right_minute_th == first_minute_th or right_minute_th == last_minute_th or 
            self.Interval == max_Interval) and right_minute_th < last_minute_th + 1:
            self.Departure(right_minute_th)
            self.Interval = 0
        else:
            if self.Interval < min_Interval or departure_factor == 0 or right_minute_th > last_minute_th:
                self.Interval += 1
            else:
                self.Departure(right_minute_th)
                self.Interval = 0
    
    def step_forward(self, right_minute_th):
        """环境前进一步"""
        # 所有车辆上下车
        for bus in self.bus_online:
            bus.up_down(self.station, right_minute_th)
        
        # 所有车辆前进
        for bus in self.bus_online:
            arrival_flag = bus.forward_one_step(right_minute_th)
            if arrival_flag == 1:
                # 到达终点，减去运力
                self.All_cap_take -= self.pn_on_max * (self.station_number - 1)
                if not self.last_departure_cap_use.empty():
                    last_cap_use = self.last_departure_cap_use.get()
                    self.All_cap_uesd -= last_cap_use
        
        # 车站时间前进
        self.station.forward_one_step(right_minute_th)



# ==================== BidirectionalBusSystem类 ====================
class BidirectionalBusSystem:
    """双向公交系统 - 同时管理上行和下行"""
    
    def __init__(self, upward_station, downward_station, pn_on_max, trf_con_up, trf_con_down):
        self.upward_system = DirectionSystem(upward_station, pn_on_max, trf_con_up)
        self.downward_system = DirectionSystem(downward_station, pn_on_max, trf_con_down)
        self.end_label = 0
    
    def get_full_state(self, right_minute_th):
        """获取完整的10维状态（论文公式2.6-2.14）"""
        # 时间特征（2维）- 论文公式2.6
        time_hour = (right_minute_th // 60) / 24
        time_minute = (right_minute_th % 60) / 60
        
        # 获取发车次数
        num_departures_up = len(self.upward_system.bus_online)
        num_departures_down = len(self.downward_system.bus_online)
        
        # 上行状态（4维）- 论文公式2.7-2.10
        upward_state = self.upward_system.get_state_components(right_minute_th, num_departures_down)
        
        # 下行状态（4维）- 论文公式2.11-2.14
        downward_state = self.downward_system.get_state_components(right_minute_th, num_departures_up)
        
        # 拼接为10维状态
        full_state = torch.cat([
            torch.tensor([time_hour, time_minute], dtype=torch.float32),
            upward_state,
            downward_state
        ])
        
        return full_state.numpy()
    
    def Action(self, action_tuple, right_minute_th, min_Interval, max_Interval,
               start_minute_th=None, end_minute_th=None, ideal_interval=None, avg_flag=0):
        """执行双向发车动作"""
        aup, adown = action_tuple
        
        # 上行发车
        self.upward_system.Action(
            aup, right_minute_th, min_Interval, max_Interval,
            start_minute_th, end_minute_th, ideal_interval, avg_flag
        )
        
        # 下行发车
        self.downward_system.Action(
            adown, right_minute_th, min_Interval, max_Interval,
            start_minute_th, end_minute_th, ideal_interval, avg_flag
        )
        
        # 更新结束标志
        if self.upward_system.end_label == 1 and self.downward_system.end_label == 1:
            self.end_label = 1
    
    def calculate_reward(self, action_tuple, omega):
        """
        严格按照论文公式2.16-2.17计算奖励
        
        论文公式：
        - 公式2.16 (上行): 
          a^up=0: r_up = 1 - (o_up/e_up) - (ω×W_up) - (β×d_up) + ζ(c_up - c_down)
          a^up=1: r_up = (o_up/e_up) - (β×d_up) - ζ(c_up - c_down)
        
        - 公式2.17 (下行):
          a^down=0: r_down = 1 - (o_down/e_down) - (ω×W_down) - (β×d_down) - ζ(c_up - c_down)
          a^down=1: r_down = (o_down/e_down) - (β×d_down) + ζ(c_up - c_down)
        
        变量对应关系：
        - o_m: Cap_used (实际消耗容量)
        - e_m: alpha × pn_on_max × (station_number - 1) (总容量)
        - W_m: if_depart_wait_time (等待时间，需要归一化)
        - d_m: Cant_taken_once (滞留乘客数)
        - c_m: len(bus_online) (发车次数)
        """
        aup, adown = action_tuple
        
        # 计算容量利用率 o_m/e_m（论文公式2.9）
        e_up = alpha * self.upward_system.pn_on_max * (self.upward_system.station_number - 1)
        e_down = alpha * self.downward_system.pn_on_max * (self.downward_system.station_number - 1)
        
        o_up_div_e_up = self.upward_system.Cap_used / e_up
        o_down_div_e_down = self.downward_system.Cap_used / e_down
        
        # 等待时间（使用原始值，不归一化）
        # 注意：状态空间中已经归一化过了，奖励函数中应使用原始值
        W_up = self.upward_system.if_depart_wait_time
        W_down = self.downward_system.if_depart_wait_time
        
        # 滞留乘客数 d_m
        d_up = self.upward_system.Cant_taken_once
        d_down = self.downward_system.Cant_taken_once
        
        # 发车次数 c_m
        c_up = len(self.upward_system.bus_online)
        c_down = len(self.downward_system.bus_online)
        
        # 论文规范参数
        # beta = 0.2 (全局变量)
        # omega = 传入参数
        # zeta = 0.002 (全局变量)
        
        # 上行奖励（论文公式2.16）
        if aup == 0:  # 不发车
            r_up = 1 - o_up_div_e_up - (omega * W_up) - (beta * d_up) + (zeta * (c_up - c_down))
        else:  # 发车
            r_up = o_up_div_e_up - (beta * d_up) - (zeta * (c_up - c_down))
        
        # 下行奖励（论文公式2.17）
        if adown == 0:  # 不发车
            r_down = 1 - o_down_div_e_down - (omega * W_down) - (beta * d_down) - (zeta * (c_up - c_down))
        else:  # 发车
            r_down = o_down_div_e_down - (beta * d_down) + (zeta * (c_up - c_down))
        
        # 总奖励（论文公式2.15）
        reward = r_up + r_down
        
        return reward
    
    def step_forward(self, right_minute_th):
        """环境前进一步"""
        self.upward_system.step_forward(right_minute_th)
        self.downward_system.step_forward(right_minute_th)
    
    def calculate_complete_awt(self, direction='up', total_passengers=0, 
                               first_minute_th=0, last_minute_th=0):
        """
        计算乘客平均等待时间（AWT）- 方案B
        
        定义：所有接触到公交服务的乘客的平均等待时间
        包括：已上车乘客 + 被滞留乘客
        
        Args:
            direction: 'up' 或 'down'
            total_passengers: 总乘客数（未使用，保留接口兼容性）
            first_minute_th: 开始时间（未使用，保留接口兼容性）
            last_minute_th: 结束时间（未使用，保留接口兼容性）
            
        Returns:
            平均等待时间（分钟）
            
        注意：
        - 统计已上车乘客：成功上车的乘客
        - 统计被滞留乘客：等到公交但因车满未能上车的乘客
        - 不统计还在等待的乘客：到episode结束还没等到公交的乘客
        """
        import numpy as np
        
        system = self.upward_system if direction == 'up' else self.downward_system
        
        # 1. 已上车乘客的总等待时间
        boarded_wait_time = system.All_psger_wait_time
        
        # 已上车乘客数
        boarded_count = 0
        for bus in system.bus_online:
            boarded_count += len(bus.pn_on)
        
        # 2. 被滞留乘客的等待时间
        stranded_wait_time = 0
        stranded_count = 0
        for bus in system.bus_online:
            for i in range(system.station_number):
                left_passengers = bus.Left_after_pass_the_station[i]
                if len(left_passengers) > 0:
                    stranded_count += len(left_passengers)
                    # 等待时间 = 公交到达时间 - 乘客到达时间
                    stranded_wait_time += (
                        bus.Estimated_arrival_time[i] * len(left_passengers)
                        - np.sum(left_passengers.iloc[:, 4])
                    )
        
        # 总等待时间和总人数
        total_wait_time = boarded_wait_time + stranded_wait_time
        serviced_passengers = boarded_count + stranded_count
        
        # 平均等待时间 = 总等待时间 / (已上车 + 被滞留)
        awt = total_wait_time / serviced_passengers if serviced_passengers > 0 else 0
        
        return awt
