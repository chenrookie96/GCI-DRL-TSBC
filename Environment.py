import pandas as pd
import numpy as  np
import random
import torch
import copy
from RL_brain import DQN  #DQN
import time
import sys
import matplotlib.pyplot as plt
from queue import *
import warnings
warnings.filterwarnings("ignore")
#coding:utf-8
__all__ = ["Environment_Calculation"]

np.random.seed(10086)

train_counter = 5 # 记忆满后每五步训练一次，若模型不收敛，可增大此值，若收敛过慢，可减小此值

# 初始发车时间，最后发车时间
first_time="06:00"
last_time="21:00"

# 特定区间内控制均匀排班、均匀排班间隔  or  方差最小化均匀调整（特定区间）
start_time = "10:00"
end_time = "11:00"
ideal_interval = 12  # 理想发车间隔为12分钟

avg_flag = 0  # 均匀排班控制 0--不使用；1--特定区间等间隔排班；2--方差最小化均匀调整（全局） 3--方差最小化均匀调整（特定区间）

station_num = 24  # 线路的站数

# 定义线路号，方向
busline = 208
direction = 1
# 定义omega_factor超参数，用于控制总发车次数，omega_factor越大发车次数越少
omega_factor = 1000

pn_on_max = 47 # 每辆车的最大运载人数

alpha = 1 # 这个参数表示最大能超载多少的系数，如若为1.5则实际可搭载1.5倍人数，若在最大运载人数已经体现，可设置为1


weight_time_wait_time = 5000  #这个是一个用于归一化等待时间的参数，不需设置

min_Interval = 5 # 最小发车间隔

max_Interval =22 # 最大发车间隔

epsilon = 0.9   

All_passenger_wait_time = 0

departure_label = 0

All_cap_out = 0

All_cap_uesd = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 交通情况
trf_path="./test_data/" + str(busline) + "/traffic-" + str(direction) +".csv"
trf_con = pd.DataFrame(pd.read_csv(trf_path))


passenger_info_path = "./test_data/" + str(busline) + "/passenger_dataframe_direction" + str(direction) +".csv"

model_save_path = "./test_data/" + str(busline) + "/model" + str(direction) + "_" + str(omega_factor) + ".pth"

max_episode = 500  # 最多训练代数
omega = 1 / omega_factor  # 两个用于计算reward的系数
beta = 0.2

first_minute_th = (int(first_time[:-3]) - int(trf_con.iloc[0, 0])) * 60 + (int(first_time[-2:]) - int(trf_con.iloc[0, 1]))
last_minute_th = (int(last_time[:-3]) - int(trf_con.iloc[0, 0])) * 60 + (int(last_time[-2:]) - int(trf_con.iloc[0, 1]))
right_minute_th = first_minute_th
start_minute_th = (int(start_time[:-3]) - int(trf_con.iloc[0, 0])) * 60 + (int(start_time[-2:]) - int(trf_con.iloc[0, 1]))
end_minute_th = (int(end_time[:-3]) - int(trf_con.iloc[0, 0])) * 60 + (int(end_time[-2:]) - int(trf_con.iloc[0, 1]))



class Station:  # 用于模拟和管理公共交通系统中车站的乘客流
    def __init__(self, station_number=station_num,all_passenger_info_path=passenger_info_path):

        self.station_number = station_number  #站点数

        self.sum1 = 0

        self.init_frist_minute_passenger = [] #发车时，站点上等待的乘客

        self.right_minute_passenger = [] #当前分钟，在车站上等待的乘客

        self.next_minute_passenger = [] #下一分钟，即将到达车站的乘客

        self.all_passenger_info_path= all_passenger_info_path  # 乘客信息的路径

        self.all_passenger_info_dataframe =  pd.DataFrame(pd.read_csv(all_passenger_info_path)) # 从csv文件中读取的所有乘客信息的DataFrame

        self.all_station_all_minute_passenger = []  # 所有站点，每分钟上车乘客的信息([[all_passenger_info_dataframe[all_passenger_info_dataframe["Boarding station"]==1]],[all_passenger_info_dataframe[all_passenger_info_dataframe["Boarding station"]==2]],[],[],])

        for i in range(self.station_number): # 遍历所有站点，每分钟上车乘客
            self.all_station_all_minute_passenger.append(self.all_passenger_info_dataframe[self.all_passenger_info_dataframe["Boarding station"]==i])#每站，所有分钟的
        for i in range(self.station_number): #第一分钟，所有站点的乘客
            self.init_frist_minute_passenger.append(self.all_station_all_minute_passenger[i][self.all_station_all_minute_passenger[i]["Arrival time"]<=first_minute_th])


        for i in range(self.station_number): #下一分钟，即将到达车站的乘客
            self.next_minute_passenger.append(self.all_station_all_minute_passenger[i][self.all_station_all_minute_passenger[i]["Arrival time"] == (first_minute_th+1)])
        self.right_minute_passenger =  self.init_frist_minute_passenger

    def reset(self): # 重置车站的状态，清空各个乘客列表，并重新从CSV文件加载乘客信息
        self.init_frist_minute_passenger = []  # 发车时，站点上等待的乘客

        self.right_minute_passenger = []  # 当前分钟，在车站上等待的乘客

        self.next_minute_passenger = []  # 下一分钟，即将到达车站的乘客

        self.all_passenger_info_dataframe = pd.DataFrame(pd.read_csv(self.all_passenger_info_path))

        self.all_station_all_minute_passenger = []  # 所有站点，每分钟上车乘客([[all_passenger_info_dataframe[all_passenger_info_dataframe["Boarding station"]==1]],[all_passenger_info_dataframe[all_passenger_info_dataframe["Boarding station"]==2]],[],[],])

        for i in range(self.station_number):  # 所有站点，每分钟上车乘客
            self.all_station_all_minute_passenger.append(
                self.all_passenger_info_dataframe[self.all_passenger_info_dataframe["Boarding station"] == i])  # 每站，所有分钟的
        for i in range(self.station_number):  # 第一分钟，所有站点的乘客
            self.init_frist_minute_passenger.append(self.all_station_all_minute_passenger[i][
                                                        self.all_station_all_minute_passenger[i][
                                                            "Arrival time"] <= first_minute_th])
        for i in range(self.station_number):  # 下一分钟，即将到达车站的乘客
            self.next_minute_passenger.append(self.all_station_all_minute_passenger[i][
                                                  self.all_station_all_minute_passenger[i]["Arrival time"] == (
                                                              first_minute_th + 1)])
        self.right_minute_passenger = self.init_frist_minute_passenger

    def forward_one_step(self): #模拟时间前进一步，将下一分钟乘客加入到当前分钟的乘客中，并准备下一分钟即将到达的乘客信息。
        for i in range(self.station_number):
            self.right_minute_passenger[i]=pd.concat([self.right_minute_passenger[i],self.next_minute_passenger[i]])
        self.next_minute_passenger = []
        #print(right_minute_th, "每车人数", [len(k) for k in self.right_minute_passenger])
        for i in range(self.station_number):  # 下一分钟，即将到达车站的乘客
            self.next_minute_passenger.append(self.all_station_all_minute_passenger[i][
                                                  self.all_station_all_minute_passenger[i]["Arrival time"] == (
                                                              right_minute_th + 2)])


    def Estimated_psger_procs(self,station_no,time_start,time_finish): #计算在特定时间段内到达指定站点的乘客信息
        psger_procs1=self.all_station_all_minute_passenger[station_no][self.all_station_all_minute_passenger[station_no]["Arrival time"]>time_start]
        psger_procs2 = psger_procs1[psger_procs1["Arrival time"]<=time_finish]
        return psger_procs2


class Bus: #用于模拟和管理公共交通系统中公交的行为、状态
    def __init__(self,pn_on_max=pn_on_max, station_number=station_num,start_time= right_minute_th): #三个参数：最大载客数、站点数量、出发时间


        self.start_time =start_time #发出的时间

        self.station_number = station_number

        self.pn_on_max = pn_on_max #最大人数

        self.pn_on =pd.DataFrame(columns=["Label", "Boarding time","Boarding station","Alighting station","Arrival time"])   #车上人员

        self.pn_on_num = 0 #车上人数

        self.position = np.zeros(station_number) #车辆位置
        self.position[0]=1

        self.pass_position = np.zeros(station_number) #车辆通过的站点，用于计算运力
        self.pass_position[0]=1

        self.left_every_station_pn_on = np.zeros(station_number) #记录车辆在每个站点下车的乘客数量

        self.there_label = 1

        self.pass_minute = 0

        self.Passenger_on_board_leaving_station = pd.DataFrame(columns=["Label", "Boarding time","Boarding station","Alighting station","Arrival time"])

        self.arrv_mark = 0 #到达终点站

        self.onlie_mark =1 #目前仍未到达终点站

        self.bus_miage = 0 #走过某站后的距离

        self.cant_taken_once = 0

        self.Estimated_arrival_time = np.zeros(station_number)  #预计到达某站的时间
        self.Estimated_arrival_time[0] = right_minute_th
        for i in range(1, station_number):
            #预计到达各个的时间等于当前时间加上交通状况 passminute在这里为0
            self.Estimated_arrival_time[i] = right_minute_th +sum(trf_con.iloc[(right_minute_th // 15), 6:6+i]) - self.pass_minute

        self.To_be_process_all_station = []
        for i in range(0, station_number): #向这辆车的中加入所有乘客的信息标签
            self.To_be_process_all_station.append(pd.DataFrame(columns=["Label", "Boarding time","Boarding station","Alighting station","Arrival time"]))

        self.Left_after_pass_the_station = []
        for i in range(0, station_number):
            self.Left_after_pass_the_station.append(pd.DataFrame(
                columns=["Label", "Boarding time", "Boarding station", "Alighting station", "Arrival time"]))

    def forward_one_step(self): # 模拟车辆前进一个步骤后，更新车辆位置和通过站点的状态
        self.there_label = 0
        arrival_flag = 0
        for i in range(len(self.position) - 1):
            if self.position[i] == 1:
                self.bus_miage = self.bus_miage + (1 / trf_con.iloc[(right_minute_th // 15), 6 + i])
                self.pass_minute = self.pass_minute + 1

                if self.bus_miage >= 1:
                    self.pass_minute = 0
                    self.bus_miage = self.bus_miage - 1
                    self.position[i + 1] = 1
                    self.pass_position[i + 1] = 1  # 过站标志位

                    self.position[i] = 0
                    self.there_label = 1
                    if sum(self.pass_position)==self.station_number:
                        self.arrv_mark=1
                        arrival_flag = 1
                    break
                break
        return arrival_flag



    def up_down(self,station=Station): #处理乘客在特定站点的上下车。这个方法涉及较复杂的逻辑，包括计算等候时间、更新车上乘客信息、处理乘客上车时的容量限制等。

        global All_cap_out

        global All_passenger_wait_time

        global All_cap_uesd


        for i in range(station.station_number):
            if self.there_label ==1 and self.position[i]==1:
                #print("222222",(station.right_minute_passenger[i]),len(self.pn_on))
                #print(right_minute_th,"入站",len(self.pn_on))
                #乘客下车
                # print(self.pn_on)
                self.pn_on = self.pn_on.append(self.pn_on[self.pn_on["Alighting station"]==i])
                # print(self.pn_on)
                self.pn_on = self.pn_on.drop_duplicates(subset=["Label", "Boarding time","Boarding station","Alighting station"], keep=False)


                if len(station.right_minute_passenger[i])+len(self.pn_on)<=self.pn_on_max:
                    station.sum1 = station.sum1 + len(station.right_minute_passenger[i])
                    self.pn_on=pd.concat([self.pn_on,station.right_minute_passenger[i]])
                    All_cap_out = All_cap_out + self.pn_on_max
                    All_passenger_wait_time =All_passenger_wait_time + right_minute_th*len(station.right_minute_passenger[i])-np.sum(station.right_minute_passenger[i].iloc[:,4])
                    station.right_minute_passenger[i] = pd.DataFrame(columns=["Label", "Boarding time","Boarding station","Alighting station","Arrival time"])
                    All_cap_uesd = All_cap_uesd + len(self.pn_on)
                else:

                    station.sum1 = station.sum1 + len(station.right_minute_passenger[i])
                    append_mark=self.pn_on_max - len(self.pn_on)
                    self.pn_on = self.pn_on.append(station.right_minute_passenger[i].iloc[:append_mark,:])
                    All_cap_out = All_cap_out + self.pn_on_max

                    All_passenger_wait_time = All_passenger_wait_time + right_minute_th * append_mark - np.sum(station.right_minute_passenger[i].iloc[:append_mark, 4])

                    station.right_minute_passenger[i] = station.right_minute_passenger[i].iloc[append_mark:,:]
                    All_cap_uesd = All_cap_uesd + len(self.pn_on)





class BUS_LINE_SYSTEM:# 模拟和管理实际公交系统的运行
    def __init__(self, station_number=station_num, station=Station):
        self.station_number = station_number
        self.bus_online = []
        self.Interval = 0
        # self.Interval1 = 0
        # self.Interval0 = 0
        self.end_label = 0
        self.station = station
        self.bus_online_test = []
        self.wait_time = 0
        self.All_psger_wait_time = 0
        self.All_cap_take = 0
        self.All_cap_uesd = 0

        self.All_psger_wait_time_depart_once= 0
        self.All_cap_take_depart_once = 0
        self.All_cap_uesd_depart_once = 0
        self.Cant_taken_once  = 0
        self.Cap_used = 0
        self.if_depart_wait_time = 0
        self.last_departure_cap_use = Queue()

    def Departure(self,):
        self.bus_online.append(Bus(pn_on_max=pn_on_max,station_number=station_num,start_time= right_minute_th)) #在列表中加入一辆巴士
        cap_use_now = self.All_cap_uesd
        self.All_cap_take = self.All_cap_take + pn_on_max*(station_num-1) # 每辆车发车的时候增加总容量
        self.All_cap_take_depart_once = self.All_cap_take_depart_once + pn_on_max * (station_num - 1) # 每辆车发车的时候增加总容量
        if   len(self.bus_online) == 0:
            pass
        elif len(self.bus_online) == 1: #若只有一辆巴士
            for i in range(self.station_number):
                self.bus_online[0].To_be_process_all_station[i] = pd.concat([self.station.right_minute_passenger[i],self.station.Estimated_psger_procs(i, right_minute_th, self.bus_online[0].Estimated_arrival_time[i])]) #计算得到每一个站点将要上车的乘客是哪些

                self.bus_online[0].pn_on = self.bus_online[0].pn_on.append(self.bus_online[0].pn_on[self.bus_online[0].pn_on["Alighting station"] == i]) #计算要下车的乘客

                self.bus_online[0].pn_on = self.bus_online[0].pn_on.drop_duplicates(subset=["Label", "Boarding time", "Boarding station", "Alighting station"], keep=False) #在列表中删掉下车的乘客

                if len(self.bus_online[0].To_be_process_all_station[i]) + len(self.bus_online[0].pn_on) <= self.bus_online[0].pn_on_max: # 若没有出现滞留
                    self.bus_online[0].pn_on = pd.concat([self.bus_online[0].pn_on, self.bus_online[0].To_be_process_all_station[i]]) #加入将要处理的站点信息
                    self.bus_online[0].Left_after_pass_the_station[i] = pd.DataFrame(columns=["Label", "Boarding time", "Boarding station", "Alighting station", "Arrival time"]) #创建一个表示滞留乘客的
                    self.All_psger_wait_time = self.All_psger_wait_time +self.bus_online[0].Estimated_arrival_time[i]*len(self.bus_online[0].To_be_process_all_station[i]) - np.sum(self.bus_online[0].To_be_process_all_station[i].iloc[:,4]) #计算所有乘客所需的等待时间
                    self.All_psger_wait_time_depart_once = self.All_psger_wait_time_depart_once + self.bus_online[0].Estimated_arrival_time[i] * len(self.bus_online[0].To_be_process_all_station[i]) - np.sum(self.bus_online[0].To_be_process_all_station[i].iloc[:, 4])
                    self.All_cap_uesd = self.All_cap_uesd + len(self.bus_online[-1].pn_on)
                    self.All_cap_uesd_depart_once = self.All_cap_uesd_depart_once + len(self.bus_online[-1].pn_on)
                    self.bus_online[-1].left_every_station_pn_on[i]=len(self.bus_online[-1].pn_on)
                else: # 若出现滞留
                    append_mark = (self.bus_online[0].pn_on_max - len(self.bus_online[0].pn_on))
                    self.bus_online[0].pn_on.append(self.bus_online[0].To_be_process_all_station[i].iloc[:append_mark, :])
                    self.bus_online[0].Left_after_pass_the_station[i] = self.bus_online[0].To_be_process_all_station[i].iloc[append_mark:, :]
                    self.All_psger_wait_time = self.All_psger_wait_time + self.bus_online[0].Estimated_arrival_time[i] * append_mark - np.sum(self.bus_online[0].To_be_process_all_station[i].iloc[:append_mark, 4])
                    self.All_psger_wait_time_depart_once = self.All_psger_wait_time_depart_once + self.bus_online[0].Estimated_arrival_time[i] * append_mark - np.sum(self.bus_online[0].To_be_process_all_station[i].iloc[:append_mark, 4])
                    self.All_cap_uesd = self.All_cap_uesd + len(self.bus_online[-1].pn_on)
                    self.All_cap_uesd_depart_once = self.All_cap_uesd_depart_once + len(self.bus_online[-1].pn_on)
                    self.bus_online[-1].left_every_station_pn_on[i] = len(self.bus_online[-1].pn_on)
                    self.bus_online[-1].cant_taken_once = self.bus_online[-1].cant_taken_once + len(self.bus_online[0].To_be_process_all_station[i])-append_mark

        elif len(self.bus_online)  > 1:
            for i in range(self.station_number):
                self.bus_online[-1].To_be_process_all_station[i]= pd.concat([self.bus_online[-2].Left_after_pass_the_station[i],self.station.Estimated_psger_procs(i, self.bus_online[-2].Estimated_arrival_time[i], self.bus_online[-1].Estimated_arrival_time[i])])


                self.bus_online[-1].pn_on = self.bus_online[-1].pn_on.append(self.bus_online[-1].pn_on[self.bus_online[-1].pn_on["Alighting station"] == i])
                self.bus_online[-1].pn_on = self.bus_online[-1].pn_on.drop_duplicates(subset=["Label", "Boarding time", "Boarding station", "Alighting station"], keep=False)


                if len(self.bus_online[-1].To_be_process_all_station[i]) + len(self.bus_online[-1].pn_on) <= self.bus_online[-1].pn_on_max:

                    self.bus_online[-1].pn_on = pd.concat([self.bus_online[-1].pn_on, self.bus_online[-1].To_be_process_all_station[i]])
                    self.bus_online[-1].Left_after_pass_the_station[i] = pd.DataFrame(columns=["Label", "Boarding time", "Boarding station", "Alighting station", "Arrival time"])
                    self.All_psger_wait_time = self.All_psger_wait_time + self.bus_online[-1].Estimated_arrival_time[i] * len(self.bus_online[-1].To_be_process_all_station[i]) - np.sum(self.bus_online[-1].To_be_process_all_station[i].iloc[:, 4])
                    self.All_cap_uesd = self.All_cap_uesd + len(self.bus_online[-1].pn_on)
                    self.All_psger_wait_time_depart_once = self.All_psger_wait_time_depart_once + self.bus_online[-1].Estimated_arrival_time[
                        i] * len(self.bus_online[-1].To_be_process_all_station[i]) - np.sum(
                        self.bus_online[-1].To_be_process_all_station[i].iloc[:, 4])
                    self.All_cap_uesd_depart_once = self.All_cap_uesd_depart_once + len(self.bus_online[-1].pn_on)


                    self.bus_online[-1].left_every_station_pn_on[i] = len(self.bus_online[-1].pn_on)


                else:
                    append_mark = (self.bus_online[-1].pn_on_max - len(self.bus_online[-1].pn_on))
                    self.bus_online[-1].pn_on.append(self.bus_online[-1].To_be_process_all_station[i].iloc[:append_mark, :])
                    self.bus_online[-1].Left_after_pass_the_station[i] = self.bus_online[-1].To_be_process_all_station[i].iloc[append_mark:, :]
                    self.All_psger_wait_time = self.All_psger_wait_time + self.bus_online[-1].Estimated_arrival_time[i] * append_mark - np.sum(self.bus_online[-1].To_be_process_all_station[i].iloc[:append_mark, 4])
                    self.All_cap_uesd = self.All_cap_uesd + len(self.bus_online[-1].pn_on)
                    self.All_psger_wait_time_depart_once = self.All_psger_wait_time_depart_once + self.bus_online[-1].Estimated_arrival_time[
                        i] * append_mark - np.sum(
                        self.bus_online[-1].To_be_process_all_station[i].iloc[:append_mark, 4])
                    self.All_cap_uesd_depart_once = self.All_cap_uesd_depart_once + len(self.bus_online[-1].pn_on)
                    self.bus_online[-1].left_every_station_pn_on[i] = len(self.bus_online[-1].pn_on)
                    self.bus_online[-1].cant_taken_once = self.bus_online[-1].cant_taken_once + len(
                        self.bus_online[-1].To_be_process_all_station[i]) - append_mark
        self.last_departure_cap_use.put(self.All_cap_uesd - cap_use_now)
        #print("fache")

    def Arrival_test(self): #判断车辆是否到达终点站
        for i in range (len(self.bus_online)):
            # print(i,self.bus_online[i].position)
            if self.bus_online[i].position[-1]==1:
                self.bus_online[i].arrv_mark = 1
                self.bus_online[i].onlie_mark = 0
                break

    def Action(self, Departure_factor=0, min_Interval=min_Interval, max_Interval=max_Interval): #决定是否发车
        # 人进站
        global departure_label
        if Departure_factor != 1 and Departure_factor != 0:  # 确定发车因子
            af = random.random()
            if af <= 0.5:
                Departure_factor = 0
            else:
                Departure_factor = 1
        else:
            pass
        # 结束标志位
        if right_minute_th > last_minute_th + 50 :
            self.end_label = 1

        # 发车
        if (start_minute_th <= right_minute_th <= end_minute_th) and self.Interval == ideal_interval-1 and avg_flag == 1:
            self.Departure()
            self.Interval = 0
            departure_label = 1

        elif (right_minute_th == first_minute_th or right_minute_th == last_minute_th or self.Interval == max_Interval) and right_minute_th < last_minute_th+1:   #
            self.Departure()
            # self.Interval1 = self.Interval0  #上一时刻间隔
            # self.Interval0 = self.Interval  #当前时刻间隔
            self.Interval = 0
            departure_label = 1
        else:
            if self.Interval < min_Interval or Departure_factor == 0 or right_minute_th > last_minute_th:
                self.Interval = self.Interval + 1
                departure_label = 0
            else:
                self.Departure()
                # self.Interval1 = self.Interval0  #上一时刻间隔
                # self.Interval0 = self.Interval  #当前时刻间隔
                self.Interval = 0
                departure_label = 1




    def get_state(self): # 计算并返回当前系统的状态，包括乘客等待时间、乘客上车情况等。
        #print("hhhhhhh",len(self.bus_online),self.bus_online[0].position)




        Capacity_need = 0

        Passenger_num_on_board_leaving_station = pd.DataFrame(columns=["Label", "Boarding time","Boarding station","Alighting station","Arrival time"])

        Passenger_num_on_board_leaving_station_num = []

        Passenger_tobe_proc_per_station = []

        right_deapart_estimated_arrival_time=np.zeros(self.station_number)  # 预计到达某站的时间

        right_deapart_estimated_arrival_time[0] = right_minute_th



        for i in range(1,self.station_number-1):
            right_deapart_estimated_arrival_time[i] =  right_minute_th +sum(trf_con.iloc[(right_minute_th // 15), 6:6+i]) # 通过当前时间点和历史的交通状况计算得到预计到站的时间

        #print(right_minute_th, "预发车辆到站时间", right_deapart_estimated_arrival_time)
        #print(right_minute_th, "上一辆车离开的时间", self.bus_online[-1].Estimated_arrival_time)

        last_bus_left_per_station = []
        #print("在线车数",len(bus_on_line_now))


        This_deaparture_pre_wait_time = 0

        Passenger_cant_takeon_once = 0

        # 遍历所有站点，计算当前状态
        for i in range(self.station_number):
                
                Passenger_tobe_proc_per_station.append(pd.concat([self.bus_online[-1].Left_after_pass_the_station[i], self.station.Estimated_psger_procs(i,self.bus_online[-1].Estimated_arrival_time[i],right_deapart_estimated_arrival_time[i])])) # 包含了滞留乘客信息和预计到达站点的乘客的信息，即包含了这个站点中需要被处理的乘客的信息
                #print("下车前",i,len(Passenger_num_on_board_leaving_station))
                #print(Passenger_num_on_board_leaving_station)
                
                #通过往列表中加入再次需要下车的乘客，使得列表中存在两个重复的需要下车的乘客的信息，再通过去重（同时去掉重复的两个）来实现删掉需要下车的乘客
                Passenger_num_on_board_leaving_station = Passenger_num_on_board_leaving_station.append(Passenger_num_on_board_leaving_station[Passenger_num_on_board_leaving_station["Alighting station"] == i]) #计算第i站下车的乘客
                Passenger_num_on_board_leaving_station = Passenger_num_on_board_leaving_station.drop_duplicates(subset=["Label", "Boarding time", "Boarding station", "Alighting station"], keep=False) #在列表中删掉已经下车的乘客 
                
                #print("下车后",i,len(Passenger_num_on_board_leaving_station))
                #print(Passenger_num_on_board_leaving_station)
                if len(Passenger_tobe_proc_per_station[i]) + len(Passenger_num_on_board_leaving_station) <= pn_on_max: #若无被滞留的乘客
                    #print('leave', Passenger_num_on_board_leaving_station)
                    # print(right_minute_th,1,len(self.pn_on),len(station.right_minute_passenger[i]))
                    Passenger_num_on_board_leaving_station = pd.concat([Passenger_num_on_board_leaving_station, Passenger_tobe_proc_per_station[i]]) # 乘客上车
                    #print('i=',i,Passenger_num_on_board_leaving_station)
                    
                    #print('tobeproc',  Passenger_tobe_proc_per_station[i])
                    This_deaparture_pre_wait_time = This_deaparture_pre_wait_time + right_deapart_estimated_arrival_time[i] * len(Passenger_tobe_proc_per_station[i]) - np.sum(Passenger_tobe_proc_per_station[i].iloc[:, 4]) #计算等待时间


                else: #存在被滞留的乘客
                    append_mark = (pn_on_max - len(Passenger_num_on_board_leaving_station)) 
                    Passenger_num_on_board_leaving_station = pd.concat([Passenger_num_on_board_leaving_station,Passenger_tobe_proc_per_station[i].iloc[:append_mark, :]])
                    This_deaparture_pre_wait_time = This_deaparture_pre_wait_time + right_deapart_estimated_arrival_time[i] * append_mark - np.sum(Passenger_tobe_proc_per_station[i].iloc[:append_mark, 4])
                    Passenger_cant_takeon_once = Passenger_cant_takeon_once + len(Passenger_tobe_proc_per_station[i]) - append_mark

                Passenger_num_on_board_leaving_station_num.append(len(Passenger_num_on_board_leaving_station))
        state = torch.cat([torch.Tensor([(right_minute_th//60)/24]),torch.Tensor([(right_minute_th%60)/60]),torch.Tensor([(np.max(Passenger_num_on_board_leaving_station_num))/pn_on_max ]),torch.Tensor([(This_deaparture_pre_wait_time)/weight_time_wait_time]),
                 torch.Tensor([(np.sum(Passenger_num_on_board_leaving_station_num))/(alpha * pn_on_max * (station_num - 1))])])
        self.Cant_taken_once = Passenger_cant_takeon_once
        self.Cap_used = (np.sum(Passenger_num_on_board_leaving_station_num))
        self.if_depart_wait_time = This_deaparture_pre_wait_time
        #print(right_minute_th, "每站待处理人数     ", [len(i) for i in Passenger_tobe_proc_per_station])
        #print(right_minute_th, "每站离开时车上的人数", Passenger_num_on_board_leaving_station_num)

        return state



if __name__ == "__main__":
    model = DQN(5, 2) # 输入为5个状态，输出为2个状态的DQN网络
    step = 0 #总训练次数

    for episode in range(max_episode):
        XM_2_station = Station(all_passenger_info_path=passenger_info_path)
        XM_2_station.reset()
        #print(XM_2_station.next_minute_passenger)
        XM_2system = BUS_LINE_SYSTEM(station=XM_2_station)
        right_minute_th = first_minute_th
        #print(XM_2_station.next_minute_passenger)
        XM_2system.Action(1) #在始发时间进行一次发车
        state = XM_2system.get_state() #获取当前状态
        last_departure = right_minute_th
        departure_time = []
        ep_reward = 0
        Tmin = min_Interval
        # reward_list = []
        # plt.ion()
        # fig, ax = plt.subplots()
        Cap_half_hour = 0
        Cap_list = []
        real_need_half_hour = 0
        real_need_list = [ ]
        count_departures = 0
        cant_taken_once = 0
        tot_wait_time = 0
        while True:
            begin = time.time()
            #计算reward
            cant_taken_once += XM_2system.Cant_taken_once

            # if start_minute_th <= right_minute_th <= end_minute_th:
            #     beta = 0.2
            #     gama = 0.1
            #     # beta = 0.2
            #     # gama = 0
            #     print("当前时刻：", XM_2system.Interval,"理想时刻：", ideal_interval)
            #     reward1 = (XM_2system.Cap_used / (alpha * pn_on_max * (station_num - 1))) - beta * (XM_2system.Cant_taken_once * (XM_2system.Interval - ideal_interval < 0 ) + (0 * XM_2system.Cant_taken_once * (XM_2system.Interval - ideal_interval >= 0))) + gama * avg_flag * (4*(XM_2system.Interval - ideal_interval >= 0) - 0*(XM_2system.Interval - ideal_interval < 0))
            #     reward0 = reward = 1 - (XM_2system.Cap_used / (alpha * pn_on_max * (station_num - 1))) - omega * XM_2system.if_depart_wait_time - beta * (XM_2system.Cant_taken_once * (XM_2system.Interval - ideal_interval < 0) + (0 * XM_2system.Cant_taken_once * (XM_2system.Interval - ideal_interval >= 0))) + gama * avg_flag * (-(3*(XM_2system.Interval - ideal_interval >= 0)) + 0*(XM_2system.Interval - ideal_interval < 0))  #abs(XM_2system.Interval0 - ideal_interval)
            #     print("reward1_alpha：", XM_2system.Cap_used / (alpha * pn_on_max * (station_num - 1)), "reward1_beta：", -beta * (XM_2system.Cant_taken_once * (XM_2system.Interval - ideal_interval < 0) + (0 * XM_2system.Cant_taken_once * (XM_2system.Interval - ideal_interval >= 0))),"reward1_gama：",gama * avg_flag * (4*(XM_2system.Interval - ideal_interval >= 0) - 0*(XM_2system.Interval - ideal_interval < 0)))
            #     print("reward0_alpha：", 1 - (XM_2system.Cap_used / (alpha * pn_on_max * (station_num - 1))), "reward0_omega：",-omega * XM_2system.if_depart_wait_time, "reward0_beta：",- beta * (XM_2system.Cant_taken_once * (XM_2system.Interval - ideal_interval < 0) + (0 * XM_2system.Cant_taken_once * (XM_2system.Interval - ideal_interval >= 0))),"reward0_gama：", gama * avg_flag * (-(3*(XM_2system.Interval - ideal_interval >= 0)) + 0*(XM_2system.Interval - ideal_interval < 0)) )
            # else:
            #     beta = 0.2
            reward1 = (XM_2system.Cap_used / (alpha * pn_on_max * (station_num - 1))) - beta * XM_2system.Cant_taken_once #原
            reward0 = reward = 1 - (XM_2system.Cap_used / (alpha * pn_on_max * (station_num - 1))) - omega * XM_2system.if_depart_wait_time - beta * XM_2system.Cant_taken_once #原

            action = model.choose_action(state, Tmin, max_Interval, epsilon, XM_2system.Interval)
            if right_minute_th != first_minute_th: #除了发车时间之外，通过DQN网络选择是否发车
                XM_2system.Action(action) # 若发车，往在线车辆队列中加入一辆车
            # 人上下车，车辆前进

            action = departure_label #获取最终是否发车
            if (action == 0): # 根据状态选择最终reward
                reward = reward0
            else:
                count_departures += 1
                departure_time.append([right_minute_th // 60, right_minute_th % 60, right_minute_th - last_departure])
                last_departure = right_minute_th
                reward = reward1
                # Tmin = min_Interval
            for bus in XM_2system.bus_online: #更新每一个站点的人流数据，进行上下车
                bus.up_down(XM_2_station)
            on_bus = 0
            for bus in XM_2system.bus_online: #所有巴士前进一分钟
                on_bus += len(bus.pn_on)
                arrival_flag = bus.forward_one_step()
                if arrival_flag == 1:
                    #到达终点站后减去这辆巴士提供的载客容量
                    XM_2system.All_cap_take -= pn_on_max * (station_num - 1)
                    last_cap_use = XM_2system.last_departure_cap_use.get()
                    XM_2system.All_cap_uesd -= last_cap_use
            XM_2system.station.forward_one_step() #获取下一个状态的各个车站人流信息
            
            right_minute_th += 1 #时间加一
            next_state = XM_2system.get_state() #获取下一个状态
            
            ep_reward += reward
            bus_on_line_now=[] #计算当前在线车辆数量
            for j in range(len(XM_2system.bus_online)):
                if XM_2system.bus_online[j].arrv_mark!= 1:
                    bus_on_line_now.append(len(XM_2system.bus_online[j].pn_on))
            Cap_half_hour += XM_2system.All_cap_take
            if right_minute_th <= last_minute_th and right_minute_th % 30 == 0: #统计提供的载客量
                Cap_half_hour = Cap_half_hour / 30
                Cap_list.append(Cap_half_hour)
                Cap_half_hour = 0
            tot_wait_time += XM_2system.if_depart_wait_time
            #real_need_half_hour += XM_2system.All_cap_uesd
            real_need_half_hour += on_bus
            if right_minute_th <= last_minute_th and right_minute_th % 30 == 0: #统计消耗的载客量
                real_need_half_hour = real_need_half_hour / 30
                real_need_list.append(real_need_half_hour)
                real_need_half_hour = 0
            #print(right_minute_th,"运力及等车时间",XM_2system.All_cap_take,XM_2system.All_cap_uesd,XM_2system.All_psger_wait_time)
            # reward_list.append(XM_2system.All_cap_take)
            # print(right_minute_th,"运力及等车时间",XM_2system.All_cap_take,XM_2system.All_cap_uesd,XM_2system.All_psger_wait_time)
            #print("在线车数：",len(bus_on_line_now)) 
            model.store_transition(state, action, reward, next_state) #每次存储当前的经验
            if step > 3000 and step % train_counter == 0: #当经验数量存满经验回放的容量时开始训练
                model.train_network(model_save_path)
            step += 1
            if (step % 10 == 0):
                print("episode: ", episode, " step: ", step,'rightminute: ',right_minute_th, ' reward: ', reward, ' action: ', action, 'bus_on_line_now: ', len(bus_on_line_now), 'reward0: ', reward0, ' reward1: ', reward1)
            state = next_state
            #print("step", step)
            # print(right_minute_th,"运力及等车时间",XM_2system.Cap_used,XM_2system.Cant_taken_once,XM_2system.if_depart_wait_time)
            if XM_2system.end_label == 1: #结束一天的标志
                print("episode: {} , the episode reward is {}".format(episode, round(ep_reward, 3)))
                print("departure time", departure_time) #发车时间
                print('Cap', Cap_list) #消耗容量
                print('departure times', count_departures) #发车次数
                print('AWT', tot_wait_time / (last_minute_th - first_minute_th) / station_num) #平均等待时间
                print('real_Cap_need', real_need_list)
                print('cant take once', cant_taken_once)
                break
            end = time.time()
            #print(begin - end)