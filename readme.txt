训练时运行Environment.py
数据格式可见test_data，需保证格式一致
passenger_dataframe_direction0包含
Label,Boarding time,Boarding station,Alighting station,Arrival time
表示唯一乘客标识，上车时间，上车站点，下车站点，到达上车站点的预估时间
traffic-0包含
time_h1,time_h2,time_m1,time_m2,start_m,finish_m,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,s25,
表示小时数1，小时数2，分钟数1，分钟数2，开始时间1（小时数1*60+分钟数1），开始时间2（小时数1*60+分钟数1），后面表示当前站点到下一个站点所需时间(分钟)，最后一列可用0补齐，表示末站，实际上不影响模型性能
训练时运行Environment.py，在文件开头处的参数可进行配置，包含：
train_counter = 5 #记忆满后每五步训练一次，若模型不收敛，可增大此值，若收敛过慢，可减小此值

#初始发车时间，最后发车时间
first_time="06:00"
last_time="21:00"

station_num  = 26  #线路的站数

# 定义线路号，方向
busline = 208
direction = 0
#定义omega_factor超参数，用于控制总发车次数，omega_factor越大发车次数越少
omega_factor = 1000

pn_on_max = 47 #每辆车的最大运载人数

alpha = 1 #这个参数表示最大能超载多少的系数，如若为1.5则实际可搭载1.5倍人数，若在最大运载人数已经体现，可设置为1


weight_time_wait_time = 5000 #这个是一个用于归一化等待时间的参数，不需设置

min_Interval = 5 #最小发车间隔

max_Interval =22 #最大发车间隔

-------------------
模型训练时在最初3000步，会累积记忆，此时不会进行训练，只有在每一天结束后会出现一个输出，此时发车次数会很大，输出的内容可见代码注释，当3000步后模型会进行训练，每100步输出一个loss，每一天输出一个总reward，当loss较低（0.001数量级）且总reward维持在一个较高的水平时，可判定训练结束

若需测试模型，请运行work_env.py，Environment.py中有关模型性能的输出是不准确的，通过调整文件开头的有关参数，可选择不同的模型


----------------
另外在data_process文件夹下，traffic_csv_trans.py为将提供的原始数据中的列表现形式换位行形式，并且补充前后的数据，首班车之前的时间可用0补齐，末班车之后的数据可用末班车时刻点数据补齐
cal_pass_arr.py为计算乘客预期到站时间的文件
通过运行以上两个文件可得到模型训练所需的数据