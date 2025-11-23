"""
数据加载模块
改动说明：
1. 统一管理所有数据加载逻辑
2. 参考cal_pass_arr.py的乘客到站时间估算方法
3. 添加数据验证和错误处理
4. 支持208和211两条线路
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

np.random.seed(10086)


class BusDataLoader:
    """
    公交数据加载器
    功能：
    1. 加载乘客数据
    2. 加载交通状况数据
    3. 估算乘客到站时间（如果数据中没有）
    4. 数据验证
    """
    
    def __init__(self, data_dir="./test_data"):
        self.data_dir = data_dir
    
    def load_passenger_data(self, busline, direction):
        """
        加载乘客数据
        
        参数：
        - busline: 线路号（208或211）
        - direction: 方向（0=上行，1=下行）
        
        返回：
        - DataFrame: 乘客数据，包含Arrival time列
        """
        file_path = f"{self.data_dir}/{busline}/passenger_dataframe_direction{direction}.csv"
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"乘客数据文件不存在: {file_path}")
        
        print(f"Loading passenger data: {file_path}")
        df = pd.read_csv(file_path)
        
        # 检查必需的列
        required_cols = ["Label", "Boarding time", "Boarding station", "Alighting station"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"乘客数据缺少必需列: {col}")
        
        # 如果没有Arrival time列，需要估算
        if "Arrival time" not in df.columns:
            print(f"  Warning: 'Arrival time' column not found, will use 'Boarding time' as estimate")
            # 简单估算：假设乘客在上车前1-5分钟到达
            df["Arrival time"] = df["Boarding time"] - np.random.randint(1, 6, size=len(df))
            df["Arrival time"] = df["Arrival time"].clip(lower=0)
        
        print(f"  Loaded {len(df)} passenger records")
        return df
    
    def load_traffic_data(self, busline, direction):
        """
        加载交通状况数据
        
        参数：
        - busline: 线路号（208或211）
        - direction: 方向（0=上行，1=下行）
        
        返回：
        - DataFrame: 交通状况数据
        """
        file_path = f"{self.data_dir}/{busline}/traffic-{direction}.csv"
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"交通数据文件不存在: {file_path}")
        
        print(f"Loading traffic data: {file_path}")
        df = pd.read_csv(file_path)
        
        print(f"  Loaded traffic data with {len(df)} time periods")
        return df
    
    def estimate_arrival_time(self, passenger_df, arrival_depart_df, date='2023-08-14'):
        """
        估算乘客到站时间（参考cal_pass_arr.py的方法）
        
        参数：
        - passenger_df: 乘客数据（包含Boarding time, Boarding station等）
        - arrival_depart_df: 公交到站/离站时间数据
        - date: 日期
        
        返回：
        - DataFrame: 添加了Arrival time列的乘客数据
        """
        print("Estimating passenger arrival times...")
        
        # 如果arrival_depart_df为None，使用简单估算
        if arrival_depart_df is None:
            passenger_df["Arrival time"] = passenger_df["Boarding time"] - np.random.randint(1, 6, size=len(passenger_df))
            passenger_df["Arrival time"] = passenger_df["Arrival time"].clip(lower=0)
            return passenger_df
        
        # 使用正态分布估算（参考cal_pass_arr.py）
        for index, row in passenger_df.iterrows():
            boarding_station = row["Boarding station"]
            boarding_time = row["Boarding time"]
            
            # 找到该站点上一辆车的离站时间
            station_buses = arrival_depart_df[
                (arrival_depart_df["s"] - 1) == boarding_station
            ]
            
            # 找到小于boarding_time的最大arrival_time
            prev_buses = station_buses[station_buses["arrival_time"] < boarding_time]
            
            if len(prev_buses) > 0:
                last_depart_time = prev_buses["arrival_time"].max()
                # 使用正态分布估算
                std = abs(boarding_time - last_depart_time) / 2
                estimate = np.random.normal(boarding_time, std)
                estimate = abs(estimate - boarding_time)
                arrival_time = int(boarding_time - estimate)
            else:
                # 如果没有前车数据，简单估算
                arrival_time = boarding_time - np.random.randint(1, 6)
            
            passenger_df.loc[index, "Arrival time"] = max(0, arrival_time)
        
        print(f"  Estimated arrival times for {len(passenger_df)} passengers")
        return passenger_df
    
    def validate_data(self, passenger_df, traffic_df, station_num):
        """
        验证数据的有效性
        
        参数：
        - passenger_df: 乘客数据
        - traffic_df: 交通数据
        - station_num: 站点数量
        
        返回：
        - bool: 数据是否有效
        """
        print("Validating data...")
        
        # 检查乘客数据
        if passenger_df["Boarding station"].max() >= station_num:
            print(f"  Warning: Boarding station exceeds station_num ({station_num})")
        
        if passenger_df["Alighting station"].max() >= station_num:
            print(f"  Warning: Alighting station exceeds station_num ({station_num})")
        
        if passenger_df["Boarding time"].min() < 0:
            print(f"  Warning: Negative boarding time found")
        
        # 检查交通数据
        if len(traffic_df) == 0:
            print(f"  Error: Traffic data is empty")
            return False
        
        print("  Data validation passed")
        return True
    
    def load_all_data(self, busline):
        """
        加载指定线路的所有数据
        
        参数：
        - busline: 线路号（208或211）
        
        返回：
        - tuple: (upward_passenger_df, downward_passenger_df, config_dict)
        """
        print(f"\n{'='*60}")
        print(f"Loading data for Line {busline}")
        print(f"{'='*60}")
        
        # 加载上行数据
        print("\n[Upward Direction]")
        upward_passenger = self.load_passenger_data(busline, 0)
        upward_traffic = self.load_traffic_data(busline, 0)
        
        # 加载下行数据
        print("\n[Downward Direction]")
        downward_passenger = self.load_passenger_data(busline, 1)
        downward_traffic = self.load_traffic_data(busline, 1)
        
        # 配置信息
        config = {
            'busline': busline,
            'upward': {
                'station_num': self._get_station_num(upward_traffic),
                'traffic': upward_traffic,
                'passenger_count': len(upward_passenger)
            },
            'downward': {
                'station_num': self._get_station_num(downward_traffic),
                'traffic': downward_traffic,
                'passenger_count': len(downward_passenger)
            }
        }
        
        print(f"\n{'='*60}")
        print(f"Data loading completed")
        print(f"  Upward: {config['upward']['station_num']} stations, "
              f"{config['upward']['passenger_count']} passengers")
        print(f"  Downward: {config['downward']['station_num']} stations, "
              f"{config['downward']['passenger_count']} passengers")
        print(f"{'='*60}\n")
        
        return upward_passenger, downward_passenger, config
    
    def _get_station_num(self, traffic_df):
        """
        从交通数据中推断站点数量
        
        参数：
        - traffic_df: 交通数据
        
        返回：
        - int: 站点数量
        """
        # 交通数据格式：
        # 前6列：time_h1, time_h2, time_m1, time_m2, start_m, finish_m
        # 后面：s0, s1, s2, ..., sN（站间行驶时间）
        # 最后可能有Unnamed列（忽略）
        
        # 计算s+数字开头的列数（排除start_m, finish_m等）
        import re
        s_columns = [col for col in traffic_df.columns if re.match(r'^s\d+$', col)]
        
        # 站点数 = s列的数量
        # 例如：s0到s25共26列 → 26个站点
        station_num = len(s_columns)
        
        return station_num


def check_data_files(busline, data_dir="./test_data"):
    """
    检查数据文件是否存在
    
    参数：
    - busline: 线路号
    - data_dir: 数据目录
    
    返回：
    - dict: 文件存在情况
    """
    files = {
        'passenger_up': f"{data_dir}/{busline}/passenger_dataframe_direction0.csv",
        'passenger_down': f"{data_dir}/{busline}/passenger_dataframe_direction1.csv",
        'traffic_up': f"{data_dir}/{busline}/traffic-0.csv",
        'traffic_down': f"{data_dir}/{busline}/traffic-1.csv"
    }
    
    print(f"\nChecking data files for Line {busline}:")
    print(f"{'='*60}")
    
    all_exist = True
    for name, path in files.items():
        exists = os.path.exists(path)
        status = "[OK]" if exists else "[MISSING]"
        print(f"  {status} {name}: {path}")
        if not exists:
            all_exist = False
    
    print(f"{'='*60}")
    
    if all_exist:
        print("All data files found!")
    else:
        print("Some data files are missing!")
    
    return all_exist


if __name__ == "__main__":
    # 测试数据加载
    print("Testing BusDataLoader...")
    
    # 检查208线数据文件
    check_data_files(208)
    
    # 尝试加载数据
    try:
        loader = BusDataLoader()
        upward_passenger, downward_passenger, config = loader.load_all_data(208)
        
        print("\nSample passenger data (upward):")
        print(upward_passenger.head())
        
        print("\nSample traffic data (upward):")
        print(config['upward']['traffic'].head())
        
    except Exception as e:
        print(f"\nError loading data: {e}")
        print("\nPlease ensure data files exist in ./test_data/208/ directory:")
        print("  - passenger_dataframe_direction0.csv")
        print("  - passenger_dataframe_direction1.csv")
        print("  - traffic-0.csv")
        print("  - traffic-1.csv")
