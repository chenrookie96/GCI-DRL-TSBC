"""
系统测试脚本
用于快速验证DRL-TSBC系统的各个组件是否正常工作
"""

import sys
import os

def test_imports():
    """测试所有模块是否能正常导入"""
    print("="*60)
    print("Testing imports...")
    print("="*60)
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✓ Pandas version: {pd.__version__}")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        from drl_tsbc_brain import DQN, Net
        print("✓ drl_tsbc_brain module imported")
    except ImportError as e:
        print(f"✗ drl_tsbc_brain import failed: {e}")
        return False
    
    try:
        from drl_tsbc_environment import (
            Station, Bus, DirectionSystem, BidirectionalBusSystem
        )
        print("✓ drl_tsbc_environment module imported")
    except ImportError as e:
        print(f"✗ drl_tsbc_environment import failed: {e}")
        return False
    
    try:
        from data_loader import BusDataLoader, check_data_files
        print("✓ data_loader module imported")
    except ImportError as e:
        print(f"✗ data_loader import failed: {e}")
        return False
    
    try:
        import config_drl_tsbc
        print("✓ config_drl_tsbc module imported")
    except ImportError as e:
        print(f"✗ config_drl_tsbc import failed: {e}")
        return False
    
    print("\nAll imports successful!")
    return True


def test_data_files():
    """测试数据文件是否存在"""
    print("\n" + "="*60)
    print("Testing data files...")
    print("="*60)
    
    from data_loader import check_data_files
    
    # 检查208线数据
    exists_208 = check_data_files(208, "./test_data")
    
    # 检查211线数据（如果有）
    print("\n")
    exists_211 = check_data_files(211, "./test_data")
    
    return exists_208


def test_network():
    """测试DQN网络是否能正常创建"""
    print("\n" + "="*60)
    print("Testing DQN network...")
    print("="*60)
    
    try:
        from drl_tsbc_brain import DQN
        import torch
        
        # 创建网络（10维状态，4个动作）
        model = DQN(n_states=10, n_actions=4)
        print("✓ DQN network created successfully")
        
        # 测试前向传播
        test_state = torch.randn(1, 10).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        with torch.no_grad():
            output = model.eval_net(test_state)
        print(f"✓ Forward pass successful, output shape: {output.shape}")
        
        # 测试动作选择
        test_state_np = test_state.cpu().numpy()[0]
        action_idx, action_tuple = model.choose_action(
            test_state_np, 5, 22, 0.1, 10, 10
        )
        print(f"✓ Action selection successful: action_idx={action_idx}, action_tuple={action_tuple}")
        
        return True
    except Exception as e:
        print(f"✗ Network test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading():
    """测试数据加载"""
    print("\n" + "="*60)
    print("Testing data loading...")
    print("="*60)
    
    try:
        from data_loader import BusDataLoader
        
        loader = BusDataLoader(data_dir="./test_data")
        upward_passenger, downward_passenger, config = loader.load_all_data(208)
        
        print(f"✓ Data loaded successfully")
        print(f"  Upward passengers: {len(upward_passenger)}")
        print(f"  Downward passengers: {len(downward_passenger)}")
        print(f"  Upward stations: {config['upward']['station_num']}")
        print(f"  Downward stations: {config['downward']['station_num']}")
        
        return True
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment():
    """测试仿真环境"""
    print("\n" + "="*60)
    print("Testing simulation environment...")
    print("="*60)
    
    try:
        from drl_tsbc_environment import Station, BidirectionalBusSystem
        from data_loader import BusDataLoader
        import pandas as pd
        
        # 加载数据
        loader = BusDataLoader(data_dir="./test_data")
        upward_passenger, downward_passenger, config = loader.load_all_data(208)
        
        # 创建车站
        first_minute_th = 360  # 6:00
        upward_station = Station(
            config['upward']['station_num'],
            "./test_data/208/passenger_dataframe_direction0.csv",
            first_minute_th
        )
        downward_station = Station(
            config['downward']['station_num'],
            "./test_data/208/passenger_dataframe_direction1.csv",
            first_minute_th
        )
        print("✓ Stations created")
        
        # 创建双向系统
        bus_system = BidirectionalBusSystem(
            upward_station,
            downward_station,
            47,  # pn_on_max
            config['upward']['traffic'],
            config['downward']['traffic']
        )
        print("✓ Bidirectional bus system created")
        
        # 测试获取状态
        state = bus_system.get_full_state(first_minute_th)
        print(f"✓ State obtained, shape: {state.shape}")
        print(f"  State values: {state}")
        
        # 测试执行动作
        bus_system.Action((1, 1), first_minute_th, 5, 22)
        print("✓ Action executed")
        
        # 测试计算奖励
        reward = bus_system.calculate_reward((1, 1), 0.001)
        print(f"✓ Reward calculated: {reward:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("DRL-TSBC System Test")
    print("="*60)
    
    results = {}
    
    # 测试导入
    results['imports'] = test_imports()
    
    # 测试数据文件
    results['data_files'] = test_data_files()
    
    # 测试网络
    results['network'] = test_network()
    
    # 测试数据加载
    if results['data_files']:
        results['data_loading'] = test_data_loading()
    else:
        print("\n⚠ Skipping data loading test (data files not found)")
        results['data_loading'] = False
    
    # 测试环境
    if results['data_files'] and results['data_loading']:
        results['environment'] = test_environment()
    else:
        print("\n⚠ Skipping environment test (prerequisites not met)")
        results['environment'] = False
    
    # 总结
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("All tests passed! System is ready.")
        print("\nYou can now run:")
        print("  python train_drl_tsbc.py    # To train the model")
        print("  python inference_drl_tsbc.py # To generate timetables")
    else:
        print("Some tests failed. Please check the errors above.")
        if not results['data_files']:
            print("\nMissing data files. Please ensure data files exist in:")
            print("  ./test_data/208/passenger_dataframe_direction0.csv")
            print("  ./test_data/208/passenger_dataframe_direction1.csv")
            print("  ./test_data/208/traffic-0.csv")
            print("  ./test_data/208/traffic-1.csv")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
