"""
测试双向仿真环境
"""

import sys
sys.path.append('.')

from drl_tsbc.utils.data_loader import BusDataLoader
from drl_tsbc.utils.config_manager import LineConfigManager
from drl_tsbc.core.direction_env import DirectionEnvironment
from drl_tsbc.core.environment import BidirectionalBusEnvironment


def test_environment():
    """测试环境基本功能"""
    print("="*50)
    print("测试双向仿真环境")
    print("="*50)
    
    # 加载数据
    loader = BusDataLoader()
    line = '208'
    
    print(f"\n加载{line}线数据...")
    upward_passengers, downward_passengers, line_config_dict = loader.load_all_data(line)
    
    # 获取线路配置
    line_config = LineConfigManager(line)
    
    # 创建上行环境
    print("\n创建上行环境...")
    upward_env = DirectionEnvironment(
        direction='upward',
        config=line_config.get_upward_config(),
        passenger_data=upward_passengers
    )
    
    # 创建下行环境
    print("创建下行环境...")
    downward_env = DirectionEnvironment(
        direction='downward',
        config=line_config.get_downward_config(),
        passenger_data=downward_passengers
    )
    
    # 创建双向环境
    print("创建双向环境...")
    env = BidirectionalBusEnvironment(upward_env, downward_env)
    
    # 重置环境
    print("\n重置环境...")
    state = env.reset()
    print(f"初始状态: {state}")
    print(f"状态维度: {len(state)}")
    
    # 测试几步
    print("\n测试环境步进...")
    for i in range(5):
        # 随机动作
        import random
        action = (random.randint(0, 1), random.randint(0, 1))
        
        next_state, reward, done, info = env.step(action)
        
        print(f"\n步骤 {i+1}:")
        print(f"  动作: {action}")
        print(f"  奖励: {reward:.4f}")
        print(f"  完成: {done}")
        print(f"  上行发车次数: {info['upward_departures']}")
        print(f"  下行发车次数: {info['downward_departures']}")
        
        if done:
            break
    
    print("\n" + "="*50)
    print("环境测试完成！")
    print("="*50)


if __name__ == '__main__':
    test_environment()
