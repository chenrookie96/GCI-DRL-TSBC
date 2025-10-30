#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试导入"""

print("开始导入...")
try:
    import drl_tsbc_environment
    print(f"模块导入成功: {drl_tsbc_environment}")
    print(f"模块属性: {dir(drl_tsbc_environment)}")
    
    # 尝试访问类
    if hasattr(drl_tsbc_environment, 'Station'):
        print("Station 类存在")
    else:
        print("Station 类不存在")
        
    # 检查模块文件
    print(f"模块文件: {drl_tsbc_environment.__file__}")
    
    # 尝试直接执行文件内容
    print("\n尝试直接执行文件...")
    with open('drl_tsbc_environment.py', 'r', encoding='utf-8') as f:
        code = f.read()
        exec(code, globals())
    
    print(f"执行后 Station 是否存在: {'Station' in globals()}")
    
except Exception as e:
    import traceback
    print(f"错误: {e}")
    traceback.print_exc()
