#!/usr/bin/env python3
# -*-coding:utf8-*-
# 注意demo无法直接运行，需要pip安装sdk后才能运行
# 使能机械臂
import time
from piper_sdk import *

# 测试代码
if __name__ == "__main__":
    piper = C_PiperInterface_V2("can_piper")
    piper.ConnectPort()
    time.sleep(0.1)
    while( not piper.EnablePiper()):
        print(piper.EnablePiper())
        time.sleep(0.01)
    print("使能成功!!!!")
    
    