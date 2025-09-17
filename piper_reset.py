#!/usr/bin/env python3
# -*-coding:utf8-*-
# 注意demo无法直接运行，需要pip安装sdk后才能运行
# 设置机械臂重置，需要在mit或者示教模式切换为位置速度控制模式时执行
from piper_sdk import *
import time
# 测试代码
if __name__ == "__main__":
    piper = C_PiperInterface_V2("can_piper")

    print(piper.GetArmStatus().arm_status.ctrl_mode)
    time.sleep(1)
    print(piper.GetArmStatus().arm_status.ctrl_mode)

    piper.ConnectPort()
    while( not piper.EnablePiper()):
        time.sleep(0.01)

    print(piper.GetArmStatus().arm_status.ctrl_mode)
    time.sleep(1)
    print(piper.GetArmStatus().arm_status.ctrl_mode)

    piper.ConnectPort()

    print(piper.GetArmStatus().arm_status.ctrl_mode)
    time.sleep(1)
    print(piper.GetArmStatus().arm_status.ctrl_mode)

    piper.MotionCtrl_1(0x02,0,0)#恢复

    print(piper.GetArmStatus().arm_status.ctrl_mode)
    time.sleep(1)
    print(piper.GetArmStatus().arm_status.ctrl_mode)

    piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)#cancontrol位置速度模式，can模式

    print(piper.GetArmStatus().arm_status.ctrl_mode)
    time.sleep(1)
    print(piper.GetArmStatus().arm_status.ctrl_mode)

    piper.GripperCtrl(0,1000,0x02, 0)#禁用并清除错误

    print(piper.GetArmStatus().arm_status.ctrl_mode)
    time.sleep(1)
    print(piper.GetArmStatus().arm_status.ctrl_mode)

    piper.GripperCtrl(0,1000,0x01, 0)#启用
    #闭合夹爪以夹持标定版


    print(piper.GetArmStatus().arm_status.ctrl_mode)
    time.sleep(1)
    print(piper.GetArmStatus().arm_status.ctrl_mode)

    piper.GripperCtrl(50000, 3000, 0x01, 0)
