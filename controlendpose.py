#!/usr/bin/env python3
# -*-coding:utf8-*-
# 注意demo无法直接运行，需要pip安装sdk后才能运行
import time
from piper_sdk import *

if __name__ == "__main__":
    piper = C_PiperInterface_V2("can_piper")

    #连接后等待1秒获取当前状态
    piper.ConnectPort()
    time.sleep(1)

    print(f"当前机械臂状态为{piper.GetArmStatus().arm_status.ctrl_mode}")

    if piper.GetArmStatus().arm_status.ctrl_mode == 0x00:#如果是待机
        print("尝试从 待机模式->can控制模式")
        piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)#设置can模式
        print(piper.GetArmStatus().arm_status.ctrl_mode)
        time.sleep(1)
        print(piper.GetArmStatus().arm_status.ctrl_mode)
    elif piper.GetArmStatus().arm_status.ctrl_mode == 0x02:#如果是示教模式
        print("尝试从 示教模式->can控制模式")
        piper.MotionCtrl_1(0x02,0,0)#恢复，示教模式->待机模式(会恢复到别的模式吗？没有我可就写死了)
        # print(piper.GetArmStatus().arm_status.ctrl_mode)
        time.sleep(1)#这里必须要等
        # print(piper.GetArmStatus().arm_status.ctrl_mode)

        piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)#设置can模式
        time.sleep(1)#这里也必须要等

        while( not piper.EnablePiper()):#使能机械臂
            time.sleep(0.01)

        piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)#设置can模式
        while(piper.GetArmStatus().arm_status.ctrl_mode != 0x01):#等待进入can控制模式
            time.sleep(0.01)
        print("成功切换到can控制模式")


    elif piper.GetArmStatus().arm_status.ctrl_mode == 0x01:#如果是can控制模式
        print("can控制模式") 
        while( not piper.EnablePiper()):#使能机械臂
            time.sleep(0.01)
        
    else:
        print(piper.GetArmStatus().arm_status.ctrl_mode)
    #清除夹爪错误使其可以被正常控制
    piper.GripperCtrl(0,1000,0x02, 0)
    piper.GripperCtrl(0,1000,0x01, 0)

    print(piper.GetArmStatus().arm_status.ctrl_mode)
    piper.GripperCtrl(0,1000,0x01, 0)
    factor = 1000
    position = [
                57.0, \
                0.0, \
                215.0, \
                0, \
                85.0, \
                0, \
                0]

    count = 0
    # while True:
    #     #print(piper.GetArmEndPoseMsgs())
    #     count  = count + 1
    #     if(count == 0):
    #         print("1-----------")
    #         position = [153.804, 7.624, 510.978, -179.0, 55.365, -175.947, 0]
    #     elif(count == 400):
    #         print("2-----------")
    #         position = [153.804, 7.624, 510.978, -124.646, 1.757, -88.375, 50]
    #     elif(count == 800):
    #         print("3-----------")
    #         position = [152.647, 10.661, 508.677, 126.092, -1.233, 94.347, 0]
    #     elif(count == 1200):
    #         print("4-----------")
    #         position = [132.876, -207.387, 352.569, -147.268, 69.036, -148.716, 50]
    #     elif(count == 1600):
    #         print("5-----------")
    #         position = [217.339, 305.423, 327.091, 166.451, 60.481, 158.628, 0]
    #         count = 0
        
    #     X = round(position[0]*factor)
    #     Y = round(position[1]*factor)
    #     Z = round(position[2]*factor)
    #     RX = round(position[3]*factor)
    #     RY = round(position[4]*factor)
    #     RZ = round(position[5]*factor)
    #     joint_6 = round(position[6]*factor)
    #     #print(X,Y,Z,RX,RY,RZ)
    #     piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
    #     piper.EndPoseCtrl(X,Y,Z,RX,RY,RZ)
    #     piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
    #     #print(piper.GetArmStatus().arm_status.ctrl_mode)
    #     time.sleep(0.01)