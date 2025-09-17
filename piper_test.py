#!/usr/bin/env python3
# -*-coding:utf8-*-
# 注意demo无法直接运行，需要pip安装sdk后才能运行
# 读取机械臂消息并打印,需要先安装piper_sdk
import time
from piper_sdk import *

# 测试代码
if __name__ == "__main__":
    piper = C_PiperInterface_V2("can_piper")
    piper.ConnectPort()
    piper.GripperCtrl(0,1000,0x02, 0)#禁用并清除错误
    piper.GripperCtrl(0,1000,0x01, 0)#启用
    #闭合夹爪以夹持标定版
    piper.GripperCtrl(0, 1000, 0x01, 0)#夹径0,1N，启用
    
    try:
        # 4. 进入拖动示教模式
        piper.MotionCtrl_1(grag_teach_ctrl=0x01)
        print("进入拖动示教模式，按Ctrl+C退出...")
        
        while True:
            endpose = piper.GetArmEndPoseMsgs().end_pose
            pose_list = [endpose.X_axis, endpose.Y_axis, endpose.Z_axis, 
                         endpose.RX_axis, endpose.RY_axis, endpose.RZ_axis]
            pose_list = [number / 1000 for number in pose_list]
            print(pose_list)
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n收到退出信号，正在退出拖动示教模式...")
        
    finally:
        #piper.MotionCtrl_1(grag_teach_ctrl=0x02)#退出拖动示教模式

        print(piper.GetArmStatus().arm_status.ctrl_mode)

        if piper.GetArmStatus().arm_status.ctrl_mode == 0x02:#如果处于can控制模式
            piper.MotionCtrl_1(0x02,0,0)#复位
            piper.MotionCtrl_1(0x02,0,0)


        print(piper.GetArmStatus().arm_status.ctrl_mode)
        time.sleep(1)
        print(piper.GetArmStatus().arm_status.ctrl_mode)

        time.sleep(0.1)
        piper.ConnectPort()
        while( not piper.EnablePiper()):
            time.sleep(0.01)

        print(piper.GetArmStatus().arm_status.ctrl_mode)
        time.sleep(1)
        print(piper.GetArmStatus().arm_status.ctrl_mode)

        piper.GripperCtrl(0,1000,0x02, 0)#禁用并清除错误
        piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)#位置速度模式
        # range = 50 * 1000 # 50mm
        # range = round(range)
        # piper.GripperCtrl(abs(range), 1000, 0x01, 0)
        print("已退出拖动示教模式")
        print(piper.GetArmStatus().arm_status.ctrl_mode)
        time.sleep(1)
        print(piper.GetArmStatus().arm_status.ctrl_mode)