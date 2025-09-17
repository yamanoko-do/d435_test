
import time
from piper_sdk import *
from typing import List,Tuple


class PiperClass():
    def __init__(self,can_name = "can_piper"):
        self.piper = C_PiperInterface_V2(can_name)
        self.piper.ConnectPort()
        while( not self.piper.EnablePiper()):#使能机械臂
            time.sleep(0.01)
        print("init: 成功连接到piper")
        print(f"init: 当前机械臂状态：{self.piper.GetArmStatus().arm_status.ctrl_mode}")

    def control_gripper(self, isclose = True):
        """
        开关夹爪
        """
        if isclose:
            self.piper.GripperCtrl(0,1000,0x02, 0)#禁用并清除错误
            self.piper.GripperCtrl(0,1000,0x01, 0)#启用
            #闭合夹爪以夹持标定版
            self.piper.GripperCtrl(0, 1000, 0x01, 0)#夹径0,1N，启用
        else:
            self.piper.GripperCtrl(0,1000,0x02, 0)#禁用并清除错误
            self.piper.GripperCtrl(0,1000,0x01, 0)#启用
            #闭合夹爪以夹持标定版
            range = 50 * 1000 # 50mm
            range = round(range)
            self.piper.GripperCtrl(abs(range), 1000, 0x01, 0)

    def getpose(self) -> dict:
        """
        获取机械臂末端位姿和关节角度,单位: mm和度
        """
        endpose = self.piper.GetArmEndPoseMsgs().end_pose
        pose_list = [endpose.X_axis, endpose.Y_axis, endpose.Z_axis, endpose.RX_axis, endpose.RY_axis, endpose.RZ_axis]
        pose_list = [number / 1000 for number in pose_list]

        joint_state = self.piper.GetArmJointMsgs().joint_state
        joint_state_list = [joint_state.joint_1, joint_state.joint_2, joint_state.joint_3, joint_state.joint_4, joint_state.joint_5, joint_state.joint_6]
        joint_state_list = [number / 1000 for number in joint_state_list]

        pose_dict = {
            "end_pose": pose_list,
            "joint_state": joint_state_list
        }
        return pose_dict

    def disconnect(self):
        """
        断开机械臂连接
        """
        return self.piper.DisconnectPort()
    
    def get_ctrl_mode(self):   
        """
        获取机械臂状态
        """
        return self.piper.GetArmStatus().ctrl_mode
    
    def set_ctrl_mode2can(self):
        """
        切换机械臂到can控制模式
        """
        print(f"当前机械臂状态为{self.piper.GetArmStatus().arm_status.ctrl_mode}")

        if self.piper.GetArmStatus().arm_status.ctrl_mode == 0x00:#如果是待机
            print("尝试从 待机模式->can控制模式")
            self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)#设置can模式
            print(self.piper.GetArmStatus().arm_status.ctrl_mode)
            time.sleep(1)
            print(self.piper.GetArmStatus().arm_status.ctrl_mode)
        elif self.piper.GetArmStatus().arm_status.ctrl_mode == 0x02:#如果是示教模式
            print("尝试从 示教模式->can控制模式")
            self.piper.MotionCtrl_1(0x02,0,0)#恢复，示教模式->待机模式(会恢复到别的模式吗？没有我可就写死了)
            # print(piper.GetArmStatus().arm_status.ctrl_mode)
            time.sleep(1)#这里必须要等切换到待机模式
            # print(piper.GetArmStatus().arm_status.ctrl_mode)

            self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)#设置can模式
            time.sleep(1)#这里也必须要等

            while( not self.piper.EnablePiper()):#使能机械臂
                time.sleep(0.01)

            self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)#设置can模式
            while(self.piper.GetArmStatus().arm_status.ctrl_mode != 0x01):#等待进入can控制模式
                time.sleep(0.01)
            print("成功切换到can控制模式")


        elif self.piper.GetArmStatus().arm_status.ctrl_mode == 0x01:#如果是can控制模式
            print("can控制模式") 
            while( not self.piper.EnablePiper()):#使能机械臂
                time.sleep(0.01)
            
        else:
            print(self.piper.GetArmStatus().arm_status.ctrl_mode)
        #清除夹爪错误使其可以被正常控制
        self.piper.GripperCtrl(0,1000,0x02, 0)
        self.piper.GripperCtrl(0,1000,0x01, 0)