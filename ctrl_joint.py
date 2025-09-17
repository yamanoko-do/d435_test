#!/usr/bin/env python3
# -*-coding:utf8-*-
# 注意demo无法直接运行，需要pip安装sdk后才能运行
import time
from piper_sdk import *

if __name__ == "__main__":
    piper = C_PiperInterface_V2("can_piper")
    piper.ConnectPort()
    while( not piper.EnablePiper()):
        time.sleep(0.01)
    piper.GripperCtrl(0,1000,0x01, 0)
    #factor = 57295.7795 #1000*180/3.1415926
    factor = 1000
    position = [0,0,0,0,0,0,0]
    count = 0
    while True:
        count  = count + 1
        # print(count)
        if(count == 0):
            print("1-----------")
            position = [0,0,0,0,0,0,0]
        elif(count == 400):
            print("2-----------")
            position = [-1.544, 33.653, -29.525, -0.527, 22.337, -5.176]
        elif(count == 800):
            print("3-----------")
            position = [0.742, 33.653, -29.525, -4.818, 20.0, -90.816]
        elif(count == 1200):
            print("4-----------")
            position = [-2.714, 35.231, -35.427, -0.534, 29.508, 89.574]
        elif(count == 1600):
            print("5-----------")
            position = [-6.629, 65.126, -97.607, -2.257, 69.928, -1.917]
        elif(count == 2000):
            print("2-----------")
            position = [-56.863, 103.156, -89.872, 77.655, 74.306, -97.296]
        elif(count == 2400):
            print("2-----------")
            position = [54.837, 101.131, -89.985, 97.247, -52.333, -97.944]
        elif(count == 2800):
            print("2-----------")
            position = [-4.433, 124.54, -98.65, -2.063, 60.773, -97.858]
            count = 0
        
        joint_0 = round(position[0]*factor)
        joint_1 = round(position[1]*factor)
        joint_2 = round(position[2]*factor)
        joint_3 = round(position[3]*factor)
        joint_4 = round(position[4]*factor)
        joint_5 = round(position[5]*factor)
        #joint_6 = round(position[6]*1000*1000)
        piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
        #piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
        print(piper.GetArmStatus())
        print(position)
        time.sleep(0.005)
    