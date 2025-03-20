import cv2
import numpy as np
import math
file_path = "./data/eye2hand_images/pose_data.txt"
R_end2base_list = []
t_end2base_list= []
# 读取记录的位姿数据
with open(file_path, "r") as file:
    for line in file:
        # 去掉行首行尾的空白字符（如换行符）
        line = line.strip()
        # 将字符串转换为浮点数列表
        arm_pose = [float(value) for value in line.strip('[]').split(', ')]
        # 将解析后的数据添加到列表中
        R_end2base=[0,0,math.radians(arm_pose[3])]
        R_end2base, _ = cv2.Rodrigues(np.array(R_end2base, dtype=np.float32))#该函数输入为弧度
        t_end2base = [arm_pose[0],arm_pose[1],arm_pose[2]]

        R_end2base_list.append(R_end2base)
        t_end2base_list.append(np.array(t_end2base, dtype=np.float32))

