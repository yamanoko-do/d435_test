import cv2
import numpy as np
from d435_test.calibration import calibrate_extrinsic_bysolvepnp
import math
from d435_test.camera import CameraD435

# 1. 获取相机内参
cam_intrinsic,_,_,_= CameraD435.get_intrinsics()
# 2. 由pnp计算外参
r_world2cam,t_world2cam=calibrate_extrinsic_bysolvepnp(
    chessboard_picpath="./data/eye2hand_images",
    chessboard_size = (5, 5, 8),
    cam_intrinsic=cam_intrinsic,
    confirm=False
    )
print(t_world2cam)
# 3. 处理T_end2base
pose_data = "./data/eye2hand_images/pose_data.txt"
R_end2base_list = []
t_end2base_list= []
with open(pose_data, "r") as file:
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

# 4. 计算cam2base
R_base2cam, t_base2cam = cv2.calibrateHandEye(
    R_gripper2base=R_end2base_list,
    t_gripper2base=t_end2base_list,
    R_target2cam=r_world2cam,
    t_target2cam=t_world2cam,
    method=cv2.CALIB_HAND_EYE_ANDREFF
)

print("基座到相机的旋转矩阵：\n", R_base2cam)
print("基座到相机的平移向量：\n", t_base2cam.reshape(-1))

R_base2cam, t_base2cam = cv2.calibrateHandEye(
    R_gripper2base=R_end2base_list,
    t_gripper2base=t_end2base_list,
    R_target2cam=r_world2cam,
    t_target2cam=t_world2cam,
    method=cv2.CALIB_HAND_EYE_TSAI
)

print("基座到相机的旋转矩阵：\n", R_base2cam)
print("基座到相机的平移向量：\n", t_base2cam.reshape(-1))

R_base2cam, t_base2cam = cv2.calibrateHandEye(
    R_gripper2base=R_end2base_list,
    t_gripper2base=t_end2base_list,
    R_target2cam=r_world2cam,
    t_target2cam=t_world2cam,
    method=cv2.CALIB_HAND_EYE_DANIILIDIS
)

print("基座到相机的旋转矩阵：\n", R_base2cam)
print("基座到相机的平移向量：\n", t_base2cam.reshape(-1))

R_base2cam, t_base2cam = cv2.calibrateHandEye(
    R_gripper2base=R_end2base_list,
    t_gripper2base=t_end2base_list,
    R_target2cam=r_world2cam,
    t_target2cam=t_world2cam,
    method=cv2.CALIB_HAND_EYE_HORAUD
)

print("基座到相机的旋转矩阵：\n", R_base2cam)
print("基座到相机的平移向量：\n", t_base2cam.reshape(-1))