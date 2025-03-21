import cv2
import numpy as np
from .solvepnp import calibrate_extrinsic_bysolvepnp
import math
from ..camera import CameraD435

def eye2hand_calibration(cam_intrinsic,chessboard_picpath,pose_data_path):
    # 1. 由pnp计算外参
    r_world2cam,t_world2cam=calibrate_extrinsic_bysolvepnp(
        chessboard_picpath=chessboard_picpath,
        chessboard_size = (5, 5, 8),
        cam_intrinsic=cam_intrinsic,
        confirm=False
        )
    #print(t_world2cam)
    # 2. 处理T_end2base
    pose_data_path = "./data/eye2hand_images/pose_data.txt"
    R_base2end_list = []
    t_base2end_list= []
    with open(pose_data_path, "r") as file:
        for line in file:
            # 去掉行首行尾的空白字符（如换行符）
            line = line.strip()
            # 将字符串转换为浮点数列表
            arm_pose = [float(value) for value in line.strip('[]').split(', ')]
            # 将解析后的数据添加到列表中
            R_end2base=[0,0,math.radians(arm_pose[3])]
            R_end2base, _ = cv2.Rodrigues(np.array(R_end2base, dtype=np.float32))#该函数输入为弧度
            t_end2base = np.array([arm_pose[0],arm_pose[1],arm_pose[2]], dtype=np.float32)
            t_base2end=-R_end2base.T@t_end2base
            R_base2end_list.append(R_end2base.T)
            t_base2end_list.append(t_base2end)

    # 3. 计算cam2base
    methods = [
        "cv2.CALIB_HAND_EYE_ANDREFF",
        "cv2.CALIB_HAND_EYE_TSAI",
        "cv2.CALIB_HAND_EYE_DANIILIDIS",
        "cv2.CALIB_HAND_EYE_PARK",
        "cv2.CALIB_HAND_EYE_HORAUD"
    ]
    # t_base2end_list = [array / 1000 for array in t_base2end_list]
    # t_world2cam = [array / 1000 for array in t_world2cam]
    # 4. 循环计算每种方法的结果
    for method in methods:
        R_cam2base, t_cam2base = cv2.calibrateHandEye(
            R_gripper2base=R_base2end_list,
            t_gripper2base=t_base2end_list,
            R_target2cam=r_world2cam,
            t_target2cam=t_world2cam,
            method=eval(method)
        )
        print(f"使用 {method[19:]} 计算的结果：")
        print("相机到基座的旋转矩阵：\n", R_cam2base.tolist())
        print("相机到基座的平移向量：\n", t_cam2base.reshape(-1).tolist())
        print()  # 空一行

if __name__=="__main__":
    cam_intrinsic,_,_,_= CameraD435.get_intrinsics()
    # cam_intrinsic,dist=calibrate_intrinsic(chessboard_picpath="./data/eye2hand_images",chessboard_size = (5, 5, 8),confirm=False)
    # print(cam_intrinsic)
    chessboard_picpath="./data/eye2hand_images"
    pose_data_path = "./data/eye2hand_images/pose_data.txt"
    eye2hand_calibration(cam_intrinsic=cam_intrinsic,chessboard_picpath=chessboard_picpath,pose_data_path=pose_data_path)

