import cv2
import numpy as np
from .solvepnp import calibrate_extrinsic_bysolvepnp
import math
from ..camera import CameraD435
import ast

def calculate_transformation_error(transformations):
    """
    计算一系列变换矩阵之间的误差波动
    
    参数:
    transformations: 变换矩阵列表
    
    返回:
    旋转误差的平均值和标准差
    平移误差的平均值和标准差
    """
    if len(transformations) < 2:
        return (0, 0), (0, 0)
    
    # 计算平均变换矩阵
    T_avg = np.mean(transformations, axis=0)
    
    # 计算每个变换与平均变换之间的差异
    rot_errors = []
    trans_errors = []
    
    for T in transformations:
        # 旋转部分误差 (使用角度差)
        R_diff = T[:3, :3] @ T_avg[:3, :3].T
        angle_error = np.linalg.norm(cv2.Rodrigues(R_diff)[0]) * 180 / np.pi
        rot_errors.append(angle_error)
        
        # 平移部分误差 (使用欧氏距离)
        t_diff = T[:3, 3] - T_avg[:3, 3]
        trans_error = np.linalg.norm(t_diff)
        trans_errors.append(trans_error)
    
    # 计算统计量
    rot_mean = np.mean(rot_errors)
    rot_std = np.std(rot_errors)
    trans_mean = np.mean(trans_errors)
    trans_std = np.std(trans_errors)
    
    return (rot_mean, rot_std), (trans_mean, trans_std)

def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    将欧拉角转换为旋转矩阵（ZYX顺序）
    
    参数:
    roll: 绕X轴旋转角度（弧度）
    pitch: 绕Y轴旋转角度（弧度）
    yaw: 绕Z轴旋转角度（弧度）
    
    返回:
    3x3 旋转矩阵
    """
    # 计算各个轴的旋转矩阵
    Rz = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    Ry = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])
    
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ])
    
    # 组合旋转矩阵 (ZYX顺序)
    R = Rz @ Ry @ Rx
    return R

def eye2hand_calibration(cam_intrinsic,chessboard_picpath,pose_data_path):
    # 1. 由pnp计算外参
    r_world2cam,t_world2cam=calibrate_extrinsic_bysolvepnp(
        chessboard_picpath=chessboard_picpath,
        chessboard_size = (9, 6, 21),
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
            # 去除行首尾的空白字符
            line = line.strip()
            if line:  # 跳过空行
                # 解析字符串为字典
                data = ast.literal_eval(line)
                # 提取end_pose并添加到列表
                end_pose = data['end_pose']
                euler_end2base = [math.radians(end_pose[3]), math.radians(end_pose[4]), math.radians(end_pose[5])]#读取姿态角度值并转换为弧度


                #R_end2base, _ = cv2.Rodrigues(np.array(R_end2base, dtype=np.float32))#将旋转向量转换为矩阵，输入为弧度

                R_end2base = euler_to_rotation_matrix(euler_end2base[0], euler_end2base[1], euler_end2base[2])
                t_end2base = np.array([end_pose[0],end_pose[1],end_pose[2]], dtype=np.float32)

                t_base2end=-R_end2base.T@t_end2base
                R_base2end_list.append(R_end2base.T)
                t_base2end_list.append(t_base2end)


                # R_base2end_list.append(R_end2base)
                # t_base2end_list.append(t_end2base)


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
        print()

        # 收集所有估计的世界到基座的变换矩阵
        T_w2b_list = []
        for R_b2e, t_b2e, R_w2c, t_w2c in zip(R_base2end_list, t_base2end_list, r_world2cam, t_world2cam):
            # 构造变换矩阵
            T_b2e = np.eye(4)
            T_b2e[:3, :3] = R_b2e
            T_b2e[:3, 3] = t_b2e

            T_c2b = np.eye(4)
            T_c2b[:3, :3] = R_cam2base
            T_c2b[:3, 3] = t_cam2base.reshape(-1)

            T_w2c = np.eye(4)
            T_w2c[:3, :3] = R_w2c
            T_w2c[:3, 3] = t_w2c.reshape(-1)

            # 计算世界到基座的变换
            T_w2b_estimated = T_b2e @ T_c2b @ T_w2c
            T_w2b_list.append(T_w2b_estimated)
            np.set_printoptions(precision=2, suppress=True)
            #print("世界到基座的变换矩阵：\n", T_w2b_estimated)
        
        # 计算并打印误差波动
        rot_stats, trans_stats = calculate_transformation_error(T_w2b_list)
        print(f"旋转误差 - 平均值: {rot_stats[0]:.4f}°，标准差: {rot_stats[1]:.4f}°")
        print(f"平移误差 - 平均值: {trans_stats[0]:.4f}，标准差: {trans_stats[1]:.4f}")
        print("-" * 80)


if __name__=="__main__":
    cam_intrinsic,_,_,_= CameraD435.get_intrinsics()
    # cam_intrinsic,dist=calibrate_intrinsic(chessboard_picpath="./data/eye2hand_images",chessboard_size = (5, 5, 8),confirm=False)
    # print(cam_intrinsic)
    chessboard_picpath="./data/eye2hand_images"
    pose_data_path = "./data/eye2hand_images/pose_data.txt"
    eye2hand_calibration(cam_intrinsic=cam_intrinsic,chessboard_picpath=chessboard_picpath,pose_data_path=pose_data_path)

