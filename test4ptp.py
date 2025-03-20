import cv2
import numpy as np
from d435_test.camera import CameraD435
camera_matrix,dist_coeffs,_,_=CameraD435.get_intrinsics()

# 标定板参数（以棋盘格为例）
pattern_size = (5, 5)          # 内部角点数量（宽度，高度）
square_size = 0.008             # 棋盘格方格大小（单位：米）

file_path = "./data/eye2hand_images/pose_data.txt"
arm_poses_data = []
# 读取记录的位姿数据
with open(file_path, "r") as file:
    for line in file:
        # 去掉行首行尾的空白字符（如换行符）
        line = line.strip()
        # 将字符串转换为浮点数列表
        arm_pose = [float(value) for value in line.strip('[]').split(', ')]
        # 将解析后的数据添加到列表中
        arm_poses_data.append(arm_pose)

# 生成标定板在末端坐标系中的3D坐标（假设标定板固定在末端）
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), dtype=np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

# 存储所有3D点（基座坐标系）和对应的2D图像点
object_points = []  # 基座坐标系中的3D点
image_points = []   # 图像中的2D点

# 示例：遍历所有采集的数据（需替换为实际数据加载逻辑）
for each_arm_pose in arm_poses_data:
    # --------------------------------------------
    # 1. 从机器人读取末端在基座坐标系中的位姿（需替换为实际数据）
    # 假设位姿格式为 [x, y, z, rx, ry, rz]（平移 + 欧拉角）
    t_vec_base = np.array([each_arm_pose[0], each_arm_pose[1], each_arm_pose[2]], dtype=np.float32)          # 平移向量
    r_vec_base = np.array([0, 0, each_arm_pose[3]], dtype=np.float32)       # 旋转向量（欧拉角形式）
    
    # 将欧拉角转换为旋转矩阵
    R_base_end, _ = cv2.Rodrigues(r_vec_base)
    
    # --------------------------------------------
    # 2. 将标定板角点从末端坐标系转换到基座坐标系
    obj_points_base = []
    for point in objp:
        # 标定板角点在末端坐标系中的坐标
        point_end = point.reshape(3, 1)
        # 转换到基座坐标系：R_base_end * point_end + t_vec_base
        point_base = np.dot(R_base_end, point_end) + t_vec_base.reshape(3, 1)
        obj_points_base.append(point_base.ravel())
    object_points.append(np.array(obj_points_base, dtype=np.float32))
    
    # --------------------------------------------
    # 3. 从图像中检测标定板角点（需替换为实际检测逻辑）
    # 假设image是当前帧的图片
    # ret, corners = cv2.findChessboardCorners(image, pattern_size, None)
    # if ret:
    #     corners_refined = cv2.cornerSubPix(...)  # 亚像素细化
    #     image_points.append(corners_refined)
    
    # 为了示例，这里生成虚拟图像点（实际应使用真实检测数据）
    projected_points, _ = cv2.projectPoints(
        objp, r_vec_base, t_vec_base, camera_matrix, dist_coeffs)
    image_points.append(projected_points.reshape(-1, 2))

# 转换为NumPy数组
object_points = np.array(object_points)
image_points = np.array(image_points)

# --------------------------------------------
# 4. 使用solvePnP进行眼在手外标定
ret, rvec, tvec = cv2.solvePnP(
    object_points.reshape(-1, 3),  # 所有3D点展平
    image_points.reshape(-1, 2),   # 所有2D点展平
    camera_matrix,
    dist_coeffs
)

if ret:
    print("标定成功！")
    # 旋转向量和平移向量表示相机到基座坐标系的变换
    print("旋转向量 (rvec):\n", rvec)
    print("平移向量 (tvec):\n", tvec)
    
    # 转换为旋转矩阵
    R_cam_to_base, _ = cv2.Rodrigues(rvec)
    print("旋转矩阵:\n", R_cam_to_base)
    
    # 构建变换矩阵（基座坐标系到相机坐标系）
    T_cam_to_base = np.eye(4)
    T_cam_to_base[:3, :3] = R_cam_to_base
    T_cam_to_base[:3, 3] = tvec.ravel()
    print("变换矩阵 (基座到相机):\n", T_cam_to_base)
else:
    print("标定失败！")

# --------------------------------------------
# 可选：计算重投影误差验证精度
total_error = 0
for i in range(len(object_points)):
    projected, _ = cv2.projectPoints(
        object_points[i], rvec, tvec, camera_matrix, dist_coeffs)
    error = cv2.norm(image_points[i], projected, cv2.NORM_L2)
    total_error += error

mean_error = total_error / len(object_points)
print(f"重投影误差: {mean_error} 像素")