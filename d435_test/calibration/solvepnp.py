import glob
import cv2
import numpy as np
from typing import Tuple, List
from ..camera import CameraD435
def calibrate_extrinsic_bysolvepnp(chessboard_picpath: str, chessboard_size: Tuple[int, int, float],cam_intrinsic: np.ndarray, confirm: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    通过PnP方法计算每张棋盘图片的外参world2cam(旋转矩阵和平移向量)
    Args:
        chessboard_picpath (str): 棋盘格照片路径
        chessboard_size (tuple): 包含三个元素的元组，表示棋盘格的尺寸，如 (8, 5, 26.36)
            - 内角点长 (int)
            - 内角点宽 (int)
            - 边长 (float, 毫米)
        cam_intrinsic (np.ndarray): 相机内参
        confirm (bool): 是否等待用户通过键盘确认 OpenCV 检测的角点
    
    Returns:
        tuple: 包含两个列表，分别是：
            - r_matrix_list (List[np.ndarray]): 每张图片的旋转矩阵
            - tvecs_list (List[np.ndarray]): 每张图片的平移向量
    """
    
    # 加载棋盘格照片
    images_path_list = glob.glob(chessboard_picpath + '/*.jpg')
    print(images_path_list)
    def sort_key(fname):
        # 提取文件名中的数字部分
        import re
        match = re.search(r'pose_(\d+)\.jpg', fname)
        if match:
            return int(match.group(1))
        return 0
    images_path_list.sort(key=sort_key)
    print(images_path_list)
    # 准备世界坐标点
    p_world = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    p_world[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * chessboard_size[2]
    # print(p_world)

    # 读取相机内参和畸变系数（假设已标定）
    # 这里需要你提供相机内参矩阵 mtx 和畸变系数 dist
    # 如果没有，可使用 cv2.calibrateCamera 事先标定
    mtx = cam_intrinsic
    dist = np.zeros((5, 1), dtype=np.float32)  # 假设无畸变
    
    r_matrix_list, tvecs_list = [], []

    for fname in images_path_list:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size[:2], None)
        if ret:
            # 计算 PnP
            ret, rvec, tvec = cv2.solvePnP(p_world, corners, mtx, dist)
            r_matrix, _ = cv2.Rodrigues(rvec)  # 将旋转向量转换为旋转矩阵
            r_matrix_list.append(r_matrix)
            tvecs_list.append(tvec)
            
            # 绘制角点并显示
            cv2.drawChessboardCorners(img, chessboard_size[:2], corners, ret)
            #显示点的序号
            for i, corner in enumerate(corners):
                corner = tuple(map(int, corner.ravel()))  # 确保corner是一个包含两个整数值的元组
                cv2.putText(img, str(i+1), corner, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(img, fname, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            if confirm:
                cv2.imshow('Chessboard Corners - Press any key for next', img)
                key = cv2.waitKey(0)
                if key == ord('q'):  # 如果按下 'q' 键，则退出
                    print("用户终止标定过程。")
                    exit()
        else:
            print(f"{fname}检测棋盘格角点失败")
            
    cv2.destroyAllWindows()
    
    return r_matrix_list, tvecs_list
