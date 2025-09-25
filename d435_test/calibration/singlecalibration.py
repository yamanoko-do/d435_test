
import glob
import cv2
import numpy as np
from typing import Tuple
def calibrate_intrinsic(chessboard_picpath: str,chessboard_size: str,confirm: bool = False) -> Tuple[np.ndarray ,np.ndarray]:
    """
    通过给定的棋盘格图片标定内参
    Args:
        chessboard_folder (str): 棋盘格照片路径
        chessboard_size (tuple): 包含三个元素的元组，表示棋盘格的尺寸，如 9x6 棋盘格的内角点为 8x5,边长为26.36mm则输入为(8, 5, 26.36)
            - 内角点长 (int)
            - 内角宽 (int)
            - 边长(float, 毫米)
        confirm (bool): 是否等待用户通过键盘确认opencv检测的角点
    Returns:
        tuple: 包含两个元素的元组，分别是：
            - matrix (np.ndarray): 一个3x3的数组,表示相机内参
            - dist (np.ndarray):  一个5x1的数组,表示畸变系数
    """

    #加载拍摄的棋盘格照片
    images_path_list = glob.glob(chessboard_picpath+'/*.jpg')
    #print(images_path_list)
    img_ = cv2.imread(images_path_list[0])
    resolution=img_.shape[:2]

    p_pixel_list = []  # 世界坐标点
    p_world_list = []  # 图像坐标点

    p_world=np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    p_world[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * chessboard_size[2]
    p_pixel_list=[]
    def sort_key(fname):
        # 提取文件名中的数字部分
        import re
        match = re.search(r'pose_(\d+)\.jpg', fname)
        if match:
            return int(match.group(1))
        return 0
    images_path_list.sort(key=sort_key)
    cv2.namedWindow('Chessboard Corners,press anykey to next', cv2.WINDOW_NORMAL)
    for fname in images_path_list:
        #获取像素坐标
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #ret, corners = cv2.findChessboardCorners(gray, chessboard_size[:2], None)
        #ret, corners = cv2.findChessboardCornersSB(gray, chessboard_size[:2], cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size[:2], None)

        # 如果找到角点，进行亚像素精化
        if ret:
            # 定义亚像素迭代的终止条件（迭代次数或精度）
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            # 精化角点位置（需要灰度图、初始角点、搜索窗口大小、死区、终止条件）
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        if ret:
            p_world_list.append(p_world)
            p_pixel_list.append(corners)
            # 绘制角点并显示
            cv2.drawChessboardCorners(img, chessboard_size[:2], corners, ret)
            #显示点的序号
            for i, corner in enumerate(corners):
                corner = tuple(map(int, corner.ravel()))  # 确保corner是一个包含两个整数值的元组
                cv2.putText(img, str(i+1), corner, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.putText(img, fname, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            if confirm:
                cv2.imshow('Chessboard Corners,press anykey to next', img)
                key=cv2.waitKey(0)
                if key == ord('q'):  # 如果按下 'q' 键，则退出
                    print("用户终止标定过程。")
                    exit()
            else:
                pass
            

    cv2.destroyAllWindows()
    print("World points list length:", len(p_world_list))
    print("Image points list length:", len(p_pixel_list))
    # 标定相机
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(p_world_list, p_pixel_list, resolution[::-1], None, None)

    # 打印相机内参矩阵和畸变系数
    print("相机内参矩阵 (Intrinsic Camera Matrix):")
    print(mtx.tolist())
    print("\n畸变系数 (Distortion Coefficients):")
    print(dist.tolist())
    dist=np.zeros((5, 1))


    # 可选：对一张图像进行去畸变
    
    if images_path_list:
        img = cv2.imread(images_path_list[0])
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        # 去畸变
        undistorted_img = cv2.undistort(img, mtx, dist, None, newcameramtx)
        cv2.namedWindow('Undistorted Image,press q to quit', cv2.WINDOW_NORMAL)
        cv2.imshow('Undistorted Image,press q to quit', undistorted_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return mtx ,dist