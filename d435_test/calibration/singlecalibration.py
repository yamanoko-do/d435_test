
import glob
import cv2
import numpy as np

def calibrate_intrinsic(chessboard_folder):
    """
    标定内参
    """
    # 棋盘格的尺寸（内角点数量，例如 9x6 棋盘格的内角点为 8x5）和边长（mm）
    chessboard_size = (8, 5, 26.36)
    
    #加载拍摄的棋盘格照片
    images_path_list = glob.glob(chessboard_folder+'/*.jpg')
    img_ = cv2.imread(images_path_list[0])
    resolution=img_.shape[:2]

    p_pixel_list = []  # 世界坐标点
    p_world_list = []  # 图像坐标点

    p_world=np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    p_world[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * chessboard_size[2]

    for fname in images_path_list:
        #获取像素坐标
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size[:2], None)
        p_pixel_list
        if ret:
            p_world_list.append(p_world)
            p_pixel_list.append(corners)
            # 绘制角点并显示
            cv2.drawChessboardCorners(img, chessboard_size[:2], corners, ret)
            #显示点的序号
            for i, corner in enumerate(corners):
                corner = tuple(map(int, corner.ravel()))  # 确保corner是一个包含两个整数值的元组
                cv2.putText(img, str(i+1), corner, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow('Chessboard Corners', img)
            cv2.waitKey(500)
            # key=cv2.waitKey(0)
            # if key == ord('q'):  # 如果按下 'q' 键，则退出
            #     print("用户终止标定过程。")
            #     exit()

    cv2.destroyAllWindows()

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
        cv2.imshow('Undistorted Image', undistorted_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()