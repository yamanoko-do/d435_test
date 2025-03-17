
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import glob
from ..hardware import CameraD435
def take_photo(save_dir):
    """
    使用d435拍照
    """
    # 创建保存目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)

    # 初始化照片计数器
    photo_count = 1
    
    # 检查现有文件并确定起始编号
    existing_files = glob.glob(os.path.join(save_dir, "d435_*.jpg"))
    if existing_files:
        max_num = 0
        for file_path in existing_files:
            try:
                # 从文件名中提取数字
                filename = os.path.splitext(os.path.basename(file_path))[0]
                number = int(filename.split('_')[-1])
                if number > max_num:
                    max_num = number
            except (ValueError, IndexError):
                continue
        photo_count = max_num + 1

    # 配置相机
    cam=CameraD435()
    cam.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # 启动相机
    cam.start()

    try:
        while True:
            frames = cam.get_frame()
            color_frame = frames["color"] 
            # 显示实时画面
            cv2.imshow('RealSense Camera', color_frame)
            key = cv2.waitKey(1)

            # 按下's'保存照片
            if key & 0xFF == ord('s'):
                save_path = os.path.join(save_dir, f"d435_{photo_count}.jpg")
                cv2.imwrite(save_path, color_frame)
                print(f"照片已保存至：{save_path}")
                photo_count += 1

            # 按下'q'或ESC退出
            if key & 0xFF in (ord('q'), 27):
                break

    finally:
        # 清理资源
        cam.stop()
        cv2.destroyAllWindows()
