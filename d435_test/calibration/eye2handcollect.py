from d435_test.robot_arm import DobotSession,DobotTypes, PiperClass
import time
import os
import glob
import cv2
from d435_test.camera import CameraD435
import pyrealsense2 as rs
import math
import random
import threading
def eye2hand_collect_dobot(save_dir,pose_file_path):
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

    # 相机
    cam=CameraD435()
    cam.enable_stream(rs.stream.color, 1920,1080, rs.format.bgr8, 30)
    cam.start()
    # 机械臂
    dobot_chd = DobotSession(control_lib_path="./d435_test/robot_arm/dobot/Linux/x64/libDobotDll.so")
    state,_,_=dobot_chd.ConnectDobot(portName=dobot_chd.SearchDobot()[0],baudrate=115200)
    #dobot_chd.ClearAllAlarmsState()
    state=DobotTypes.CONNECT_RESULT[state]
    print(f"连接{state}")
    dobot_chd.SetEndEffectorSuctionCup(enableCtrl=1,isSucked=1,isQueued=0)


    def set_rhead():
        last_pose=dobot_chd.GetPose()
        while True:
            current_pose=dobot_chd.GetPose()
            # 计算前三项的差值的平方
            squared_diffs = [(current_pose[i] - last_pose[i]) ** 2 for i in range(3)]
            # 求和
            sum_of_squared_diffs = sum(squared_diffs)
            # 开根号
            distance = math.sqrt(sum_of_squared_diffs)
            
            if distance > 50:
                last_pose = current_pose
                print(f"末端位移为，{distance}")
                random_float = random.uniform(-45, 45)
                dobot_chd.SetPTPCmd(DobotTypes.PTPMode.PTP_MOVL_XYZ_Mode, current_pose[0], current_pose[1], current_pose[2], random_float, isQueued = 0)
                
            time.sleep(0.1)


    thread = threading.Thread(target=set_rhead)
    thread.daemon = True
    thread.start()

    try:
        while True:
            frames = cam.get_frame()
            color_frame = frames["color"] 
            # 显示实时画面
            cv2.imshow('RealSense Camera', color_frame)
            key = cv2.waitKey(1)

            # 按下's'保存照片
            if key & 0xFF == ord('s'):
                save_path = os.path.join(save_dir, f"pose_{photo_count}.jpg")
                cv2.imwrite(save_path, color_frame)
                print(f"照片已保存至：{save_path}")
                pose_list=dobot_chd.GetPose()
                
                print(pose_list)
                with open(pose_file_path, "a") as f:  # 使用追加模式
                    f.write(str(pose_list) + "\n")  # 将 pose_list 转换为字符串并写入文件

                photo_count += 1

            # 按下'q'或ESC退出
            if key & 0xFF in (ord('q'), 27):
                break

    finally:
        # 清理资源
        cam.stop()
        dobot_chd.SetEndEffectorSuctionCup(enableCtrl=0,isSucked=1,isQueued=0)
        dobot_chd.DisconnectDobot()
        cv2.destroyAllWindows()

from piper_sdk import *
def eye2hand_collect_piper(save_dir,pose_file_path):
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

    # 初始化相机
    cam=CameraD435()
    cam.enable_stream(rs.stream.color, 1280,720, rs.format.bgr8, 30)
    cam.start()

    # 初始化piper
    piper = PiperClass(can_name = "can_piper")

    #初始化一个窗口
    cv2.namedWindow('RealSense Camera', cv2.WINDOW_NORMAL)
    try:
        while True:
            frames = cam.get_frame()
            color_frame = frames["color"] 
            # 显示实时画面
            cv2.imshow('RealSense Camera', color_frame)
            key = cv2.waitKey(1)

            # 按下's'保存照片
            if key & 0xFF == ord('s'):
                save_path = os.path.join(save_dir, f"pose_{photo_count}.jpg")
                cv2.imwrite(save_path, color_frame)
                print(f"照片已保存至：{save_path}")
                pose_list=piper.getpose()
                
                print(pose_list)
                with open(pose_file_path, "a") as f:  # 使用追加模式
                    f.write(str(pose_list) + "\n")  # 将 pose_list 转换为字符串并写入文件

                photo_count += 1

            # 按下'q'或ESC退出
            if key & 0xFF in (ord('q'), 27):
                break

    finally:
        # 清理资源
        cam.stop()

        # print(piper.piper.get_ctrl_mode())
        # if piper.get_ctrl_mode() == 0x02:#如果是示教模式
        #     piper.piper.MotionCtrl_1(0x02,0,0)
        #     piper.piper.MotionCtrl_1(0x02,0,0)

        # piper.control_gripper(isclose=False)
        # piper.disconnect()
        cv2.destroyAllWindows()


