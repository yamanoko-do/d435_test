from d435_test.calibration import take_photo,pnp_Checkchessboard,calibrate_intrinsic,eye2hand_calibration,eye2hand_collect_piper
from d435_test.camera import CameraD435
from d435_test.robot_arm import DobotSession,DobotTypes
import numpy as np
import time
if __name__=="__main__":
    '''
    拍摄rgb照片
    '''
    #take_photo(save_dir="./data/chessboard_images")
    '''
    实时解算pnp
    '''
    # chessboard_size = (11, 8, 15) #角点数即格子数减去1
    # cam_intrinsic,cam_coe,_,_= CameraD435.get_intrinsics()
    # pnp_Checkchessboard(chessboard_size=chessboard_size,cam_intrinsic=cam_intrinsic)
    '''
    相机内参标定
    '''
    #matrix,dist=calibrate_intrinsic(chessboard_picpath="./data/chessboard_images",chessboard_size = (9, 6, 21),confirm=True)
    '''
    眼在手外标定
    '''
    # 1. 收集数据
    # save_dir="./data/eye2hand_images"
    # pose_file_path = save_dir+"/pose_data.txt"
    # eye2hand_collect_piper(save_dir=save_dir,pose_file_path=pose_file_path)
    # 2. 计算
    # cam_intrinsic,_,_,_= CameraD435.get_intrinsics()
    # print(cam_intrinsic)
    cam_intrinsic,dist=calibrate_intrinsic(chessboard_picpath="./data/eye2hand_images",chessboard_size = (11, 8, 15),confirm=False)
    

    # chessboard_picpath="./data/eye2hand_images"
    # pose_data_path = "./data/eye2hand_images/pose_data.txt"
    # eye2hand_calibration(cam_intrinsic=cam_intrinsic,chessboard_picpath=chessboard_picpath,pose_data_path=pose_data_path)
    # # 3. 验证:尝试沿着相机z轴移动
    # dobot_chd = DobotSession(control_lib_path="./d435_test/robot_arm/dobot/Linux/x64/libDobotDll.so")
    # state,_,_=dobot_chd.ConnectDobot(portName=dobot_chd.SearchDobot()[0],baudrate=115200)
    # R_cam2base = np.array(
    #     [[0.6583838938578689, 0.5280620464653956, -0.5363591365785881],
    #     [0.7359896373980563, -0.6009040660116564, 0.31182295793179826],
    #     [-0.1576385167122167, -0.6000539796830255, -0.7842737529175223]]
    # )
    # t_cam2base = np.array([485.51578514980116, -175.05358005625143,310])

    # T_cam2base = np.eye(4)
    # T_cam2base[:3, :3] = R_cam2base  # 旋转部分
    # T_cam2base[:3, 3] = t_cam2base   # 平移部分

    # try:
    #     for i in range(330,471,10):
    #         P_cam=[0,0,i,1]
    #         P_base=T_cam2base@P_cam
    #         print(P_base)
    #         dobot_chd.ClearAllAlarmsState()
    #         dobot_chd.SetPTPCmd(DobotTypes.PTPMode.PTP_MOVL_XYZ_Mode, P_base[0], P_base[1], P_base[2], 0, isQueued = 0)
    #         time.sleep(0.5)
    # except KeyboardInterrupt as E:
    #     dobot_chd.DisconnectDobot()
    