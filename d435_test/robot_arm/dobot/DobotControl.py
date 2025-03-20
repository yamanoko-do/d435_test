# import threading

# import DobotTypes 
# import DobotAPI 

# #将dll读取到内存中并获取对应的CDLL实例
# #Load Dll and get the CDLL object
# api = DobotAPI.load()
# #建立与dobot的连接
# #Connect Dobot
# state = DobotAPI.ConnectDobot(api, "COM5", 115200)[0]
# print("Connect status:",DobotTypes.CONNECT_RESULT[state])

# if (state == DobotTypes.DobotConnect.DobotConnect_Successfully):
    
#     #清空队列
#     #Clean Command Queued
#     DobotAPI.SetQueuedCmdClear(api)
    
#     #设置运动参数
#     #Async Motion Params Setting
#     DobotAPI.SetHOMEParams(api, 200, 200, 200, 200, isQueued = 1)
#     DobotAPI.SetPTPJointParams(api, 200, 200, 200, 200, 200, 200, 200, 200, isQueued = 1)
#     DobotAPI.SetPTPCommonParams(api, 100, 100, isQueued = 1)

#     #回零
#     #Async Home
#     DobotAPI.SetHOMECmd(api, temp = 0, isQueued = 1)

#     #设置ptpcmd内容并将命令发送给dobot
#     #Async PTP Motion
#     for i in range(0, 5):
#         if i % 2 == 0:
#             offset = 50
#         else:
#             offset = -50
#         lastIndex = DobotAPI.SetPTPCmd(api, DobotTypes.PTPMode.PTP_MOVL_XYZ_Mode, 200 + offset, offset, offset, offset, isQueued = 1)[0]

#     #开始执行指令队列
#     #Start to Execute Command Queue
#     DobotAPI.SetQueuedCmdStartExec(api)

#     #如果还未完成指令队列则等待
#     #Wait for Executing Last Command 
#     while lastIndex > DobotAPI.GetQueuedCmdCurrentIndex(api)[0]:
#         DobotAPI.dSleep(100)

#     #停止执行指令
#     #Stop to Execute Command Queued
#     DobotAPI.SetQueuedCmdStopExec(api)

# #断开连接
# #Disconnect Dobot
# DobotAPI.DisconnectDobot(api)
