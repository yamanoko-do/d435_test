from typing import List,Tuple

from . import DobotAPI
from . import DobotTypes


class DobotSession:
    dobot_index = 0
    #加载api库
    def __init__(self, control_lib_path, dobotId=None, split='_'):
        if dobotId is None:
            self.dobotId = DobotSession.dobot_index
            DobotSession.dobot_index += 1
        else:
            self.dobotId = dobotId
        self.api = DobotAPI.load(control_lib_path,self.dobotId, split)
    def SetCmdTimeout():
        pass
    #
    def SearchDobot(self, maxLen=1000)->list:
        """
        搜索Dobot
        Returns:
            list: 串口列表,eg:['COM5','COM6'] .
        """
        return DobotAPI.SearchDobot(self.api, maxLen)

    def ConnectDobot(self, portName="", baudrate=115200)->Tuple[int,str,str]:
        """
        连接dobot
        Args:
            portName (str): SearchDobot方法返回的端口号.
            baudrate (int): 波特率.
        Returns:
            tuple:
                - state(int):连接状态,该值是DobotTypes.DobotConnect的类属性之一
                - dev_type(str):设备类型
                - dev_version(str):版本
        """
        return DobotAPI.ConnectDobot(self.api, portName, baudrate)

    def DisconnectDobot(self)->int:
        """
        断开dobot
        Returns:
            int:连接状态,该值是DobotTypes.DobotConnect的类属性之一
        """
        return DobotAPI.DisconnectDobot(self.api)

    def DobotExec(self):
        """
        在某些语言中,当调用 API 接口后,如果没有事件循环,应用程序将直接退出,
        导致指令没有下发至 Dobot 控制器。为避免这种情况发生,我们提供了事件循环接口,
        在应用程序退出前调用
        """
        DobotAPI.DobotExec(self.api)

    def SetQueuedCmdStartExec(self)->int:
        """
        执行队列中的指令:Dobot控制器开始循环查询指令队列,如果队列中有指令,则顺序取出并执行,
        执行完一条指令后才会取出下一条继续执行
        Returns:
            - int:通讯状态,该值是DobotTypes.DobotCommunicate的类属性之一
        """
        return DobotAPI.SetQueuedCmdStartExec(self.api)
    
    def SetQueuedCmdStopExec(self)->int:
        """
        停止执行队列中的指令:控制器停止循环查询队列并停止执行指令。
        在停止过程中若 Dobot 控制器正在执行一条指令，则待该指令执行完成后再停止
        Returns:
            int:通讯状态,该值是DobotTypes.DobotCommunicate的类属性之一
        """
        return DobotAPI.SetQueuedCmdStopExec(self.api)
    
    def SetQueuedCmdForceStopExec(self)->int:
        """
        停止执行队列中的指令:控制器停止循环查询队列并停止执行指令。
        在停止过程中若 Dobot 控制器正在执行一条指令，则该指令将会停止执行
        Returns:
            int:通讯状态,该值是DobotTypes.DobotCommunicate的类属性之一
        """
        return DobotAPI.SetQueuedCmdForceStopExec(self.api)

    def SetQueuedCmdClear(self)->int:
        """
        清空指令队列
        Returns:
            int:通讯状态,该值是DobotTypes.DobotCommunicate的类属性之一
        """
        return DobotAPI.SetQueuedCmdClear(self.api)
    
    def GetQueuedCmdCurrentIndex(self)->int:
        """
        查询当前执行完成的指令的索引,在 Dobot 控制器指令队列机制中，有一个 64 位内部计数器。
        当控制器每执行完一条指令时，该计数器将自动加一。通过该指令，可以查询当前执行完成的指令的索引
        Returns:
            int:指令索引
        """
        return DobotAPI.GetQueuedCmdCurrentIndex(self.api)

    def GetAutoLevelingResult(self):
        return DobotAPI.GetAutoLevelingResult(self.api)

    def SetQueuedCmdStartDownload(self, totalLoop, linePerLoop):
        return DobotAPI.SetQueuedCmdStartDownload(self.api, totalLoop, linePerLoop)

    def SetQueuedCmdStopDownload(self):
        return DobotAPI.SetQueuedCmdStopDownload(self.api)

    def SetDeviceSN(self, str_)->str:
        
        return DobotAPI.SetDeviceSN(self.api, str_)

    def GetDeviceSN(self):
        """
        获取设备序列号
        Returns:
            str:设备序列号
        """
        return DobotAPI.GetDeviceSN(self.api)

    def SetDeviceName(self, dev_name)->int:
        """
        设置设备名称
        Args:
            dev_name(str):名称
        Returns:
            int:通讯状态,该值是DobotTypes.DobotCommunicate的类属性之一
        """
        return DobotAPI.SetDeviceName(self.api, dev_name)

    def GetDeviceName(self) -> str:
        """
        获取设备名称
        Returns:
            str:设备名称
        """
        return DobotAPI.GetDeviceName(self.api)

    def GetDeviceVersion(self) -> str:
        """
        获取设备版本号
        Returns:
            str:设备版本号
        """
        majorVersion, minorVersion, revision=DobotAPI.GetDeviceVersion(self.api)
        version=str(majorVersion)+'.'+str(minorVersion)+'.'+str(revision)
        return version
    
    def GetDeviceTime(self)->int:
        """
        获取设备时钟
        Returns:
            int:设备时钟
        """
        return DobotAPI.GetDeviceTime(self.api)

    def SetDeviceWithL(self, isWithL):
        return DobotAPI.SetDeviceWithL(self.api, isWithL)

    def GetDeviceWithL(self):
        return DobotAPI.GetDeviceWithL(self.api)

    def GetPose(self) -> List[float]:
        """
        获取机械臂实时位姿,毫米和度,R轴为末端舵机坐标系相对于原点的姿态,其值为J1轴和J4轴之和
        Returns:
            - List:[x,y,z,rhead,joint1angle,joint2angle,joint3angle,joint4angle]
        """
        return DobotAPI.GetPose(self.api)

    def ResetPose(self, manual, rearArmAngle, frontArmAngle):
        return DobotAPI.ResetPose(self.api, manual, rearArmAngle, frontArmAngle)

    def GetPoseL(self):
        return DobotAPI.GetPoseL(self.api)

    def GetKinematics(self):
        return DobotAPI.GetKinematics(self.api)

    def GetAlarmsState(self, maxLen=100)->list:
        """
        获取系统报警状态,索引对应的含义见Dobot Magician ALARM说明文档
        Returns:
            List:元素为报警项索引,eg:['0x45', '0x48']
        """
        return DobotAPI.GetAlarmsState(self.api, maxLen)

    def ClearAllAlarmsState(self):
        """
        清除系统报警状态
        """
        return DobotAPI.ClearAllAlarmsState(self.api)

    def GetUserParams(self):
        return DobotAPI.GetUserParams(self.api)

    def SetAutoLevelingCmd(self, controlFlag, precision, isQueued=0):
        """
        执行自动调平功能,需要使用自动调平末端执行器
        Args:
            controlFlag (int): 使能标志
            precision (float): 调平精度,最小为0.02
            isQueued(bool):是否加入指令队列
        """
        return DobotAPI.SetAutoLevelingCmd(self.api, controlFlag, precision, isQueued)
    
    def SetHOMEParams(self, x, y, z, r, isQueued=0):
        """
        设置回零位置
        Args:
            x (float): 机械臂坐标系 x
            y (float): 机械臂坐标系 y
            z (float): 机械臂坐标系 z
            r (float): 机械臂坐标系 r
            isQueued(bool):是否加入指令队列
        Returns:
            int:指令在队列的索引号
        """
        return DobotAPI.SetHOMEParams(self.api, x, y, z, r, isQueued)

    def GetHOMEParams(self):
        """
        获取回零位置
        Returns:
            list:[x (float),y (float),z (float),r (float)]
        """
        return DobotAPI.GetHOMEParams(self.api)

    def SetHOMECmd(self, reserved=1, isQueued=1):
        """
        执行回零功能
        Args:
            reserved(int):作用不明,似乎是保留关键字
        """
        return DobotAPI.SetHOMECmd(self.api, reserved, isQueued)

    def SetArmOrientation(self, armOrientation, isQueued=0):
        return DobotAPI.SetArmOrientation(self.api, armOrientation, isQueued)

    def GetArmOrientation(self):
        return DobotAPI.GetArmOrientation(self.api)

    def SetHHTTrigMode(self, hhtTrigMode):
        return DobotAPI.SetHHTTrigMode(self.api, hhtTrigMode)

    def GetHHTTrigMode(self):
        return DobotAPI.GetHHTTrigMode(self.api)

    def SetHHTTrigOutputEnabled(self, isEnabled):
        return DobotAPI.SetHHTTrigOutputEnabled(self.api, isEnabled)

    def GetHHTTrigOutputEnabled(self):
        return DobotAPI.GetHHTTrigOutputEnabled(self.api)

    def GetHHTTrigOutput(self):
        return DobotAPI.GetHHTTrigOutput(self.api)

    def SetEndEffectorParams(self, xBias, yBias, zBias, isQueued=0):
        """
        设置末端坐标偏移量
        Args:
            x_bias,y_bias,z_bias(float):末端坐标偏移量
            isQueued(bool):是否加入指令队列
        """
        return DobotAPI.SetEndEffectorParams(self.api, xBias, yBias, zBias, isQueued)

    def GetEndEffectorParams(self)->List[float]:
        """
        获取末端坐标偏移量
        Returns:
            list:[x_bias,y_bias,z_bias]
        """
        return DobotAPI.GetEndEffectorParams(self.api)

    def SetEndEffectorLaser(self, enableCtrl, on, isQueued=0):
        return DobotAPI.SetEndEffectorLaser(self.api, enableCtrl, on, isQueued)

    def GetEndEffectorLaser(self):
        return DobotAPI.GetEndEffectorLaser(self.api)

    #气泵的实现G/SetEndEffectorSuctionCup有问题,无法吹气,这里实际调用G/SetEndEffectorGripper,不再封装夹爪的方法
    def GetEndEffectorSuctionCup(self)->List[int]:
        """
        获取气泵状态
        Returns:
            list:
                - enableCtrl:末端是否使能
                - isSucked:气泵是否吸气,1吸气,0吹气
        """
        return DobotAPI.GetEndEffectorGripper(self.api)

    def SetEndEffectorSuctionCup(self, enableCtrl, isSucked, isQueued=0):
        """
        设置气泵状态。
        Args:
            - enableCtrl(int) : 末端是否使能
            - isSucked(int) : 气泵是否吸气
            - isQueued(int) : 是否加入指令队列

        """
        print(type(enableCtrl))
        return DobotAPI.SetEndEffectorGripper(self.api, enableCtrl, isSucked, isQueued)

    def SetJOGJointParams(self, j1Velocity, j1Acceleration, j2Velocity, j2Acceleration, j3Velocity, j3Acceleration,
                          j4Velocity, j4Acceleration, isQueued=0):
        """
        设置点动时各关节坐标轴的动速度(degree/s)和加速度(degree/s^2)
        """
        return DobotAPI.SetJOGJointParams(self.api, j1Velocity, j1Acceleration, j2Velocity, j2Acceleration, j3Velocity,
                                          j3Acceleration, j4Velocity, j4Acceleration, isQueued)

    def GetJOGJointParams(self)->List[float]:
        """
        获取点动时各关节坐标轴的动速度(degree/s)和加速度(degree/s^2)
        """
        return DobotAPI.GetJOGJointParams(self.api)

    def SetJOGCoordinateParams(self, xVelocity, xAcceleration, yVelocity, yAcceleration, zVelocity, zAcceleration,
                               rVelocity, rAcceleration, isQueued=0):
        return DobotAPI.SetJOGCoordinateParams(self.api, xVelocity, xAcceleration, yVelocity, yAcceleration, zVelocity,
                                               zAcceleration, rVelocity, rAcceleration, isQueued)

    def GetJOGCoordinateParams(self)->List[float]:
        """
        获取点动时各坐标轴(笛卡尔)的速度(mm/s)和加速度(mm/s^2)
        """
        return DobotAPI.GetJOGCoordinateParams(self.api)

    def SetJOGLParams(self, velocity, acceleration, isQueued=0):
        return DobotAPI.SetJOGLParams(self.api, velocity, acceleration, isQueued)

    def GetJOGLParams(self):
        return DobotAPI.GetJOGLParams(self.api)

    def SetJOGCommonParams(self, value_velocityratio, value_accelerationratio, isQueued=0):
        return DobotAPI.SetJOGCommonParams(self.api, value_velocityratio, value_accelerationratio, isQueued)

    def GetJOGCommonParams(self)->List[float]:
        """
        获取点动速度百分比和加速度百分比(0~100)
        Returns:
            - velocityRatio(float):速度百分比
            - accelerationRatio(float):加速度百分比
        """
        return DobotAPI.GetJOGCommonParams(self.api)

    def SetJOGCmd(self, isJoint, cmd, isQueued=0):
        return DobotAPI.SetJOGCmd(self.api, isJoint, cmd, isQueued)

    def SetPTPJointParams(self, j1Velocity, j1Acceleration, j2Velocity, j2Acceleration, j3Velocity, j3Acceleration,
                          j4Velocity, j4Acceleration, isQueued=0):
        return DobotAPI.SetPTPJointParams(self.api, j1Velocity, j1Acceleration, j2Velocity, j2Acceleration, j3Velocity,
                                          j3Acceleration, j4Velocity, j4Acceleration, isQueued)

    def GetPTPJointParams(self):
        return DobotAPI.GetPTPJointParams(self.api)

    def SetPTPCoordinateParams(self, xyzVelocity, xyzAcceleration, rVelocity, rAcceleration, isQueued=0):
        return DobotAPI.SetPTPCoordinateParams(self.api, xyzVelocity, xyzAcceleration, rVelocity, rAcceleration,
                                               isQueued)

    def GetPTPCoordinateParams(self):
        return DobotAPI.GetPTPCoordinateParams(self.api)

    def SetPTPLParams(self, velocity, acceleration, isQueued=0):
        return DobotAPI.SetPTPLParams(self.api, velocity, acceleration, isQueued)

    def GetPTPLParams(self):
        return DobotAPI.GetPTPLParams(self.api)

    def SetPTPJumpParams(self, jumpHeight, zLimit, isQueued=0):
        return DobotAPI.SetPTPJumpParams(self.api, jumpHeight, zLimit, isQueued)

    def GetPTPJumpParams(self):
        return DobotAPI.GetPTPJumpParams(self.api)

    def SetPTPCommonParams(self, velocityRatio, accelerationRatio, isQueued=0):
        return DobotAPI.SetPTPCommonParams(self.api, velocityRatio, accelerationRatio, isQueued)

    def GetPTPCommonParams(self):
        return DobotAPI.GetPTPCommonParams(self.api)

    def SetPTPCmd(self, ptpMode, x, y, z, rHead, isQueued=0):
        """
        获取机械臂实时位姿,毫米和度,该函数不能在setpose后立即调用,需等待机械臂停下才能获取正确的值
        Args:
            - ptpMode : 运动模式,该值是DobotTypes.PTPMode的类属性之一
            - x(float) : 末端x坐标,单位mm
            - y(float) : 末端y坐标,单位mm
            - z(float) : 末端z坐标,单位mm
            - rHead(float) : 末端的R轴转角,为末端舵机坐标系相对于原点的姿态,其值为J1轴和J4轴之和
            - isQueued(bool):是否加入指令队列
        """

        return DobotAPI.SetPTPCmd(self.api, ptpMode, x, y, z, rHead, isQueued)

    def SetPTPWithLCmd(self, ptpMode, x, y, z, rHead, l, isQueued=0):
        return DobotAPI.SetPTPWithLCmd(self.api, ptpMode, x, y, z, rHead, l, isQueued)

    def SetCPParams(self, planAcc, juncitionVel, acc, realTimeTrack=0, isQueued=0):
        return DobotAPI.SetCPParams(self.api, planAcc, juncitionVel, acc, realTimeTrack, isQueued)

    def GetCPParams(self):
        return DobotAPI.GetCPParams(self.api)

    def SetCPCmd(self, cpMode, x, y, z, velocity, isQueued=0):
        return DobotAPI.SetCPCmd(self.api, cpMode, x, y, z, velocity, isQueued)

    def SetCPLECmd(self, cpMode, x, y, z, power, isQueued=0):
        return DobotAPI.SetCPLECmd(self.api, cpMode, x, y, z, power, isQueued)

    def SetARCParams(self, xyzVelocity, rVelocity, xyzAcceleration, rAcceleration, isQueued=0):
        return DobotAPI.SetARCParams(self.api, xyzVelocity, rVelocity, xyzAcceleration, rAcceleration, isQueued)

    def GetARCParams(self):
        return DobotAPI.GetARCParams(self.api)

    def SetARCCmd(self, cirPoint, toPoint, isQueued=0):
        return DobotAPI.SetARCCmd(self.api, cirPoint, toPoint, isQueued)

    def SetWAITCmd(self, waitTimeMs, isQueued=1):
        return DobotAPI.SetWAITCmd(self.api, waitTimeMs, isQueued)

    def SetTRIGCmd(self, address, mode, condition, threshold, isQueued=0):
        return DobotAPI.SetTRIGCmd(self.api, address, mode, condition, threshold, isQueued)

    def SetIOMultiplexing(self, address, multiplex, isQueued=0):
        return DobotAPI.SetIOMultiplexing(self.api, address, multiplex, isQueued)

    def GetIOMultiplexing(self, addr):
        return DobotAPI.GetIOMultiplexing(self.api, addr)

    def SetIODO(self, address, level, isQueued=0):
        return DobotAPI.SetIODO(self.api, address, level, isQueued)

    def GetIODO(self, addr):
        return DobotAPI.GetIODO(self.api, addr)

    def SetIOPWM(self, address, frequency, dutyCycle, isQueued=0):
        return DobotAPI.SetIOPWM(self.api, address, frequency, dutyCycle, isQueued)

    def GetIOPWM(self, addr):
        return DobotAPI.GetIOPWM(self.api, addr)

    def GetIODI(self, addr):
        return DobotAPI.GetIODI(self.api, addr)

    def SetEMotor(self, index, isEnabled, speed, isQueued=0):
        return DobotAPI.SetEMotor(self.api, index, isEnabled, speed, isQueued)

    def SetEMotorS(self, index, isEnabled, speed, distance, isQueued=0):
        return DobotAPI.SetEMotorS(self.api, index, isEnabled, speed, distance, isQueued)

    def GetIOADC(self, addr):
        return DobotAPI.GetIOADC(self.api, addr)

    def SetAngleSensorStaticError(self, rearArmAngleError, frontArmAngleError):
        return DobotAPI.SetAngleSensorStaticError(self.api, rearArmAngleError, frontArmAngleError)

    def GetAngleSensorStaticError(self):
        return DobotAPI.GetAngleSensorStaticError(self.api)

    def SetAngleSensorCoef(self, rearArmAngleCoef, frontArmAngleCoef):
        return DobotAPI.SetAngleSensorCoef(self.api, rearArmAngleCoef, frontArmAngleCoef)

    def GetAngleSensorCoef(self):
        return DobotAPI.GetAngleSensorCoef(self.api)

    def SetBaseDecoderStaticError(self, baseDecoderError):
        return DobotAPI.SetBaseDecoderStaticError(self.api, baseDecoderError)

    def GetBaseDecoderStaticError(self):
        return DobotAPI.GetBaseDecoderStaticError(self.api)

    def GetWIFIConnectStatus(self):
        return DobotAPI.GetWIFIConnectStatus(self.api)

    def SetWIFIConfigMode(self, enable):
        return DobotAPI.SetWIFIConfigMode(self.api, enable)

    def GetWIFIConfigMode(self):
        return DobotAPI.GetWIFIConfigMode(self.api)

    def SetWIFISSID(self, ssid):
        return DobotAPI.SetWIFISSID(self.api, ssid)

    def GetWIFISSID(self):
        return DobotAPI.GetWIFISSID(self.api)

    def SetWIFIPassword(self, password):
        return DobotAPI.SetWIFIPassword(self.api, password)

    def GetWIFIPassword(self):
        return DobotAPI.GetWIFIPassword(self.api)

    def SetWIFIIPAddress(self, dhcp, addr1, addr2, addr3, addr4):
        return DobotAPI.SetWIFIIPAddress(self.api, dhcp, addr1, addr2, addr3, addr4)

    def GetWIFIIPAddress(self):
        return DobotAPI.GetWIFIIPAddress(self.api)

    def SetWIFINetmask(self, addr1, addr2, addr3, addr4):
        return DobotAPI.SetWIFINetmask(self.api, addr1, addr2, addr3, addr4)

    def GetWIFINetmask(self):
        return DobotAPI.GetWIFINetmask(self.api)

    def SetWIFIGateway(self, addr1, addr2, addr3, addr4):
        return DobotAPI.SetWIFIGateway(self.api, addr1, addr2, addr3, addr4)

    def GetWIFIGateway(self):
        return DobotAPI.GetWIFIGateway(self.api)

    def SetWIFIDNS(self, addr1, addr2, addr3, addr4):
        return DobotAPI.SetWIFIDNS(self.api, addr1, addr2, addr3, addr4)

    def GetWIFIDNS(self):
        return DobotAPI.GetWIFIDNS(self.api)

    def SetColorSensor(self, isEnable, colorPort):
        return DobotAPI.SetColorSensor(self.api, isEnable, colorPort)

    def GetColorSensor(self):
        return DobotAPI.GetColorSensor(self.api)

    def SetInfraredSensor(self, isEnable, infraredPort):
        return DobotAPI.SetInfraredSensor(self.api, isEnable, infraredPort)

    def GetInfraredSensor(self, infraredPort):
        return DobotAPI.GetInfraredSensor(self.api, infraredPort)

    # FIRMWARE
    def UpdateFirmware(self, firmwareParams: DobotTypes.FirmwareParams):
        DobotAPI.UpdateFirmware(self.api, firmwareParams)

    def SetFirmwareMode(self, firmwareMode):
        DobotAPI.SetFirmwareMode(self.api, firmwareMode)

    def GetFirmwareMode(self):
        DobotAPI.GetFirmwareMode(self.api)

    # LOSTSTEP
    def SetLostStepParams(self, threshold, isQueued=0):
        DobotAPI.SetLostStepParams(self.api, threshold, isQueued)

    def SetLostStepCmd(self, isQueued=1):
        DobotAPI.SetLostStepCmd(self.api, isQueued)

    # UART4 Peripherals
    def GetUART4PeripheralsType(self, p_type):
        DobotAPI.GetUART4PeripheralsType(self.api, p_type)

    def SetUART4PeripheralsEnable(self, isEnable):
        DobotAPI.SetUART4PeripheralsEnable(self.api, isEnable)

    # Function Pluse Mode
    def SendPluse(self, pluseCmd: DobotTypes.PluseCmd, isQueued=0):
        DobotAPI.SendPluse(self.api, pluseCmd, isQueued)

    def SendPluseEx(self, pluseCmd):
        DobotAPI.SendPluseEx(self.api, pluseCmd)

    def GetServoPIDParams(self):
        DobotAPI.GetServoPIDParams(self.api)

    def SetServoPIDParams(self, pid: DobotTypes.PID, isQueued=0):
        DobotAPI.SetServoPIDParams(self.api, pid, isQueued)

    def GetServoControlLoop(self):
        return DobotAPI.GetServoControlLoop(self.api)

    def SetServoControlLoop(self, p_index, controlLoop, isQueued=0):
        DobotAPI.SetServoControlLoop(self.api, p_index, controlLoop, isQueued)

    def SaveServoPIDParams(self, p_index, controlLoop, isQueued=0):
        DobotAPI.SaveServoPIDParams(self.api, p_index, controlLoop, isQueued)

    def GetPoseEx(self, index):
        return DobotAPI.GetPoseEx(self.api, index)

    def SetHOMECmdEx(self, temp, isQueued=0):
        return DobotAPI.SetHOMECmdEx(self.api, temp, isQueued)

    def SetWAITCmdEx(self, waitTimeMs, isQueued=0):
        return DobotAPI.SetWAITCmdEx(self.api, waitTimeMs, isQueued)

    def SetEndEffectorParamsEx(self, xBias, yBias, zBias, isQueued=0):
        return DobotAPI.SetEndEffectorParamsEx(self.api, xBias, yBias, zBias, isQueued)

    def SetPTPJointParamsEx(self, j1Velocity, j1Acceleration, j2Velocity, j2Acceleration, j3Velocity, j3Acceleration,
                            j4Velocity, j4Acceleration, isQueued=0):
        return DobotAPI.SetPTPJointParamsEx(self.api, j1Velocity, j1Acceleration, j2Velocity, j2Acceleration,
                                            j3Velocity, j3Acceleration, j4Velocity, j4Acceleration, isQueued)

    def SetPTPLParamsEx(self, lVelocity, lAcceleration, isQueued=0):
        return DobotAPI.SetPTPLParamsEx(self.api, lVelocity, lAcceleration, isQueued)

    def SetPTPCommonParamsEx(self, velocityRatio, accelerationRatio, isQueued=0):
        return DobotAPI.SetPTPCommonParamsEx(self.api, velocityRatio, accelerationRatio, isQueued)

    def SetPTPJumpParamsEx(self, jumpHeight, maxJumpHeight, isQueued=0):
        return DobotAPI.SetPTPJumpParamsEx(self.api, jumpHeight, maxJumpHeight, isQueued)

    def SetPTPCmdEx(self, ptpMode, x, y, z, rHead, isQueued=0):
        return DobotAPI.SetPTPCmdEx(self.api, ptpMode, x, y, z, rHead, isQueued)

    def SetIOMultiplexingEx(self, address, multiplex, isQueued=0):
        return DobotAPI.SetIOMultiplexingEx(self.api, address, multiplex, isQueued)

    def SetEndEffectorSuctionCupEx(self, enableCtrl, on, isQueued=0):
        return DobotAPI.SetEndEffectorSuctionCupEx(self.api, enableCtrl, on, isQueued)

    def SetEndEffectorGripperEx(self, enableCtrl, on, isQueued=0):
        return DobotAPI.SetEndEffectorGripperEx(self.api, enableCtrl, on, isQueued)

    def SetIODOEx(self, address, level, isQueued=0):
        return DobotAPI.SetIODOEx(self.api, address, level, isQueued)

    def SetEMotorEx(self, index, isEnabled, speed, isQueued=0):
        return DobotAPI.SetEMotorEx(self.api, index, isEnabled, speed, isQueued)

    def SetEMotorSEx(self, index, isEnabled, speed, distance, isQueued=0):
        return DobotAPI.SetEMotorSEx(self.api, index, isEnabled, speed, distance, isQueued)

    def SetIOPWMEx(self, address, frequency, dutyCycle, isQueued=0):
        return DobotAPI.SetIOPWMEx(self.api, address, frequency, dutyCycle, isQueued)

    def SetPTPWithLCmdEx(self, ptpMode, x, y, z, rHead, l, isQueued=0):
        return DobotAPI.SetPTPWithLCmdEx(self.api, ptpMode, x, y, z, rHead, l, isQueued)

    def GetColorSensorEx(self, index):
        return DobotAPI.GetColorSensorEx(self.api, index)
