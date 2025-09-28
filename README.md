该代码库包括三个部分

1. d435cam : 对d435封装的类，可以获取图像、深度图、点云、获取图像上一点的深度
2. dobot : 对dobotmagician的API封装的类
3. piper：对piper封装的类用于标定
4. calibration : 使用opencv进行标定的程序等,单目标定、pnp、eye2hand

# 使用
按照calibration_export.py中的说明进行相机&&眼在手外&&工具坐标系标定

# Piper 

can_piper

- 查找can：bash ./d435_test/robot_arm/piper/piper_sdk/piper_sdk/find_all_can_port.sh
- 激活can：bash ./d435_test/robot_arm/piper/piper_sdk/piper_sdk/can_activate.sh can_piper 1000000 "3-1.1:1.0"

# 参考

- [PointCloudGeneration-git](https://github.com/musimab/PointCloudGeneration)
- [d435官方api+demo](https://dev.intelrealsense.com/docs/python2)
