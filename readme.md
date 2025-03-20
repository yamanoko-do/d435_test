该代码库包括三个部分
1. d435cam : 对d435封装的类，可以获取图像、深度图、点云、获取图像上一点的深度
2. dobot : 对dobotmagician的API封装的类
3. calibration : 使用opencv进行标定的程序等,单目标定、pnp、eye2hand


# 未完待续
1. dobotsession的好多文档字符串没写
2. 对于不同的机械臂标定，应该创建一个抽象基类，定义与opencv契合的格式

# 参考
- [PointCloudGeneration-git](https://github.com/musimab/PointCloudGeneration)
- [d435官方api+demo](https://dev.intelrealsense.com/docs/python2)
