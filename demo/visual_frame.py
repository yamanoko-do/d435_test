import matplotlib.pyplot as plt
import numpy as np
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager

# 给定旋转矩阵
R = np.array(
    [[0.6583838938578689, 0.5280620464653956, -0.5363591365785881], [0.7359896373980563, -0.6009040660116564, 0.31182295793179826], [-0.1576385167122167, -0.6000539796830255, -0.7842737529175223]]
)

# 给定平移向量
t = np.array([485.51578514980116, -175.05358005625143, 300]) / 1000  # 转换为米

# 组合为4×4变换矩阵
T_base_cam = np.eye(4)
T_base_cam[:3, :3] = R
T_base_cam[:3, 3] = t

# 创建变换管理器
tm = TransformManager()
tm.add_transform("camera", "base", T_base_cam)

# 可视化坐标系
ax = tm.plot_frames_in("base", s=0.1)
ax.set_xlim((-0.5, 0.5))
ax.set_ylim((-0.5, 0.5))
ax.set_zlim((-0.5, 0.5))
plt.show()
