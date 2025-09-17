import matplotlib.pyplot as plt
import numpy as np
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager

# 给定旋转矩阵
R = np.array(
         [[-0.012813660649991414, 0.9886788030831055, -0.14949927904476462], [0.9991678648782998, 0.0184499826734648, 0.03637548534230565], [0.03872193041786918, -0.14890877231884758, -0.9880925005439557]]
)

# 给定平移向量
t = np.array(  [457.5312970130829, -75.75914090197986, 714.1815285995431]) / 1000  # 转换为米

# 组合为4×4变换矩阵
T_base_cam = np.eye(4)
T_base_cam[:3, :3] = R
T_base_cam[:3, 3] = t
print(T_base_cam)

# 创建变换管理器
tm = TransformManager()
tm.add_transform("camera", "base", T_base_cam)

# 可视化坐标系
ax = tm.plot_frames_in("base", s=0.1)
ax.set_xlim((-0.5, 0.5))
ax.set_ylim((-0.5, 0.5))
ax.set_zlim((-0.5, 0.5))
plt.show()
