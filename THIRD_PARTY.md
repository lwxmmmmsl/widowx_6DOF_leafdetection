# 来源

- `reference/wodowx_control` / `wodowx_interfaces`：用户本机 `/home/user/workspace/lwx/wodowx_control/src`，原包声明 Apache-2.0；完整保留原文件。
- `reference/mr_descriptions.py`、WX250S URDF 与网格：本机 Interbotix 官方工作区；保留文件头，许可证另存 `reference/LICENSE.interbotix_manipulators` / `LICENSE.interbotix_toolboxes`。
- `reference/modern_robotics`：本机 Modern Robotics 1.1.1，NxRLab，MIT；许可证见 `reference/LICENSE.modern_robotics`，上游 https://github.com/NxRLab/ModernRobotics 。
- `assets/tomato_plant.glb`：用户提供的本地文件。模型授权信息未提供，不另行授予模型再分发权。
- `reference/static_transforms.yaml`：用户本机标定配置，只追溯来源，不用作 RGB optical 的完整变换。

原检测及规划源码保持不变。仿真适配层额外加入基座 YZ 法向投影、工具 +X 伸入轴修正与姿态容差搜索；派生导入 URDF 移除 mimic，双指按原行程独立驱动。资产转换中的链接名映射见 `assets/link_names.json`。
