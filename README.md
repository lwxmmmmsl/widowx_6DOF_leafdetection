# WidowX 250S / Isaac Lab 叶片检测与夹持仿真

## 1. 项目实现
| 项目 | 实现 |
|---|---|
| 机械臂 | 本机 Interbotix `wx250s.urdf.xacro` 展开；原几何、惯量、轴向、关节范围、速度/力矩上限 |
| EE 坐标 | `wx250s/base_link` → `wx250s/ee_gripper_link` |
| 检测 | 原 `perception_node.py`，同一 `best.pt`、CPU 推理、640 输入、IoU 0.5、只取 leaf；当前置信度 0.2 |
| 深度/PCA | 0.12–1.5 m、最少 80 点、步长 2 |
| 目标 | 原最近基座点规则、接近距离 0.08 m、避障半径 0.035 m、目标数 3；基座 YZ 法向投影与有界姿态搜索 |
| 运动学 | 本机 Modern Robotics 库与 Interbotix WX250S `M/Slist`；IK 容差 0.001、三组 SDK 初始猜测 |
| 时序 | 移动 2 s、加速 0.3 s、夹持等待 3 s；笛卡尔路径 0.05 s 间隔 |
| 流程 | 张开→接近→平移进入→合拢→等待→张开→退出→下一片 |
| 消息 | 同名 RGB-D、CameraInfo、检测消息、检测/测量服务与结束状态 |

`config/pipeline.yaml` 最初复制自真机；当前用户已将正式/预览置信度改为 0.2/0.1（真机原配置为 0.2/0.05），其余原有参数沿用原配置；仿真启动仅通过 ROS 参数覆盖 `use_sim_time=true` 和 `dry_run=false`（执行的是仿真后端）；`--dry-run` 保持原 `dry_run=true`。`--manual` 只关闭自动开始，便于检测调试。

相机的**真实分辨率、K/D/R/P、完整 base→RGB optical 变换、启动关节位置**必须从运行中的真机导出。原静态标定是 base→`camera_base`，不是 optical；因此没有把该值冒充 RGB 光学位姿。缺少导出文件时，默认的真机一致模式拒绝启动；`demo_camera.yaml` 明确属于演示参数。

PhysX PD、光照、帧率、植株尺寸/位置是仿真参数，在 `config/scene.yaml` 中单独列出。不能认为 PD 等价于真实 Dynamixel 的控制器/PWM。夹爪用原 URDF 两侧手指行程模拟，不输出真实 SPAD 读数。原工程仅检查 EE 中心路径与点云距离，仿真继续保留该逻辑；没有宣称加入了整臂避障或软叶片接触力模型。

## 2. 环境与架构

- Isaac：`/home/user/anaconda3/envs/env_isaaclab/bin/python`（3.11，当前 Isaac Sim 5.1）。
- ROS Humble：`/usr/bin/python3`（3.10）。
- YOLO：`/home/user/anaconda3/envs/env_perception/bin/python`（沿用已调好的推理环境）。
- 不在 Isaac 环境安装/导入 Humble 的 rclpy；用本地 Unix socket 传 EE 指令、原子 NPZ 文件传 RGB-D。
- 仿真 ROS 固定使用 **`ROS_DOMAIN_ID=77`、`ROS_LOCALHOST_ONLY=1`**，所有观察终端也设这两个变量。启动器不启动 Azure 真机驱动、Interbotix 驱动或原 `bringup.launch.py`。

```text
Isaac Lab：WX250S + tomato GLB + optical camera
      RGB-D / joints → 原子 NPZ → ROS bridge → 原 perception
      EE 执行 ← Unix socket ← 仿真 backend ← 原 control
                                       ↑ 原 workflow
```

## 3. 首次准备

```bash
cd /home/user/workspace/lwx/widowx_codex
./scripts/build_ros.sh
./scripts/test.sh
./scripts/test_sim_math.sh
```

接口在本目录的 `artifacts/ros_install` 编译，不改动真机工程的 install/build。当前目录已经完成这一步。重建无需重装 Isaac Lab 或下载另一套机械臂。

### 导出真实相机和初始关节参数（只读）

在真机相机、TF、joint_states、robot_state_publisher 已运行且机械臂静止的终端，保持它们实际使用的 ROS_DOMAIN_ID，执行：

```bash
cd /home/user/workspace/lwx/widowx_codex
./scripts/capture_real.sh
```

生成 `config/hardware.yaml` 和 `config/hardware.urdf`。该程序只订阅/查询，不发运动指令、不启动驱动。

默认机器人描述话题为 `/wx250s/robot_description`；如果你的真机发布在 `/robot_description`：

```bash
./scripts/capture_real.sh --description-topic /robot_description
```

超时会列出已收到的数据，不写猜测参数。检查会接受额外的单位变换 `world → wx250s/base_link` 固定关节，并忽略 XML 子元素顺序和等值数字的书写差异。实际轴向、限位、关节偏移或非零 world 安装变换等差异仍会停止导出，保存 `.unmatched.urdf`，并在报错中列出具体关节与字段。复制的 `reference/static_transforms.yaml` 仅用于来源追溯，不替代完整 optical TF。

## 4. 启动

终端 A（GPU/驱动正常的本机桌面）：

```bash
cd /home/user/workspace/lwx/widowx_codex
./scripts/run_sim.sh --profile config/hardware.yaml
```

未导出真机参数、仅查看场景时明确使用演示模式：

```bash
./scripts/run_sim.sh --profile config/demo_camera.yaml
```

无窗口运行加 `--headless`；它**仍需支持 RTX 的 NVIDIA GPU**。短程启动检查加 `--steps 120`。等待 `simulation_ready` 后启动终端 B：

```bash
cd /home/user/workspace/lwx/widowx_codex
./scripts/run_pipeline.sh
```

终端 B 的默认行为是自动检测并执行仿真夹持。初次只检查检测：

```bash
./scripts/run_pipeline.sh --manual --dry-run
```

保留手动触发、需要执行仿真运动：`./scripts/run_pipeline.sh --manual`。

每个进程的日志在 `artifacts/logs/`。仿真启动脚本会把完整 stdout/stderr 保存为 `simulator_日期_时间_PID.log`；运行异常还会在关闭 Kit 前保存到 `/tmp/widowx_codex/last_error.txt`（自定义 `--run-dir` 时在该目录）。`/tmp/isaaclab/logs/` 的日志可能只有框架警告，不能代替完整启动日志。Ctrl+C 停止管线后，再在终端 A 停止仿真。仿真退出时先关闭本地 IPC；由于帧已同步落盘且没有后台数据集写入任务，随后通过 Kit 的 `skip_cleanup=True` 接口直接结束独立仿真进程，避免相机 Replicator 的停止等待卡住。仿真运动有到位检查与超时；超时保持当前测得关节并返回失败。管线不创建真机后端。

### Isaac 内嵌相机和 YOLO 面板

有窗口启动时默认打开 **WidowX | Camera & YOLO** 面板：

- 左图：按真机内参映射的实时 RGB 图。
- 右图：原 `/perception/debug_image` 消息，保留真实 YOLO 分割/检测叠加；标注距原始图像时间的年龄。
- 状态：ROS 管线心跳、检测服务、原 workflow 返回的结果、已规划/已测量数量、当前仿真运动指令。
- **Refresh YOLO preview (no robot motion)**：只调用原 `/perception/debug_detect`，采用原调试阈值 0.05，不发布测量目标，也不会驱动机械臂。实际测量仍用原检测阈值 0.2。

只运行 `run_sim.sh` 时机械臂保持启动姿态，这是等待控制指令，不会自行巡航。
需要在另一个终端运行 `./scripts/run_pipeline.sh` 才会自动检测并测量；
只调试分割可运行 `./scripts/run_pipeline.sh --manual --dry-run`，再点击面板刷新按钮。
实测结果见 `artifacts/validation/panel_e2e_result.txt`，分割预览见 `artifacts/validation/yolo_preview.png`。

面板不会在仿真 Python 3.11 中重复加载 YOLO，仍使用原推理环境和模型。

关闭面板功能可加 `--no-debug-ui`。无界面运行默认不创建面板；测试 UI 初始化可显式加 `--headless --debug-ui`。
原始彩色消息和推理分辨率不变，面板仅缩小显示图。YOLO 图是最近一次推理结果，不是每个物理步自动推理。

### 检测与结束信号

观察终端：

```bash
source /opt/ros/humble/setup.bash
source /home/user/workspace/lwx/widowx_codex/artifacts/ros_install/setup.bash
export ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1
ros2 service call /perception/detect std_srvs/srv/Trigger '{}'
ros2 topic echo /perception/leaves --once --qos-durability transient_local
ros2 topic echo /workflow/status
```

手动测量时，将刚收到的 `sequence` 填入服务请求：

```bash
ros2 service call /control/measure wodowx_interfaces/srv/MeasureLeaves '{sequence: 1}'
```

图像可用 `rqt_image_view` 查看 `/perception/debug_camera_image` 与 `/perception/debug_image`。RViz 的 Fixed Frame 可选 `wx250s/base_link`，robot_state_publisher 已发布机械臂 TF；RobotModel 的描述话题是 `/robot_description`。

```bash
cd /home/user/workspace/lwx/widowx_codex
/usr/bin/python3 -m widowx_codex.debug --save artifacts/debug_frame
```

这会打印 RGB-D 序号、时间、有效深度范围、关节和 EE 坐标，并保存彩色图/毫米深度图；不触发检测或运动。

## 5. 番茄模型及相机

模型原始包围盒约为 X=[-102.4,105.1]、Y=[-0.11,311.99]、Z=[-85.59,90.81]，Y 向上。当前 `scene.yaml` 用 X 轴 +90° 转至 Z 向上，并显式设 `scale=0.001`，得到约 31.2 cm 高。**模型单位和真植物尺寸尚未实测确认**；修改 `plant.scale`、`position_m` 匹配你的实验摆放位置。相机 profile 不受植株摆放调整影响。

GLB 转 USD 保留几何、UV、基色、双面及原 alphaCutoff=0.13586128 的叶片透明裁切。法线纹理、AO 和金属粗糙度纹理未完整重建；目前采用几何法线与固定粗糙度。转换函数明确限制为本次单几何/材质模型，多材质资产请先转 USD。USD 路径也可直接填入 `plant.asset`。

RGB-D 来自同一个光学视角：RGB 输出 BGRA8，深度用 `distance_to_image_plane` 的光轴 Z（非斜距），换算 16UC1 毫米，无效值为 0。按真实 K/D 对渲染图进行重映射，再发布原 CameraInfo，因此支持非方形像素和 Brown/rational 畸变参数。三条消息使用完全相同的时间戳和 optical frame。没有模拟 Kinect 的双视点遮挡空洞、传感器噪声或曝光响应。

仿真运行不会快于墙钟，原节点的 `time.sleep(3)` 保持不变；若 GPU 渲染比实时慢，墙钟与仿真时间仍会存在差别。真实光照训练的 YOLO 对渲染植物可能检出不足；这会按原流程报告不足，**不会偷偷用模型真值替代检测结果**。

## 6. 文件与验证

- `reference/wodowx_control/`：原真机 Python 包与测试，未改动。
- `reference/wodowx_interfaces/`：原 ROS 自定义接口。
- `assets/wx250s.original.urdf`：原 Xacro 展开结果。
- `assets/wx250s.import.urdf`：仅资源路径本地化、USD 不允许的链接名斜线替换、删除 ROS 插件块；没有改变运动学参数。
- `widowx_codex/simulator.py`：Isaac 场景、物理步进、异步命令执行与 RGB-D。
- `initial_state.py`：处理仿真启动初值越界，保留原始采样与调整记录。
- `kinematics.py`：与 SDK 对齐的 FK、IK、整段笛卡尔路径预检查。
- `ros_control.py`：继承原控制节点，替换 `_get_backend()`；空目标列表直接返回 0，避免无目标时仍尝试张开夹爪。非空列表保持原测量逻辑。
- `ros_bridge.py` / `ipc.py`：ROS 发布与进程通信。
- `capture_real.py` / `debug.py`：只读配置导出与调试。
- `reference/sha256.json`：初始源文件摘要；`tests/test_contracts.py` 会对原工程逐文件比较。

`tests/test_ros_adapter.py` 使用合成 RGB-D 测试消息/TF 合约，**不是 GPU 渲染测试**。GPU 短程启动已确认资产加载、初值校验和 RGB-D 输出；已追加验证日志目标的接近、插入、夹持、退出运动；仍待确认 GLB 材质/尺度最终画面、YOLO 渲染域检测率和完整 ROS 自动测量循环。

实现以安装在本机的 Isaac Lab API 源码为准，接口参考：[URDF converter](https://isaac-sim.github.io/IsaacLab/develop/source/api/lab/isaaclab.sim.converters.html)、[Camera](https://isaac-sim.github.io/IsaacLab/v2.2.0/_modules/isaaclab/sensors/camera/camera.html)。第三方代码与模型来源见 `THIRD_PARTY.md`。
