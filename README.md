# WidowX 250S / Isaac Lab 叶片检测与夹持仿真

基于 **`/home/user/workspace/lwx/wodowx_control` 的当前真机实现**，不是先前 SPAD 的实现。原工程未修改；其 Python 节点、ROS 接口、配置和测试按字节复制到 `reference/`。番茄模型来自 `/home/user/下载/tomato_plant.glb`，已复制进 `assets/`。

**当前验证范围：** 初值越界修复后，45 项 Python 3.10 回归测试和 14 项 Isaac Python 3.11 离线测试通过。已用真实 `hardware.yaml` 完成无界面 GPU 启动检查、PhysX 初始化和 5 个物理步进，并实际输出 RGB-D；这次报错中的肩关节/手指越界已通过处理。已额外验证相机→原 YOLO→ROS bridge→内嵌面板回传链路：一次调试预览得到 5 个预测、4 个 leaf 候选（阈值 0.05）。这不代表正式检测阈值下的稳定检出率；完整检测夹持循环仍未验证。最新检查见 `artifacts/validation/initial_state_smoke_final.txt` 及 `initial_state_smoke.json`；此前 `isaac_startup.log` 是早期受限环境的历史失败日志。

## 夹持器从基座侧正向伸入：工具轴修正

WX250S 的 `ee_gripper_link` **+X 是夹指伸出方向，Y 是开合方向**。
此前错误地把 +Z 对齐伸入路径，会出现位置从基座侧移动但夹指横着走的现象。
现在用 +X 对齐伸入方向，+Y 对齐处理后的叶片法向，+Z = X × Y。

设基座系接触点为 p、单位法向为 n，伸入方向为
`d = normalize(p - dot(p,n)*n)`，接近点为 `p - approach_distance*d`。
这取的是基座原点到接触点在叶面内的投影方向，以保留平行叶面插入的要求，
并不是强行沿穿过原点的三维直线刺入叶面。接触点和接近点的位置规则未变。
姿态容差仍可使最终夹指朝向相对名义方向偏转至配置阈值；它不会改变直线路径。
当前位姿到接近点采用原关节空间运动，不保证这段也沿基座径向直线。


## 基座法向投影与姿态容差（仿真适配层）

`config/pipeline.yaml` 的 `leaf_control.ros__parameters` 新增：

```yaml
project_normal_to_base_yz: true
orientation_tolerance_deg: 15.0
orientation_search_step_deg: 5.0
```

法向先通过原 TF 转入 `wx250s/base_link`，再按 `[0, ny, nz]` 归一化；不修改
相机侧的 PCA 输出。YZ 投影长度小于 1e-6 时拒绝该叶片并报告
`normal_projection_degenerate`。保留原最近基座接触点，按投影后的平面重新计算
接近方向与位置。投影可能改变实际夹持平面，这是有意引入的几何近似。

姿态搜索先试投影后的名义姿态，再按 5°、10°、15° 绕工具三轴分别正负旋转，
选取首个可行姿态。容差指相对名义姿态的 SO(3) 总旋转角；0° 关闭邻近姿态搜索。
保留原 IK 的 0.001 rad 数值残差，FK 位置误差必须不超过 1 mm。原关节限位、
速度检查、碰撞距离和物理到位阈值不变。搜索角度配置允许 0–45°，最多 9 层。

一个候选必须通过接近位姿、整段直线插入以及退出 IK 检查，三段共用同一选中朝向；
执行端直接使用选中朝向，不会每个路点另选姿态。姿态放宽不会改变本次搜索的两个
位置目标，碰撞筛选仍使用投影后的位置。多目标间仍在执行前重新计算当前起点的
运动路径。原碰撞算法只检查 EE 点云走廊，并非整臂或旋转夹爪网格碰撞。

`control_diagnostics.json` / 控制日志中包含：

- `normal_base_original`、`normal_base_used`：投影前后法向。
- `orientation_matrix`、`selected_orientation_matrix`：名义与选中姿态。
- `ik[].attempts`：采样角度、姿态、失败阶段及 FK 位置/角度误差。
- `orientation_error_deg`：选中姿态相对名义姿态的角度。

每片叶子的搜索最多使用 20 秒墙钟时间；采样失败不代表连续角度范围内完全无解。
已验证：55 项回归测试及 20 项 Isaac Python 离线测试通过（各排除历史源码/配置快照与当前真机工程的
一致性检查）。独立 GPU 仿真通过新 RPC 回放投影后目标的接近、插入、夹持和退出；
记录见 `artifacts/validation/pose_check_result.json`。该目标在投影后以 0° 偏转通过，
邻近角度搜索由回归测试覆盖；尚未宣称完成新一轮 YOLO→ROS 自动检测测量。

新配置需要重启模拟器和 pipeline（新增 `resolve_leaf` 仿真 RPC）。原真机工程与
`reference/` 保持不变；上述投影和姿态搜索是本项目明确新增的仿真规划行为。

## 2026-10-13：已选中目标，但张开夹爪超时

本次日志中 Leaf 0 的接近/接触 IK 与碰撞检查均通过，失败发生在第一个
`open_gripper`。旧 USD 将右手指导入为没有独立 drive 的 PhysX mimic 从动关节，
实测约 -0.039717 m，无法满足张开目标 -0.037 m 的 0.001 m 到位要求。

修复仅针对仿真导入：`assets/wx250s.import.urdf` 移除 mimic 标签，关闭 importer
的 mimic 解析，让左右指接受现有控制器的对称位置指令。保留原 URDF、手指轴向、
行程和到位阈值。重启模拟器时会自动重新生成 USD；运行中的旧实例必须重启。

超时现在输出动作名及未到位关节的目标、实测、误差和单位。执行失败也保留
`planned_targets` / `selected_count`，避免已选 1 个目标却报告 planned=0。
数量不足但全部候选已选中时标为 `candidates_exhausted`，不再误报碰撞阻挡。

独立无界面 GPU 仿真已回放该日志目标：张开→接近→笛卡尔插入→闭合→等待
3 秒→张开→笛卡尔退出全部成功。记录见
`artifacts/validation/gripper_check_result.json` 和 `gripper_check_sim.log`。
这是记录目标的运动回放，不代表重新运行了 YOLO→ROS 全流程。

新增回归测试覆盖双指导入限位、目标不足的碰撞报告和执行失败的已规划数量。
测试使用独立 ROS 域 179（可通过 `WIDOWX_TEST_ROS_DOMAIN_ID` 修改），避免被运行中
的仿真域 77 干扰。当前配置一致性测试因上述用户阈值调整而失败；保持配置不变，
排除此项后运行 `./scripts/test.sh -k 'not test_reference_is_exact_copy'` 和
`./scripts/test_sim_math.sh -k 'not test_reference_is_exact_copy'` 验证其余功能，
分别 47 项和 14 项通过（各排除 1 项配置一致性测试）。

## 1. 与真机一致的部分

| 项目 | 实现 |
|---|---|
| 机械臂 | 本机 Interbotix `wx250s.urdf.xacro` 展开；原几何、惯量、轴向、关节范围、速度/力矩上限 |
| EE 坐标 | `wx250s/base_link` → `wx250s/ee_gripper_link`；随机关节姿态下 URDF FK 与本机 SDK FK 对照通过 |
| 检测 | 原 `perception_node.py`，同一 `best.pt`、CPU 推理、640 输入、IoU 0.5、只取 leaf；当前用户配置置信度 0.3 |
| 深度/PCA | 原算法；0.12–1.5 m、最少 80 点、步长 2；保持原法向选择方法 |
| 目标 | 原最近基座点规则、接近距离 0.08 m、避障半径 0.035 m、目标数 3；新增基座 YZ 法向投影与有界姿态搜索 |
| 运动学 | 本机 Modern Robotics 库与 Interbotix WX250S `M/Slist`；IK 容差 0.001、三组 SDK 初始猜测 |
| 时序 | 移动 2 s、加速 0.3 s、夹持等待 3 s；笛卡尔路径 0.05 s 间隔 |
| 流程 | 原 workflow 与 control；张开→接近→平移进入→合拢→等待→张开→退出→下一片，**不额外回 home/sleep** |
| 消息 | 同名 RGB-D、CameraInfo、检测消息、检测/测量服务与结束状态 |

`config/pipeline.yaml` 最初复制自真机；当前用户已将正式/预览置信度改为 0.3/0.1（真机原配置为 0.2/0.05），其余原有参数沿用原配置；新增的仿真姿态参数见上节。仿真启动仅通过 ROS 参数覆盖 `use_sim_time=true` 和 `dry_run=false`（执行的是仿真后端）；`--dry-run` 保持原 `dry_run=true`。`--manual` 只关闭自动开始，便于检测调试。

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

### 初始关节读数超出 URDF 限位

真机编码器采样可能超出 URDF 的软件限位。例如本次肩关节读数为
`-1.952757597 rad`，而 URDF 下限为 `-1.884955592 rad`；手指开口读数为
`±0.037492223 m`，也超过了 URDF 的 `±0.037 m` 行程。

`config/scene.yaml` 的 `initial_joint_policy: clamp` **仅将仿真初值**投影到合法
范围，并向内保留微小数值余量，避免 float32/USD 转换造成边界越界。
`hardware.yaml` 的原始读数和 URDF 限位不会修改，也不会向真机发出指令。
所以发生投影时，仿真启动姿态与这次真机采样并非完全相同。

每项调整打印为 `initial_joint_adjusted`，包含关节名、原始值、仿真值、限位、
单位和原因；成功启动后，运行目录 `ready.json` 中还保存
`simulated_initial_joints` 和 `initial_joint_adjustments`，可查看实际使用的初值。

若要求实测启动姿态不允许越界投影，将策略设为 `strict`，并在真机处于 URDF
合法范围内时重新执行 `./scripts/capture_real.sh`。严格模式仍对恰好位于边界的
合法值执行记录在案的微小数值内缩，避免 PhysX 浮点误差；真实越界会直接报错。

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

### 逐阶段诊断与目标位置

新增 **WidowX | Planning diagnostics** 滚动窗口展示最新的正式检测与控制报告。
两个报告分别包含 `sequence`，应按相同序号关联；刷新低阈值预览不会覆盖正式诊断。

运行目录（默认 `/tmp/widowx_codex`）保存：

- `perception_diagnostics.json`：当前检测阈值、逐实例置信度、类别、mask 是否存在、深度/PCA 接受或拒绝原因。
- `control_diagnostics.json`：输入有效叶片数、目标位置/姿态、逐位姿 IK 结果、碰撞尝试过程、执行状态与最终原因。
- 对应的 `*_diagnostics.jsonl`：保留每次请求及执行阶段的历史记录。

终端查看完整报告：

```bash
/usr/bin/python3 -m json.tool /tmp/widowx_codex/perception_diagnostics.json
/usr/bin/python3 -m json.tool /tmp/widowx_codex/control_diagnostics.json
```

日志分别位于 `artifacts/logs/widowx_codex.ros_perception.log` 和
`artifacts/logs/widowx_codex.ros_control.log`，搜索 `PERCEPTION_DIAGNOSTICS` / `CONTROL_DIAGNOSTICS`。
原真机源文件未改写；检测适配器读取原算法生成的逐实例原因，不进行第二次推理。

`targets` 中位置单位为米、参考系为报告的 `frame`（默认 `wx250s/base_link`）：
`approach_position_m` 是接近点，`contact_position_m` 是测量点，`orientation_matrix`
为工具在基座系下的旋转矩阵。IK 检查完整位姿，位置看似接近也可能因朝向/限位不满足而失败；
`reachable=false` 表示本次 SDK 算法未找到可接受解，不是数学上证明该位姿绝对无解。

`collision.attempts` 按贪心筛选轮次记录实际尝试：
`current_to_approach` 会检查目标叶片自身，`approach_to_contact` 排除目标叶片。
`nearest.obstacle_leaf_id` 表明阻挡来自哪片叶，`min_distance_m <= radius_m` 即冲突，
同时输出障碍点和对应路径最近点。第一段失败时第二段为 null（原算法短路，未检查）。
IK 淘汰的候选不会进入碰撞列表；达到目标数而未尝试的候选也不会被误记为碰撞失败。
这是原 EE 点云走廊预筛选，不是整臂 mesh 碰撞或连续关节轨迹碰撞验证。

### 有预览框但机械臂不动

先看面板的 **Production detection / valid 3D leaves**，不是只数 YOLO 预览框。
`Refresh YOLO preview` 使用 0.05 调试阈值，且不发布控制目标；正式流程使用
0.2 阈值，还要求有效深度/PCA，并经过 IK 和路径筛选。预览框数量不等于可测量数量。

本次日志中的正式检测是 `sequence=1, model_predictions=0, valid_leaf_instances=0`；
随后刷新的低阈值预览不会自动重启已结束的 workflow。空目标时仿真适配层现已跳过
夹爪操作，返回 0 个测量及目标不足状态，不再由无关张爪动作造成超时报错。

要对当前画面重新进行正式检测和测量，在 ROS 域 77 的终端调用：

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1
ros2 service call /workflow/start std_srvs/srv/Trigger '{}'
```

只有有效且通过规划的目标才会引起运动。该命令会请求仿真运动；单纯查看掩码请继续
使用预览按钮。不要将预览阈值直接当作正式阈值，低置信度预览可能包含机械臂误检。

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

## SPAD 侧向对准模式（仿真）

使用独立装配目录 `../widowx_spad_assembly/wx250s_with_spad.urdf` 中的修薄 SPAD。
原裸夹爪模式仍是默认模式；真机代码没有改变。

终端 1：

```bash
cd /home/user/workspace/lwx/widowx_codex
./scripts/run_sim.sh --spad --run-dir /tmp/widowx_spad
```

终端 2（避免同时启动另一个域 77 的旧仿真管线）：

```bash
cd /home/user/workspace/lwx/widowx_codex
WIDOWX_RUN_DIR=/tmp/widowx_spad ./scripts/run_pipeline.sh
```

启动器从 `ready.json` 自动读取 SPAD 模式及装配 URDF。摄像头仍使用默认
`config/hardware.yaml`，需要其他视角时显式传 `--profile`。

变化：裸夹爪沿夹持器 +X 接近；SPAD 机身朝夹持器 +Z，沿夹持器侧向 −Z（仪器长轴 +X）接近。
+Y 仍对齐处理后的叶片法向。目标位置为 `spad_measurement_tip`，不再是裸夹爪中心。
IK 在内部用 `T_base_ee = T_base_measurement @ inverse(T_ee_measurement)` 转换目标，
包括姿态容差搜索和每一个插入/退出路点；TCP 偏移从装配 URDF 读取。

当前只做 **接近→侧向伸入→停留→退出**。两指保持 ±0.0369 m、SPAD 夹口保持
0.1 rad，禁止调用裸夹爪开合接口。尚未实现压合时测量点移动补偿与接触传动。
因此 `control_diagnostics.json` 中 `aligned_targets` 表示对准完成数，
`measured=0`、`measurement_performed=false` 表示没有进行测量。

现有避障仅为测量 TCP 的点云走廊筛选，**不是整个 SPAD 外壳或机械臂的碰撞检测**。
SPAD 模式关闭机器人重力，仅验证几何与 IK；原估计 SPAD 质量会产生明显静态跟踪误差，
尚未验证负载和重力补偿。本模式不代表已完成带仪器的安全避障或真实测量。
新测试在 `tests/test_spad_tool.py`，独立 GPU 执行验证记录在 `artifacts/spad_validation/`。
