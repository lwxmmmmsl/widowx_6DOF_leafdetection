"""仿真主入口：创建机器人、植物和相机，并在唯一的物理线程中执行 EE 指令。

配置读取 → Isaac 启动 → 资产导入 → 初值检查 → RGB-D/指令循环。
ROS 在独立 Python 3.10 进程运行，通过 ipc.py 与本进程交换数据。

Isaac Lab scene, calibrated RGB-D transport and asynchronous EE execution.

Run via scripts/run_sim.sh: AppLauncher must precede all Isaac/Omni imports.
No ROS imports, real robot driver, or hardware command topics exist here.
"""
import argparse
import json
import queue
import time
from pathlib import Path
import numpy as np
from .common import ROOT, ARM, event, profile, read_yaml
from .camera import CameraMapping
from .kinematics import Model
from .ipc import Server, save_frame

def main():
    from isaaclab.app import AppLauncher
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spad',action='store_true',help='SPAD side-entry alignment only; fixed finger opening')
    parser.add_argument('--profile',type=Path,default=ROOT/'config/hardware.yaml')
    parser.add_argument('--scene',type=Path,default=ROOT/'config/scene.yaml')
    parser.add_argument('--run-dir',type=Path,default=Path('/tmp/widowx_codex'))
    parser.add_argument('--debug-ui', action='store_true', help='Create camera/YOLO panel even in headless smoke tests')
    parser.add_argument('--no-debug-ui', action='store_true', help='Disable embedded camera/YOLO panel')
    parser.add_argument('--steps',type=int,default=0,help='0 runs until GUI/window closes')
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    if not args.profile.exists():
        parser.error(f'Missing {args.profile}. Run scripts/capture_real.sh, or explicitly use --profile config/demo_camera.yaml for an uncalibrated demo.')
    cfg=profile(args.profile); scene=read_yaml(args.scene)
    if not args.enable_cameras: parser.error('RGB-D pipeline requires --enable_cameras')
    import torch
    if not torch.cuda.is_available():
        parser.error('No accessible NVIDIA CUDA GPU. Isaac RGB-D requires a working NVIDIA driver/RTX GPU even with --headless; see artifacts/validation/isaac_startup.log.')
    args.run_dir.mkdir(parents=True,exist_ok=True)
    # An exclusive run lock also prevents stale frame readers across instances.
    # 同一 run-dir 只允许一个仿真写入者，避免旧帧/旧 socket 混入新会话。
    import fcntl
    lock=(args.run_dir/'sim.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name in ('latest.npz','ready.json','rpc.sock','yolo_debug.npz','panel_status.json','preview_request.json','last_error.txt','perception_diagnostics.json','control_diagnostics.json'): (args.run_dir/name).unlink(missing_ok=True)
    # 必须先启动 Kit，再导入依赖 omni/PhysX 的类。
    app=AppLauncher(args).app
    server=None
    panel=None
    try:
        import torch
        import isaaclab.sim as sim
        from isaaclab.sim.converters import UrdfConverter,UrdfConverterCfg,MeshConverter,MeshConverterCfg
        from isaaclab.assets import Articulation,ArticulationCfg
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.sensors import Camera,CameraCfg
        from scipy.spatial.transform import Rotation
        from pxr import UsdGeom
        pipeline=read_yaml(ROOT/'config/pipeline.yaml')['leaf_control']['ros__parameters']
        from .spad_tool import SpadModel, ASSEMBLY, FINGER, JAW
        model_type=SpadModel if args.spad else Model
        model=model_type(moving_time=pipeline['moving_time_s'],accel_time=pipeline['accel_time_s'])
        context=sim.SimulationContext(sim.SimulationCfg(dt=scene['physics_dt'],device=args.device))
        from .import_config import robot_import_config
        robot_source=ROOT/'assets/wx250s.import.urdf'
        if args.spad:
            import xml.etree.ElementTree as ET
            tree=ET.parse(ASSEMBLY)
            for mesh in tree.findall('.//mesh'):
                mesh.set('filename',str((ASSEMBLY.parent/mesh.get('filename')).resolve()))
            robot_source=args.run_dir/'spad.import.urdf'
            tree.write(robot_source,encoding='utf-8',xml_declaration=True)
        converter=UrdfConverter(robot_import_config(
            UrdfConverterCfg, robot_source,
            ROOT/('artifacts/spad_robot_usd' if args.spad else 'artifacts/robot_usd'),
        ))
        # PhysX 在 reset 时先校验默认位置；因此必须在创建 Articulation 前
        # 处理编码器越界，不能等到 write_joint_state_to_sim 时才裁剪。
        from .initial_state import prepare_initial_joints
        initial, initial_adjustments = prepare_initial_joints(
            cfg['initial_joints'], model.limits,
            policy=scene.get('initial_joint_policy', 'clamp'),
        )
        if args.spad:initial.update(left_finger=FINGER,right_finger=-FINGER,spad_jaw_hinge=JAW)
        for adjustment in initial_adjustments:
            event('initial_joint_adjusted', **adjustment)
        # URDF 提供力矩/速度上限；PD 增益属于独立的仿真参数。
        actuators={}
        for name,limits in model.limits.items():
            finger=name in ('left_finger','right_finger')
            actuators[name]=ImplicitActuatorCfg(joint_names_expr=[name],
                stiffness=scene['finger_stiffness' if finger else 'arm_stiffness'],
                damping=scene['finger_damping' if finger else 'arm_damping'],
                effort_limit_sim=limits['effort'],velocity_limit_sim=limits['velocity'])
        robot=Articulation(ArticulationCfg(prim_path='/World/Robot',
            # 对准模式只验证几何与 IK；SPAD 质量未实测，不用于负载验收。
            spawn=sim.UsdFileCfg(usd_path=converter.usd_path,
                rigid_props=sim.RigidBodyPropertiesCfg(disable_gravity=True) if args.spad else None),
            init_state=ArticulationCfg.InitialStateCfg(joint_pos=initial),actuators=actuators))
        ground=sim.GroundPlaneCfg(); ground.func('/World/Ground',ground)
        light=sim.DomeLightCfg(intensity=1800); light.func('/World/Light',light)
        plant=scene['plant']; plant_path=Path(plant['asset'])
        if not plant_path.is_absolute(): plant_path=ROOT/plant_path
        if plant_path.suffix.lower() == '.glb':
            from .plant import convert_glb
            usd=convert_glb(plant_path,ROOT/'artifacts/plant_usd')
        elif plant_path.suffix.lower() in ('.usd','.usda','.usdc'): usd=str(plant_path)
        else:
            usd=MeshConverter(MeshConverterCfg(asset_path=str(plant_path),usd_dir=str(ROOT/'artifacts/plant_usd'),
                usd_file_name='tomato.usd',make_instanceable=False)).usd_path
        spawn=sim.UsdFileCfg(usd_path=usd,scale=tuple(plant['scale']))
        spawn.func('/World/Plant',spawn,translation=tuple(plant['position_m']),orientation=tuple(plant['quaternion_wxyz']))
        # Leaves are thin two-sided visual surfaces. No fictitious soft-body physics.
        for prim in context.stage.Traverse():
            if str(prim.GetPath()).startswith('/World/Plant') and prim.IsA(UsdGeom.Mesh):
                UsdGeom.Mesh(prim).CreateDoubleSidedAttr(True)
        # 在真实 optical 位姿渲染，再按 K/D 重映射，避免误用 camera_base 位姿。
        cmap=CameraMapping(cfg['camera']); c=cfg['camera']
        camera=Camera(CameraCfg(prim_path='/World/RgbCamera',height=c['height'],width=c['width'],
            update_period=0,data_types=['rgb','distance_to_image_plane'],
            spawn=sim.PinholeCameraCfg.from_intrinsic_matrix(cmap.render_k.flatten().tolist(),c['width'],c['height'],clipping_range=(.01,10))))
        # reset 会触发关节限位校验和传感器初始化；此时初值必须已经合法。
        context.reset(); robot.update(scene['physics_dt']); camera.update(scene['physics_dt'])
        transform=np.array(c['T_base_optical']); quat=Rotation.from_matrix(transform[:3,:3]).as_quat()[[3,0,1,2]]
        camera.set_world_poses(torch.tensor(transform[None,:3,3],device=context.device,dtype=torch.float32),
            torch.tensor(quat[None],device=context.device,dtype=torch.float32),convention='ros')
        cmap.set_render_k(camera.data.intrinsic_matrices[0].cpu().numpy())
        names=robot.joint_names; indices=[names.index(n) for n in ARM]
        target=robot.data.default_joint_pos.clone()
        q_command=target[0,indices].cpu().numpy().copy(); command_pose=model.fk(q_command)
        if np.any(q_command<model.lower) or np.any(q_command>model.upper): raise ValueError('Initial joints outside URDF limits')
        robot.write_joint_state_to_sim(target,torch.zeros_like(target))
        context.set_camera_view(eye=[1.0,-1.0,.8],target=[.25,0,.25])
        server=Server(args.run_dir/'rpc.sock')
        # 同时保留实测 profile 与实际仿真初值，方便追溯投影造成的姿态差异。
        metadata={'tool_mode':'spad_alignment' if args.spad else 'bare_gripper',
                  'robot_description_file':str(robot_source),
                  'gravity_disabled_for_alignment':bool(args.spad),
                  'task_tcp':'spad_measurement_tip' if args.spad else 'ee_gripper_link',
                  'profile':cfg,'joint_names':names,'pipeline_file':str(ROOT/'config/pipeline.yaml'),
                  'source':'isaaclab','plant':plant,'simulation_only':True,
                  'simulated_initial_joints':initial,
                  'initial_joint_adjustments':initial_adjustments}
        (args.run_dir/'ready.json').write_text(json.dumps(metadata,indent=2))
        event('simulation_ready',run_dir=str(args.run_dir),camera_source=cfg.get('source'),plant=str(plant_path))
        if not args.no_debug_ui and (not args.headless or args.debug_ui):
            from .debug_panel import DebugPanel
            panel = DebugPanel(args.run_dir)
        event('pipeline_required', message='Start scripts/run_pipeline.sh to detect and move; scene alone holds the robot still.')
        pending=None; motion=None; steps=0; simtime=0.; next_frame=0.; sequence=0
        wall_start=time.monotonic()
        while app.is_running() and (not args.steps or steps<args.steps):
            q_measured=robot.data.joint_pos[0,indices].cpu().numpy()
            # 每次只执行一条运动；队列线程不会并发写入机器人状态。
            if pending is None:
                try: req=server.queue.get_nowait()
                except queue.Empty: req=None
                if req is not None:
                    try:
                        if time.monotonic()>req.deadline: raise TimeoutError('Request expired before execution')
                        method=req.method
                        if method=='current_position': req.finish(model.fk(q_measured)[:3,3].tolist())
                        elif method=='pose_reachable': req.finish(model.ik(req.params['pose'],q_command) is not None)
                        elif method=='resolve_leaf':
                            from .pose_tolerance import resolve_leaf_pose
                            req.finish(resolve_leaf_pose(model, q_command, **req.params,
                                                         deadline=min(req.deadline-2, time.monotonic()+20)))
                        elif method=='shutdown': req.finish(True)
                        elif method in ('open_gripper','close_gripper'):
                            if args.spad:raise RuntimeError('SPAD alignment locks finger opening; contact actuation is not implemented')
                            opening=method=='open_gripper'
                            for name in ('left_finger','right_finger'):
                                lim=model.limits[name]
                                value=lim['upper' if (opening == (name=='left_finger')) else 'lower']
                                target[0,names.index(name)]=value
                            pending=req; motion={'start':simtime,'gripper':True}
                        else:
                            cart=method=='move_cartesian_to_pose'
                            # 先验证整段路径，任何一个路点不可达都不开始运动。
                            points=model.plan(q_command,np.array(req.params['pose']),cart,command_pose)
                            pending=req; motion={'start':simtime,'points':points,'cartesian':cart,'pose':np.array(req.params['pose']),'gripper':False}
                            event('motion_started',method=method,target=req.params['pose'])
                    except Exception as exc: req.finish(error=str(exc))
            if pending is not None:
                elapsed=simtime-motion['start']
                if not motion['gripper']:
                    target[0,indices]=torch.as_tensor(model.interpolate(motion['points'],elapsed,motion['cartesian']),device=context.device,dtype=target.dtype)
                duration=.6 if motion['gripper'] else model.moving_time
                # deadline 是墙钟超时，elapsed 是仿真时间；两者都需要限制。
                if time.monotonic()>pending.deadline or elapsed>duration+scene['motion_settle_timeout_s']:
                    # 报告每个未到位关节的目标/实测/误差，不用混合单位的单一最大误差。
                    actual = robot.data.joint_pos[0].cpu().numpy()
                    wanted = target[0].cpu().numpy()
                    errors = [{'joint':name, 'target':float(wanted[i]), 'actual':float(actual[i]),
                               'error':float(abs(actual[i]-wanted[i])),
                               'unit':'m' if name in ('left_finger','right_finger') else 'rad'}
                              for i,name in enumerate(names)
                              if abs(actual[i]-wanted[i]) > (scene['finger_tolerance_m'] if name in ('left_finger','right_finger') else scene['joint_tolerance_rad'])]
                    detail = json.dumps(errors, ensure_ascii=False)
                    event('motion_timeout', operation=pending.method, joints=errors)
                    target=robot.data.joint_pos.clone(); q_command=target[0,indices].cpu().numpy().copy(); command_pose=model.fk(q_command)
                    pending.finish(error=f'{pending.method} timeout; holding measured joints; errors={detail}'); pending=None
                elif elapsed>=duration:
                    error=float(torch.max(torch.abs(robot.data.joint_pos-target)).item())
                    # 角度与手指位移不能用同一个阈值：分别检查 rad 和 m。
                    arm_error=float(torch.max(torch.abs(robot.data.joint_pos[0,indices]-target[0,indices])).item())
                    finger_ids=[names.index(n) for n in ('left_finger','right_finger')]
                    finger_error=float(torch.max(torch.abs(robot.data.joint_pos[0,finger_ids]-target[0,finger_ids])).item())
                    if arm_error<scene['joint_tolerance_rad'] and finger_error<scene['finger_tolerance_m']:
                        if not motion['gripper']: q_command=motion['points'][-1].copy(); command_pose=motion['pose']
                        pending.finish(True); pending=None; event('motion_finished',joint_error=error)
            # 写控制目标 → 物理步进 → 更新测量缓存；所有 PhysX 操作留在本线程。
            robot.set_joint_position_target(target); robot.write_data_to_sim()
            render=simtime>=next_frame
            context.step(render=render); robot.update(scene['physics_dt'])
            simtime+=scene['physics_dt']; steps+=1
            if render:
                camera.update(1/scene['camera_fps'],force_recompute=True)
                rgba=camera.data.output['rgb'][0].cpu().numpy()
                depth=camera.data.output['distance_to_image_plane'][0].cpu().numpy()
                bgra,depth=cmap.convert(rgba,depth)
                # 原子替换完整 RGB-D/关节快照，读端不会读到半写入的文件。
                save_frame(args.run_dir/'latest.npz',color=bgra,depth=depth,sequence=sequence,
                    stamp=simtime,joints=robot.data.joint_pos[0].cpu().numpy(),k=np.array(c['k']),d=np.array(c.get('d',[])))
                if panel is not None:
                    panel.update(bgra, simtime, pending.method if pending else None)
                sequence+=1; next_frame+=1/scene['camera_fps']
            # Real pipeline uses time.sleep(hold_s); do not let simulated time outrun wall time.
            remaining=wall_start+simtime-time.monotonic()
            if remaining>0: time.sleep(min(remaining,scene['physics_dt']))
        if pending is not None: pending.finish(error='Simulation stopped')
    except KeyboardInterrupt:
        event('simulation_interrupted', message='Stopped by user')
    except BaseException:
        # Kit shutdown can terminate the interpreter before Python prints a
        # pending traceback. Persist it BEFORE calling app.close().
        import sys
        import traceback
        details = traceback.format_exc()
        failure_path = args.run_dir/'last_error.txt'
        failure_path.write_text(details)
        print(details, file=sys.stderr, flush=True)
        event('simulation_failed', traceback_file=str(failure_path))
        raise
    finally:
        if server is not None: server.close()
        (args.run_dir/'ready.json').unlink(missing_ok=True)
        if panel is not None:
            panel.close()
        # IPC 已停止、NPZ 已同步落盘，没有后台数据集写入任务。
        # 本程序独占 Kit 进程，使用官方立即关闭接口，避免相机工作流在
        # Replicator.stop 中阻塞；不适用于嵌入其他 Kit 应用的用法。
        lock.close()
        app.close(skip_cleanup=True)

if __name__=='__main__': main()
