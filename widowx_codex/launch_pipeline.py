"""启动并监控仿真 ROS 管线的各个子进程。

沿用原检测和 workflow 模块；替换相机/控制后端并统一使用 ROS 域 77。

Supervise original perception/workflow plus simulation bridge/control only."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import yaml
from .common import ROOT, event

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true',help='Keep original dry_run=true; do not move simulated robot')
    parser.add_argument('--manual',action='store_true',help='Disable workflow auto_start; call services manually')
    parser.add_argument('--perception-python',default='/home/user/anaconda3/envs/env_perception/bin/python')
    args=parser.parse_args()
    if os.environ.get('ROS_DOMAIN_ID')!='77': raise RuntimeError('Simulation pipeline requires isolated ROS_DOMAIN_ID=77')
    run_dir=Path(os.environ['WIDOWX_RUN_DIR'])
    if not (run_dir/'ready.json').exists() or not (run_dir/'rpc.sock').exists():
        raise RuntimeError('Start scripts/run_sim.sh and wait for simulation_ready first')
    metadata=json.loads((run_dir/'ready.json').read_text())
    spad=metadata.get('tool_mode')=='spad_alignment'
    logs=ROOT/'artifacts/logs'; logs.mkdir(parents=True,exist_ok=True)
    description=run_dir/'robot_state_publisher.yaml'
    description.write_text(yaml.safe_dump({'robot_state_publisher':{'ros__parameters':{'use_sim_time':True, 'robot_description':Path(metadata.get('robot_description_file',ROOT/'assets/wx250s.original.urdf')).read_text()}}}))
    base=['--ros-args','--params-file',str(ROOT/'config/pipeline.yaml'),'-p','use_sim_time:=true']
    # Python 3.10 运行 ROS，既有推理环境运行 YOLO；Isaac 3.11 不导入 rclpy。
    commands=[('/usr/bin/python3','widowx_codex.ros_bridge',[]),
              ('/usr/bin/python3','widowx_codex.ros_control',base+['-p',f'dry_run:={str(args.dry_run).lower()}','-p',f'spad_mode:={str(spad).lower()}']),
              (args.perception_python,'widowx_codex.ros_perception',base),
              ('/usr/bin/python3','wodowx_control.main_node',base+['-p',f'auto_start:={str(not args.manual).lower()}'])]
    children=[]; streams=[]
    def stop(*_): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:
        rsp_output=(logs/'robot_state_publisher.log').open('w'); streams.append(rsp_output)
        children.append(subprocess.Popen(['/opt/ros/humble/lib/robot_state_publisher/robot_state_publisher','--ros-args','--params-file',str(description),'-r','joint_states:=/wx250s/joint_states'],stdout=rsp_output,stderr=subprocess.STDOUT))
        for interpreter,module,argv in commands:
            output=(logs/(module+'.log')).open('w'); streams.append(output)
            child=subprocess.Popen([interpreter,'-m',module,*argv],stdout=output,stderr=subprocess.STDOUT)
            children.append(child); event('node_started',module=module,pid=child.pid,log=str(output.name))
        event('pipeline_running',ros_domain_id=77,dry_run=args.dry_run,run_dir=str(run_dir))
        while True:
            for child in children:
                if child.poll() is not None: raise RuntimeError(f'Node {child.args} exited {child.returncode}; inspect {logs}')
            if not (run_dir/'ready.json').exists(): raise RuntimeError('Simulator stopped')
            time.sleep(.25)
    except KeyboardInterrupt: pass
    # 任一子进程退出或用户中断后，回收本次启动的全部 ROS 子进程。
    finally:
        for child in children:
            if child.poll() is None: child.send_signal(signal.SIGINT)
        for child in children:
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
        for stream in streams: stream.close()

if __name__=='__main__': main()
