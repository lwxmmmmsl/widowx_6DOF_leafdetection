"""独立 GPU 对准回归；合成侧向目标，不代表 YOLO 抓取，不连接真机。"""
import sys,subprocess,time,os,signal,json,tempfile
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from widowx_codex.ipc import Client
from widowx_codex.spad_tool import SpadModel
from widowx_codex.common import ARM
run=Path(tempfile.mkdtemp(prefix='wx_spad_motion_'));out=root/'artifacts/spad_validation';out.mkdir(exist_ok=True)
f=(out/'simulation.log').open('w');p=subprocess.Popen([str(root/'scripts/run_sim.sh'),'--spad','--headless','--profile',str(root/'config/demo_camera.yaml'),'--run-dir',str(run)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
r={'run_dir':str(run),'kind':'synthetic_side_entry_not_yolo'}
try:
 end=time.monotonic()+90
 while not (run/'latest.npz').exists():
  if p.poll() is not None:raise RuntimeError('Simulation exited')
  if time.monotonic()>end:raise TimeoutError('startup')
  time.sleep(.2)
 info=json.loads((run/'ready.json').read_text());assert info['tool_mode']=='spad_alignment'
 with np.load(run/'latest.npz') as frame:q=frame['joints'][[info['joint_names'].index(n) for n in ARM]]
 m=SpadModel();a=m.fk(q);c=a.copy();c[:3,3]-=.01*a[:3,2]
 client=Client(run/'rpc.sock');client.call('move_pose',pose=a.tolist());client.call('move_cartesian_to_pose',pose=c.tolist())
 actual=np.array(client.call('current_position'));r['measurement_tcp_error_m']=float(np.linalg.norm(actual-c[:3,3]));assert r['measurement_tcp_error_m']<.005
 client.call('move_cartesian_to_pose',pose=a.tolist())
 try:client.call('close_gripper')
 except RuntimeError as exc:r['closing_rejected']=str(exc)
 else:raise AssertionError('SPAD close must be blocked')
 r['success']=True
except Exception as exc:
 r.update(success=False,error=str(exc))
 raise
finally:
 (out/'result.json').write_text(json.dumps(r,indent=2))
 if p.poll() is None:os.killpg(p.pid,signal.SIGINT)
 try:p.wait(timeout=10)
 except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
 f.close()
print(json.dumps(r),flush=True)
