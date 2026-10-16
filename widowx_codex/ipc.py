"""Isaac 与 ROS 的本地通信：Unix socket 请求/应答、原子 NPZ 帧文件。

网络线程只收发数据并排队，不能直接调用 PhysX；仿真主线程负责执行。

Local-only RPC; Isaac's main thread exclusively owns simulation objects."""
import json
import queue
import socket
import socketserver
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

@dataclass
class Request:
    method: str
    params: dict
    deadline: float
    done: threading.Event = field(default_factory=threading.Event)
    result: object = None
    error: str = ''
    def finish(self, result=None, error=''):
        self.result, self.error = result, error
        self.done.set()

class Server:
    def __init__(self, path, timeout=45):
        self.queue = queue.Queue(maxsize=32)
        owner = self
        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                try:
                    self.connection.settimeout(timeout+2)
                    raw = self.rfile.readline(65537)
                    if len(raw)>65536: raise ValueError('Request too large')
                    msg = json.loads(raw)
                    allowed = {'current_position','pose_reachable','resolve_leaf','move_pose','move_cartesian_to_pose','open_gripper','close_gripper','shutdown'}
                    if msg['method'] not in allowed: raise ValueError('Unsupported method')
                    # 只排队并等待；PhysX 由 simulator 的主循环推进。
                    req = Request(msg['method'], msg.get('params',{}),time.monotonic()+timeout)
                    owner.queue.put_nowait(req)
                    if not req.done.wait(timeout): raise TimeoutError('Simulation did not complete command')
                    response = {'result':req.result,'error':req.error}
                except Exception as exc:
                    response = {'error':str(exc)}
                self.wfile.write((json.dumps(response, allow_nan=False)+'\n').encode())
        self.path = Path(path)
        if self.path.exists(): raise RuntimeError(f'Socket already exists: {path}; use a fresh run directory')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.server = socketserver.ThreadingUnixStreamServer(str(path),Handler)
        self.server.daemon_threads = True
        self.path.chmod(0o600)
        self.thread = threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
    def close(self):
        self.server.shutdown(); self.server.server_close(); self.path.unlink(missing_ok=True)

class Client:
    def __init__(self,path,timeout=50): self.path,self.timeout=str(path),timeout
    def call(self,method,**params):
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as sock:
            sock.settimeout(self.timeout); sock.connect(self.path)
            # 单行 JSON 作为消息边界；禁用 NaN 避免跨进程传递非法位姿。
            sock.sendall((json.dumps({'method':method,'params':params},allow_nan=False)+'\n').encode())
            with sock.makefile('rb') as stream: result=json.loads(stream.readline(65537))
        if result.get('error'): raise RuntimeError(result['error'])
        return result.get('result')

class SimBackend:
    """Duck-typed replacement of InterbotixArmBackend; cannot command hardware."""
    def __init__(self,path): self.client=Client(path)
    def current_position(self): return np.array(self.client.call('current_position'))
    def pose_reachable(self,pose): return bool(self.client.call('pose_reachable',pose=np.asarray(pose).tolist()))
    def resolve_leaf(self, approach, contact, tolerance_deg, step_deg):
        return self.client.call('resolve_leaf', approach=np.asarray(approach).tolist(),
                                contact=np.asarray(contact).tolist(), tolerance_deg=tolerance_deg,
                                step_deg=step_deg)
    def move_pose(self,pose): self.client.call('move_pose',pose=np.asarray(pose).tolist())
    def move_cartesian_to_pose(self,pose): self.client.call('move_cartesian_to_pose',pose=np.asarray(pose).tolist())
    def open_gripper(self): self.client.call('open_gripper')
    def close_gripper(self): self.client.call('close_gripper')
    def shutdown(self): pass  # Node destruction never sends a robot motion.

def save_frame(path, **data):
    """Atomic replacement prevents RGB/depth/joint snapshots from mixing frames."""
    path=Path(path); temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as stream: np.savez_compressed(stream,**data)
    temporary.replace(path)
