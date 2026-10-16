"""只读调试：检查最新 RGB-D、关节与 EE 位置，可导出 PNG。

此入口不会调用检测/测量服务，也不会发送运动指令。

Inspect latest Isaac RGB-D/robot state without triggering measurement."""
import argparse
from pathlib import Path
import numpy as np
import cv2
from .ipc import Client
from .common import event

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=Path('/tmp/widowx_codex'))
    parser.add_argument('--save',type=Path,help='Optional directory for color.png and millimeter depth.png')
    args=parser.parse_args()
    with np.load(args.run_dir/'latest.npz',allow_pickle=False) as frame:
        valid=frame['depth'][frame['depth']>0]
        event('frame',sequence=int(frame['sequence']),stamp=float(frame['stamp']),color_shape=list(frame['color'].shape),
              valid_depth_pixels=int(valid.size),depth_m_range=[float(valid.min()/1000),float(valid.max()/1000)] if valid.size else [],
              joint_positions=frame['joints'].tolist(),ee_position=Client(args.run_dir/'rpc.sock').call('current_position'))
        if args.save:
            args.save.mkdir(parents=True,exist_ok=True)
            cv2.imwrite(str(args.save/'color.png'),frame['color']); cv2.imwrite(str(args.save/'depth_mm.png'),frame['depth'])

if __name__=='__main__': main()
