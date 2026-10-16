"""Isaac 内嵌 RGB/YOLO 面板；仅在 AppLauncher 启动后创建。

显示真实相机渲染和原 perception 节点发布的分割图，不伪造模型输出。
刷新按钮仅请求 /perception/debug_detect，不发布可用于测量的叶片目标。
"""
import time
import json
from pathlib import Path
import cv2
import numpy as np
from .display import atomic_json, panel_status, read_json


class DebugPanel:
    def __init__(self, directory):
        import omni.ui as ui
        self.directory = Path(directory)
        self.debug_mtime = None
        self.last_error = ''
        self.window = ui.Window('WidowX | Camera & YOLO', width=1100, height=560)
        self.rgb_provider = ui.ByteImageProvider()
        self.yolo_provider = ui.ByteImageProvider()
        empty = np.zeros((360, 640, 4), np.uint8)
        empty[:, :, 3] = 255
        for provider in (self.rgb_provider, self.yolo_provider):
            self.upload(provider, empty)
        with self.window.frame:
            with ui.VStack(spacing=6):
                self.status = ui.Label('Waiting for camera...', height=105, word_wrap=True)
                ui.Button('Refresh YOLO preview (no robot motion)', height=30, clicked_fn=self.request_preview)
                with ui.HStack(spacing=8):
                    with ui.VStack():
                        ui.Label('Live calibrated RGB camera', height=22)
                        ui.ImageWithProvider(self.rgb_provider, fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
                    with ui.VStack():
                        self.yolo_label = ui.Label('YOLO: waiting for /perception/debug_image', height=22)
                        ui.ImageWithProvider(self.yolo_provider, fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
                self.motion = ui.Label('No command: robot holds current pose', height=24)

        # 单独滚动窗口展示完整逐阶段报告，避免在主面板挤成一行。
        self.details_window = ui.Window('WidowX | Planning diagnostics', width=850, height=620)
        with self.details_window.frame:
            with ui.ScrollingFrame():
                self.details_label = ui.Label('Waiting for production detection...', word_wrap=True)
        self.report_versions = None

    @staticmethod
    def upload(provider, rgba):
        """仅缩小面板显示，不更改相机消息或 YOLO 输入分辨率。"""
        h, w = rgba.shape[:2]
        scale = min(1., 640/w)
        preview = cv2.resize(rgba, (round(w*scale), round(h*scale))) if scale < 1 else rgba
        provider.set_bytes_data(preview.ravel().tolist(), [preview.shape[1], preview.shape[0]])

    def request_preview(self):
        # 由 ROS bridge 异步调用服务，UI/物理线程不等待 YOLO 推理。
        atomic_json(self.directory/'preview_request.json', {'id': time.time_ns()})

    def update(self, bgra, simtime, command):
        self.upload(self.rgb_provider, cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA))
        self.status.text = panel_status(self.directory)
        reports = {name:read_json(self.directory/f'{name}_diagnostics.json') for name in ('perception','control')}
        versions = tuple(reports[name].get('wall_time') for name in reports)
        if versions != self.report_versions:
            self.details_label.text = json.dumps(reports, indent=2, ensure_ascii=False)
            self.report_versions = versions
        self.motion.text = f'Simulation time: {simtime:.2f}s | Command: {command or "idle (waiting for control)"}'
        path = self.directory/'yolo_debug.npz'
        try:
            mtime = path.stat().st_mtime_ns
            if mtime != self.debug_mtime:
                with np.load(path, allow_pickle=False) as frame:
                    self.upload(self.yolo_provider, frame['rgba'])
                    self.debug_stamp = float(frame['stamp'])
                self.debug_mtime = mtime
            # 原节点会重发旧图，必须按图像时间戳显示年龄，不能假装实时推理。
            self.yolo_label.text = f'YOLO last inference image | age {max(0., simtime-self.debug_stamp):.1f}s'
        except FileNotFoundError:
            pass
        except (OSError, ValueError, KeyError) as exc:
            self.yolo_label.text = f'Debug image read error: {exc}'

    def close(self):
        self.window.destroy()
        self.details_window.destroy()
