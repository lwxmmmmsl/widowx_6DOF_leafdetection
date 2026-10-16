"""把理想针孔渲染图重映射为真机 CameraInfo 定义的像素坐标。

输出 BGRA8 彩色图和 16UC1 毫米深度，深度为光轴 Z 而非到相机的斜距。

Render pinhole images into the real Kinect registered RGB optical model."""
import cv2
import numpy as np

class CameraMapping:
    def __init__(self, camera):
        self.width, self.height = int(camera['width']), int(camera['height'])
        self.k = np.array(camera['k'], dtype=float).reshape(3,3)
        self.d = np.array(camera.get('d', []), dtype=float)
        x,y = np.meshgrid(np.arange(self.width), np.arange(self.height))
        pixels = np.stack([x,y], axis=-1).reshape(-1,1,2).astype(np.float64)
        # 对输出像素去畸变得到归一化光线，再查找其在理想渲染图中的位置。
        rays = cv2.undistortPoints(pixels, self.k, self.d if self.d.size else None).reshape(self.height,self.width,2)
        # Fit every distorted sensor pixel inside a centered square-pixel render.
        extent = np.max(np.abs(rays), axis=(0,1))
        f = min((self.width-2)/2/max(extent[0],1e-8), (self.height-2)/2/max(extent[1],1e-8))*.98
        self.render_k = np.array([[f,0,(self.width-1)/2],[0,f,(self.height-1)/2],[0,0,1]])
        self.rays = rays
        self.set_render_k(self.render_k)

    def set_render_k(self, matrix):
        k = np.asarray(matrix).reshape(3,3)
        self.map_x = (self.rays[:,:,0]*k[0,0]+k[0,2]).astype(np.float32)
        self.map_y = (self.rays[:,:,1]*k[1,1]+k[1,2]).astype(np.float32)
        if self.map_x.min()<0 or self.map_x.max()>self.width-1 or self.map_y.min()<0 or self.map_y.max()>self.height-1:
            raise RuntimeError('Renderer intrinsic matrix does not cover real calibrated camera rays')

    def convert(self, rgba, axial_depth):
        rgb = cv2.remap(rgba[:,:,:3], self.map_x,self.map_y, cv2.INTER_LINEAR)
        bgra = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGRA)
        depth = cv2.remap(np.asarray(axial_depth).reshape(self.height,self.width), self.map_x,self.map_y,cv2.INTER_NEAREST)
        # 传感器消息使用毫米 uint16；无穷远、负深度及溢出值统一置零。
        valid = np.isfinite(depth) & (depth > 0) & (depth < 65.535)
        millimeters = np.zeros(depth.shape, np.uint16)
        millimeters[valid] = np.rint(depth[valid]*1000).astype(np.uint16)
        return bgra, millimeters
