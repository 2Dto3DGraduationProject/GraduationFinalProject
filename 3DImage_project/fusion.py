import numpy as np
from numba import njit, prange
from skimage import measure
import torch

class TSDFVolume:
    """Volumetric TSDF Fusion of RGB-D Images."""

    def __init__(self, vol_bnds, voxel_size, use_gpu=False, margin=5):
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
            from pycuda.compiler import SourceModule
            FUSION_GPU_MODE = 1
            self.cuda = cuda
            print("[INFO] PyCUDA loaded - GPU mode")
        except Exception as err:
            print("[WARNING] PyCUDA not found - CPU mode")
            print(err)
            FUSION_GPU_MODE = 0

        vol_bnds = np.asarray(vol_bnds)
        assert vol_bnds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."
        vol_bnds = np.clip(vol_bnds, -10.0, 10.0)

        self._vol_bnds = vol_bnds
        self._voxel_size = float(voxel_size)
        self._trunc_margin = margin * self._voxel_size
        self._color_const = 256 * 256

        self._vol_dim = np.round((self._vol_bnds[:, 1] - self._vol_bnds[:, 0]) / self._voxel_size).copy(
            order='C').astype(int)
        self._vol_bnds[:, 1] = self._vol_bnds[:, 0] + self._vol_dim * self._voxel_size
        self._vol_origin = self._vol_bnds[:, 0].copy(order='C').astype(np.float32)

        self._tsdf_vol_cpu = np.ones(self._vol_dim, dtype=np.float32)
        self._weight_vol_cpu = np.zeros(self._vol_dim, dtype=np.float32)
        self._color_vol_cpu = np.zeros(self._vol_dim, dtype=np.float32)

        self.gpu_mode = use_gpu and FUSION_GPU_MODE

        if not self.gpu_mode:
            xv, yv, zv = np.meshgrid(
                np.arange(self._vol_dim[0], dtype=np.int32),
                np.arange(self._vol_dim[1], dtype=np.int32),
                np.arange(self._vol_dim[2], dtype=np.int32),
                indexing='ij'
            )
            self.vox_coords = np.stack([xv.ravel(), yv.ravel(), zv.ravel()], axis=1).astype(np.int32)
        else:
            # GPU setup remains the same...
            pass

    @staticmethod
    @njit(parallel=True)
    def vox2world(vol_origin, vox_coords, vox_size):
        vol_origin = vol_origin.astype(np.float32)
        vox_coords = vox_coords.astype(np.float32)
        cam_pts = np.empty_like(vox_coords, dtype=np.float32)
        for i in prange(vox_coords.shape[0]):
            for j in range(3):
                cam_pts[i, j] = vol_origin[j] + (vox_size * vox_coords[i, j])
        return cam_pts

    @staticmethod
    @njit(parallel=True)
    def cam2pix(cam_pts, intr):
        intr = intr.astype(np.float32)
        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]
        pix = np.empty((cam_pts.shape[0], 2), dtype=np.int64)
        for i in prange(cam_pts.shape[0]):
            pix[i, 0] = int(np.round((cam_pts[i, 0] * fx / cam_pts[i, 2]) + cx))
            pix[i, 1] = int(np.round((cam_pts[i, 1] * fy / cam_pts[i, 2]) + cy))
        return pix

    def integrate(self, color_im, depth_im, cam_intr, cam_pose, obs_weight=1.):
        im_h, im_w = depth_im.shape
        if color_im is not None:
            
            color_im_flat = np.floor(color_im[..., 2].astype(np.float32) * self._color_const + 
                                     color_im[..., 1].astype(np.float32) * 256 + 
                                     color_im[..., 0]).astype(np.float32).reshape(-1)
        
        chunk_size = 500000 
        for i in range(0, self.vox_coords.shape[0], chunk_size):
            end = min(i + chunk_size, self.vox_coords.shape[0])
            vox_chunk = self.vox_coords[i:end]
            
            cam_pts = self.vox2world(self._vol_origin, vox_chunk, self._voxel_size)
            cam_pts = rigid_transform(cam_pts, np.linalg.inv(cam_pose))
            
            pix = self.cam2pix(cam_pts, cam_intr)
            valid = (pix[:, 0] >= 0) & (pix[:, 0] < im_w) & (pix[:, 1] >= 0) & (pix[:, 1] < im_h) & (cam_pts[:, 2] > 0)
            
            if not np.any(valid): continue
            
            v_idx = np.where(valid)[0]
            v_x, v_y, v_z = vox_chunk[v_idx, 0], vox_chunk[v_idx, 1], vox_chunk[v_idx, 2]
            p_x, p_y = pix[v_idx, 0], pix[v_idx, 1]
            
            depth_val = depth_im[p_y, p_x]
            depth_diff = depth_val - cam_pts[v_idx, 2]
            valid_pts = (depth_val > 0) & (depth_diff >= -self._trunc_margin)
            
            if np.any(valid_pts):
                v_x, v_y, v_z = v_x[valid_pts], v_y[valid_pts], v_z[valid_pts]
                w_old = self._weight_vol_cpu[v_x, v_y, v_z]
                dist = np.minimum(1, depth_diff[valid_pts] / self._trunc_margin)
                
                w_new = w_old + obs_weight
                self._tsdf_vol_cpu[v_x, v_y, v_z] = (w_old * self._tsdf_vol_cpu[v_x, v_y, v_z] + obs_weight * dist) / w_new
                self._weight_vol_cpu[v_x, v_y, v_z] = w_new
                
                if color_im is not None:
                    c_old = self._color_vol_cpu[v_x, v_y, v_z]
                    c_new = color_im_flat[p_y[valid_pts] * im_w + p_x[valid_pts]]
                    self._color_vol_cpu[v_x, v_y, v_z] = (w_old * c_old + obs_weight * c_new) / w_new

    def get_volume(self):
        return self._tsdf_vol_cpu, self._color_vol_cpu, self._weight_vol_cpu

    def get_mesh(self):
        tsdf_vol, color_vol, _ = self.get_volume()
        try:
            verts, faces, norms, vals = measure.marching_cubes(tsdf_vol, level=0)
        except ValueError:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        verts_ind = np.round(verts).astype(np.int32)
        verts = verts * self._voxel_size + self._vol_origin
        rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        colors = np.floor(np.stack([
            rgb_vals - np.floor(rgb_vals/(256*256))*256*256 - np.floor((rgb_vals - np.floor(rgb_vals/(256*256))*256*256)/256)*256,
            np.floor((rgb_vals - np.floor(rgb_vals/(256*256))*256*256)/256),
            np.floor(rgb_vals/(256*256))
        ], axis=1)).astype(np.uint8)
        return verts, faces, norms, colors

def rigid_transform(xyz, transform):
    xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
    return np.dot(transform, xyz_h.T).T[:, :3]