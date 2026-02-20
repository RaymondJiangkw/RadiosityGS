#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from dnnlib import EasyDict
from utils.graphics_utils import fov2focal, getWorld2View2, getProjectionMatrix
from utils.sh_utils import SH2RGB, RGB2SH

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, image_path, uid, pl_pos, pl_intensity, exr_image, 
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", 
                 transform_matrix=None, intensity_norm=1.
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image_path = image_path
        self.transform_matrix = transform_matrix

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device) if image is not None else None
        self.exr_image = exr_image.to(self.data_device) if exr_image is not None else None
        self.image_width = self.original_image.shape[2] if self.original_image is not None else (self.exr_image.shape[2] if self.exr_image is not None else None)
        self.image_height = self.original_image.shape[1] if self.original_image is not None else (self.exr_image.shape[2] if self.exr_image is not None else None)
        
        self.pl_pos = torch.nn.Parameter(torch.from_numpy((pl_pos + trans) * scale).float().to(self.data_device)) if pl_pos is not None else None
        self.pl_intensity = torch.from_numpy(RGB2SH(pl_intensity / intensity_norm)).float().to(self.data_device)[None, :] if pl_intensity is not None else None
        
        if gt_alpha_mask is not None:
            self.original_image = torch.clamp_min(self.original_image * gt_alpha_mask.to(self.data_device), 0.0) if self.original_image is not None else None
            self.exr_image = torch.clamp_min(self.exr_image * gt_alpha_mask.to(self.data_device), 0.0) if exr_image is not None else None
            self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)
        else:
            self.gt_alpha_mask = None
        
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.view_world_transform = self.world_view_transform.inverse()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.view_world_transform[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
