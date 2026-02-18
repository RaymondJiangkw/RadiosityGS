#
# Copyright (C) 2024, ShanghaiTech
# SVIP research group, https://github.com/svip-lab
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  huangbb@shanghaitech.edu.cn
#

import torch
import numpy as np
import os
import cv2
import math
from dnnlib import EasyDict
from tqdm import tqdm
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov, fov2focal
from utils.render_utils import save_img_f32, save_img_u8
from functools import partial
import open3d as o3d
from scene.light_source import LightModel
from gaussian_renderer import renderGI

def post_process_mesh(mesh, cluster_to_keep=1000):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0

def to_cam_open3d(viewpoint_stack):
    camera_traj = []
    for i, viewpoint_cam in enumerate(viewpoint_stack):
        W = viewpoint_cam.image_width
        H = viewpoint_cam.image_height
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, 0, 1]]).float().cuda().T
        intrins =  (viewpoint_cam.projection_matrix @ ndc2pix)[:3,:3].T
        intrinsic=o3d.camera.PinholeCameraIntrinsic(
            width=viewpoint_cam.image_width,
            height=viewpoint_cam.image_height,
            cx = intrins[0,2].item(),
            cy = intrins[1,2].item(), 
            fx = intrins[0,0].item(), 
            fy = intrins[1,1].item()
        )

        extrinsic=np.asarray((viewpoint_cam.world_view_transform.T).cpu().numpy())
        camera = o3d.camera.PinholeCameraParameters()
        camera.extrinsic = extrinsic
        camera.intrinsic = intrinsic
        camera_traj.append(camera)

    return camera_traj

import Imath
import OpenEXR
def read_env_exr(exr_path):
    # Read EXR file
    assert os.path.exists(exr_path)
    exr_file = OpenEXR.InputFile(exr_path)
    header = exr_file.header()

    # Extract channel information
    dw = header['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Extract RGB channels
    channels = ['R', 'G', 'B']
    exr_image = []
    for channel in channels:
        raw_data = exr_file.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT))
        exr_image.append(np.frombuffer(raw_data, dtype=np.float32).reshape(size[1], size[0]))
    return np.stack(exr_image, axis=0)

def env_map_to_cam_to_world_by_convention(envmap: np.ndarray, c2w):
    # Adapted from https://github.com/StanfordORB/Stanford-ORB/blob/8a3bdaeaaf00b5d4768b11e2950090eba73a3ac7/orb/utils/env_map.py#L12
    R = c2w[:3,:3]
    H, W = envmap.shape[:2]
    theta, phi = np.meshgrid(np.linspace(-0.5*np.pi, 1.5*np.pi, W), np.linspace(0., np.pi, H))
    viewdirs = np.stack([-np.cos(theta) * np.sin(phi), np.cos(phi), -np.sin(theta) * np.sin(phi)],
                           axis=-1).reshape(H*W, 3)    # [H, W, 3]
    viewdirs = (R.T @ viewdirs.T).T.reshape(H, W, 3)
    viewdirs = viewdirs.reshape(H, W, 3)
    # This is correspond to the convention of +Z at left, +Y at top
    # -np.cos(theta) * np.sin(phi), np.cos(phi), -np.sin(theta) * np.sin(phi)
    coord_y = ((np.arccos(viewdirs[..., 1])/np.pi*(H-1)+H)%H).astype(np.float32)
    coord_x = (((np.arctan2(viewdirs[...,0], -viewdirs[...,2])+np.pi)/2/np.pi*(W-1)+W)%W).astype(np.float32)
    envmap_remapped = cv2.remap(envmap, coord_x, coord_y, cv2.INTER_LINEAR)
    
    envmap_remapped_physg = np.roll(envmap_remapped, W//4, axis=1)
    return envmap_remapped_physg

class GaussianExtractor(object):
    def __init__(self, gaussians, render, pipe, bg_color=None):
        """
        a class that extracts attributes a scene presented by 2DGS

        Usage example:
        >>> gaussExtrator = GaussianExtractor(gaussians, render, pipe)
        >>> gaussExtrator.reconstruction(view_points)
        >>> mesh = gaussExtractor.export_mesh_bounded(...)
        """
        if bg_color is None:
            bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.gaussians = gaussians
        self.render = partial(render, pipe=pipe, bg_color=background)
        self.clean()

    @torch.no_grad()
    def clean(self, clean_albedo=True):
        self.depthmaps = []
        # self.alphamaps = []
        self.rgbmaps = []
        self.normals = []
        if clean_albedo or getattr(self, 'albedos', None) is None:
            self.albedos = []
        self.depth_normals = []
        self.viewpoint_stack = []

    @torch.no_grad()
    def reconstruction(self, viewpoint_stack):
        """
        reconstruct radiance field given cameras
        """
        self.clean()
        self.viewpoint_stack = viewpoint_stack
        for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="reconstruct radiance fields"):
            render_pkg = self.render(viewpoint_cam, self.gaussians.get_xyz, self.gaussians.get_rotation, self.gaussians.get_geovalue, self.gaussians.get_scaling, None, 0, colors_precomp=self.gaussians.get_real_diffuse_albedos)
            
            rgb = render_pkg['render']
            alpha = render_pkg['rend_alpha']
            normal = torch.nn.functional.normalize((render_pkg['rend_normal'].permute(1, 2, 0) @ viewpoint_cam.view_world_transform[:3, :3].T).permute(2, 0, 1), dim=0)
            depth = render_pkg['surf_depth']
            depth_normal = torch.nn.functional.normalize((render_pkg['surf_normal'].permute(1, 2, 0) @ viewpoint_cam.view_world_transform[:3, :3].T).permute(2, 0, 1), dim=0)
            self.albedos.append(rgb.cpu())
            # self.rgbmaps.append(rgb.cpu())
            self.depthmaps.append(depth.cpu())
            # self.alphamaps.append(alpha.cpu())
            self.normals.append(normal.cpu())
            self.depth_normals.append(depth_normal.cpu())
        
        # self.rgbmaps = torch.stack(self.rgbmaps, dim=0)
        # self.depthmaps = torch.stack(self.depthmaps, dim=0)
        # self.alphamaps = torch.stack(self.alphamaps, dim=0)
        # self.depth_normals = torch.stack(self.depth_normals, dim=0)
        self.estimate_bounding_sphere()
    
    @torch.no_grad()
    def reconstructionGINovel(self, cfg, transforms_novel, blender_HDR_path, gt_env_map_path, pipe, background, num_walks):
        self.clean()
        self.viewpoint_stack = []
        for frame in tqdm(transforms_novel, desc="reconstruct radiance fields"):
            fovx = frame["camera_angle_x"]
            fovy = None
            fx = fy = None
            c2w = np.array(frame["transform_matrix"])
            c2w[:3, 1:3] *= -1
            
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            
            H = W = None
            bg = np.array([0, 0, 0])
            
            exr_image = None
            exr_path = os.path.join(blender_HDR_path, frame['scene_name'], f"{frame['file_path']}.exr")
            assert os.path.exists(exr_path)
            
            exr_image = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
            norm_data = cv2.cvtColor(exr_image, cv2.COLOR_BGRA2RGBA)
            
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            exr_image = arr.astype(np.float32).transpose(2, 0, 1)
            exr_image = np.concatenate((exr_image, norm_data[:, :, -1][None]))
        
            W = exr_image.shape[-1]
            H = exr_image.shape[-2]
            
            if fx is None and fy is None:
                FovY = fovy if fovy is not None else focal2fov(fov2focal(fovx, W), H)
                FovX = fovx
            else:
                FovY = focal2fov(fy, H)
                FovX = focal2fov(fx, W)

            device = "cuda"

            novel_view = loadCam(
                EasyDict(data_device=device, resolution=-1, cam_scale=1.), 
                0, 
                EasyDict(image=None, exr_image=exr_image, 
                        uid=0, R=R, T=T, FovX=FovX, FovY=FovY, pl_pos=None, pl_intensity=None, 
                        image_name=None, image_path=None, transform_matrix=None), 
                1
            )
            gt_env_map = os.path.join(gt_env_map_path, frame['scene_name'], "env_map")
            gt_env_map = os.path.join(gt_env_map, frame['file_path'].split('/')[-1] + '.exr')
            gt_env_map = torch.from_numpy(read_env_exr(gt_env_map)).to(device)
            c2w = np.array(frame["transform_matrix"])
            gt_env_map = env_map_to_cam_to_world_by_convention(gt_env_map.permute(1, 2, 0).cpu().numpy(), c2w)
            gt_env_map = torch.from_numpy(gt_env_map.copy()).permute(2, 0, 1).to(device)
            gt_light_sources = LightModel(cfg)
            gt_light_sources.create_from_env_map(gt_env_map)

            render_pkg = renderGI(novel_view, self.gaussians, gt_light_sources, pipe, background, override_solver_settings={"num_walks": num_walks})

            # rgb = render_pkg['render']
            # Scale each channel following Stanford-ORB's instructions
            render = render_pkg["render"].permute(1,2,0).cpu().numpy()
            gt = novel_view.exr_image.permute(1,2,0).cpu().numpy()
            mask_gt = novel_view.gt_alpha_mask.squeeze().cpu().numpy()
            
            img_pred_pixels = render[np.where(mask_gt > 0.5)]
            img_gt_pixels = gt[np.where(mask_gt > 0.5)]
            for c in range(3):
                if (img_pred_pixels[:, c] ** 2).sum() <= 1e-6:
                    img_pred_pixels[:, c] = np.ones_like(img_pred_pixels[:, c])
            scale = (img_gt_pixels * img_pred_pixels).sum(axis=0) / (img_pred_pixels ** 2).sum(axis=0)
            assert scale.shape == (3,), scale.shape
            rgb = (torch.from_numpy(render).cuda()[None].permute(0, 3, 1, 2) * torch.from_numpy(scale).cuda()[None, :, None, None]).squeeze(0) * novel_view.gt_alpha_mask.squeeze()[None]

            alpha = render_pkg['rend_alpha']
            normal = torch.nn.functional.normalize((render_pkg['rend_normal'].permute(1, 2, 0) @ novel_view.view_world_transform[:3, :3].T).permute(2, 0, 1), dim=0)
            depth = render_pkg['surf_depth']
            depth_normal = torch.nn.functional.normalize((render_pkg['surf_normal'].permute(1, 2, 0) @ novel_view.view_world_transform[:3, :3].T).permute(2, 0, 1), dim=0)
            self.rgbmaps.append(rgb.cpu())
            self.depthmaps.append(depth.cpu())
            # self.alphamaps.append(alpha.cpu())
            self.normals.append(normal.cpu())
            self.depth_normals.append(depth_normal.cpu())
            self.viewpoint_stack.append(novel_view)
        
    @torch.no_grad()
    def reconstructionGI(self, viewpoint_stack, ls, pipe, background, num_walks, clean_albedo=False, caching=False):
        """
        reconstruct radiance field given cameras
        """
        self.clean(clean_albedo=clean_albedo)
        self.viewpoint_stack = viewpoint_stack
        cached_radiosities = None
        for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="reconstruct radiance fields"):
            render_pkg = renderGI(viewpoint_cam, self.gaussians, ls.from_camera_if_possible(viewpoint_cam), pipe, background, override_solver_settings={"num_walks": num_walks}, override_radiosities=cached_radiosities)
            cached_radiosities = render_pkg["radiosity"] if caching else None
            rgb = render_pkg['render']
            alpha = render_pkg['rend_alpha']
            normal = torch.nn.functional.normalize((render_pkg['rend_normal'].permute(1, 2, 0) @ viewpoint_cam.view_world_transform[:3, :3].T).permute(2, 0, 1), dim=0)
            depth = render_pkg['surf_depth']
            depth_normal = torch.nn.functional.normalize((render_pkg['surf_normal'].permute(1, 2, 0) @ viewpoint_cam.view_world_transform[:3, :3].T).permute(2, 0, 1), dim=0)
            self.rgbmaps.append(rgb.cpu())
            self.depthmaps.append(depth.cpu())
            # self.alphamaps.append(alpha.cpu())
            self.normals.append(normal.cpu())
            self.depth_normals.append(depth_normal.cpu())
        
        # self.rgbmaps = torch.stack(self.rgbmaps, dim=0)
        # self.depthmaps = torch.stack(self.depthmaps, dim=0)
        # self.alphamaps = torch.stack(self.alphamaps, dim=0)
        # self.depth_normals = torch.stack(self.depth_normals, dim=0)
        self.estimate_bounding_sphere()

    def estimate_bounding_sphere(self):
        """
        Estimate the bounding sphere given camera pose
        """
        from utils.render_utils import transform_poses_pca, focus_point_fn
        torch.cuda.empty_cache()
        c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in self.viewpoint_stack])
        poses = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
        center = (focus_point_fn(poses))
        self.radius = np.linalg.norm(c2ws[:,:3,3] - center, axis=-1).min()
        self.center = torch.from_numpy(center).float().cuda()
        print(f"The estimated bounding radius is {self.radius:.2f}")
        print(f"Use at least {2.0 * self.radius:.2f} for depth_trunc")

    @torch.no_grad()
    def extract_mesh_bounded(self, voxel_size=0.004, sdf_trunc=0.02, depth_trunc=3, mask_backgrond=True):
        """
        Perform TSDF fusion given a fixed depth range, used in the paper.
        
        voxel_size: the voxel size of the volume
        sdf_trunc: truncation value
        depth_trunc: maximum depth range, should depended on the scene's scales
        mask_backgrond: whether to mask backgroud, only works when the dataset have masks

        return o3d.mesh
        """
        print("Running tsdf volume integration ...")
        print(f'voxel_size: {voxel_size}')
        print(f'sdf_trunc: {sdf_trunc}')
        print(f'depth_truc: {depth_trunc}')

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length= voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        for i, cam_o3d in tqdm(enumerate(to_cam_open3d(self.viewpoint_stack)), desc="TSDF integration progress"):
            rgb = self.rgbmaps[i]
            depth = self.depthmaps[i]
            
            # if we have mask provided, use it
            if mask_backgrond and (self.viewpoint_stack[i].gt_alpha_mask is not None):
                depth[(self.viewpoint_stack[i].gt_alpha_mask < 0.5)] = 0

            # make open3d rgbd
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.asarray(np.clip(rgb.permute(1,2,0).cpu().numpy(), 0.0, 1.0) * 255, order="C", dtype=np.uint8)),
                o3d.geometry.Image(np.asarray(depth.permute(1,2,0).cpu().numpy() * 1000., order="C", dtype=np.uint16)),
                depth_trunc = depth_trunc, convert_rgb_to_intensity=False,
                depth_scale = 1000.0
            )

            volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)

        mesh = volume.extract_triangle_mesh()
        return mesh

    @torch.no_grad()
    def extract_mesh_unbounded(self, resolution=1024):
        """
        Experimental features, extracting meshes from unbounded scenes, not fully test across datasets. 
        return o3d.mesh
        """
        def contract(x):
            mag = torch.linalg.norm(x, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))
        
        def uncontract(y):
            mag = torch.linalg.norm(y, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, y, (1 / (2-mag) * (y/mag)))

        def compute_sdf_perframe(i, points, depthmap, rgbmap, viewpoint_cam):
            """
                compute per frame sdf
            """
            new_points = torch.cat([points, torch.ones_like(points[...,:1])], dim=-1) @ viewpoint_cam.full_proj_transform
            z = new_points[..., -1:]
            pix_coords = (new_points[..., :2] / new_points[..., -1:])
            mask_proj = ((pix_coords > -1. ) & (pix_coords < 1.) & (z > 0)).all(dim=-1)
            sampled_depth = torch.nn.functional.grid_sample(depthmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(-1, 1)
            sampled_rgb = torch.nn.functional.grid_sample(rgbmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(3,-1).T
            sdf = (sampled_depth-z)
            return sdf, sampled_rgb, mask_proj

        def compute_unbounded_tsdf(samples, inv_contraction, voxel_size, return_rgb=False):
            """
                Fusion all frames, perform adaptive sdf_funcation on the contract spaces.
            """
            if inv_contraction is not None:
                mask = torch.linalg.norm(samples, dim=-1) > 1
                # adaptive sdf_truncation
                sdf_trunc = 5 * voxel_size * torch.ones_like(samples[:, 0])
                sdf_trunc[mask] *= 1/(2-torch.linalg.norm(samples, dim=-1)[mask].clamp(max=1.9))
                samples = inv_contraction(samples)
            else:
                sdf_trunc = 5 * voxel_size

            tsdfs = torch.ones_like(samples[:,0]) * 1
            rgbs = torch.zeros((samples.shape[0], 3)).cuda()

            weights = torch.ones_like(samples[:,0])
            for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="TSDF integration progress"):
                sdf, rgb, mask_proj = compute_sdf_perframe(i, samples,
                    depthmap = self.depthmaps[i],
                    rgbmap = self.rgbmaps[i],
                    viewpoint_cam=self.viewpoint_stack[i],
                )

                # volume integration
                sdf = sdf.flatten()
                mask_proj = mask_proj & (sdf > -sdf_trunc)
                sdf = torch.clamp(sdf / sdf_trunc, min=-1.0, max=1.0)[mask_proj]
                w = weights[mask_proj]
                wp = w + 1
                tsdfs[mask_proj] = (tsdfs[mask_proj] * w + sdf) / wp
                rgbs[mask_proj] = (rgbs[mask_proj] * w[:,None] + rgb[mask_proj]) / wp[:,None]
                # update weight
                weights[mask_proj] = wp
            
            if return_rgb:
                return tsdfs, rgbs

            return tsdfs

        normalize = lambda x: (x - self.center) / self.radius
        unnormalize = lambda x: (x * self.radius) + self.center
        inv_contraction = lambda x: unnormalize(uncontract(x))

        N = resolution
        voxel_size = (self.radius * 2 / N)
        print(f"Computing sdf gird resolution {N} x {N} x {N}")
        print(f"Define the voxel_size as {voxel_size}")
        sdf_function = lambda x: compute_unbounded_tsdf(x, inv_contraction, voxel_size)
        from utils.mcube_utils import marching_cubes_with_contraction
        R = contract(normalize(self.gaussians.get_xyz)).norm(dim=-1).cpu().numpy()
        R = np.quantile(R, q=0.95)
        R = min(R+0.01, 1.9)

        mesh = marching_cubes_with_contraction(
            sdf=sdf_function,
            bounding_box_min=(-R, -R, -R),
            bounding_box_max=(R, R, R),
            level=0,
            resolution=N,
            inv_contraction=inv_contraction,
        )
        
        # coloring the mesh
        torch.cuda.empty_cache()
        mesh = mesh.as_open3d
        print("texturing mesh ... ")
        _, rgbs = compute_unbounded_tsdf(torch.tensor(np.asarray(mesh.vertices)).float().cuda(), inv_contraction=None, voxel_size=voxel_size, return_rgb=True)
        mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs.cpu().numpy())
        return mesh

    @torch.no_grad()
    def export_image(self, path):
        render_path = os.path.join(path, "renders")
        gts_path = os.path.join(path, "gt")
        albedo_path = os.path.join(path, "albedos")
        gts_albedo_path = os.path.join(path, "gt_albedos")
        vis_path = os.path.join(path, "vis")
        # mask_path = os.path.join(path, "mask")
        os.makedirs(render_path, exist_ok=True)
        os.makedirs(vis_path, exist_ok=True)
        os.makedirs(gts_path, exist_ok=True)
        # os.makedirs(mask_path, exist_ok=True)
        if len(self.albedos) > 0:
            assert len(self.albedos) == len(self.rgbmaps)
            os.makedirs(albedo_path, exist_ok=True)
        for idx, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="export images"):
            if viewpoint_cam.exr_image is not None:
                gt = viewpoint_cam.exr_image[0:3, :, :].clamp(0., 1.)
            else:
                gt = viewpoint_cam.original_image.clamp(0., 1.)
            
            if getattr(viewpoint_cam, 'original_albedo_image', None) is not None:
                os.makedirs(gts_albedo_path, exist_ok=True)
                save_img_u8(viewpoint_cam.original_albedo_image.clamp(0., 1.).permute(1,2,0).cpu().numpy(), os.path.join(gts_albedo_path, '{0:05d}'.format(idx) + ".png"))
            
            rgbmap = self.rgbmaps[idx]
            # We apply masking to all baselines
            if viewpoint_cam.gt_alpha_mask is not None:
                rgbmap *= viewpoint_cam.gt_alpha_mask.squeeze()[None].to(rgbmap.device)
            
            save_img_u8(gt.permute(1,2,0).cpu().numpy(), os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            save_img_u8(rgbmap.clamp(0., 1.).permute(1,2,0).cpu().numpy(), os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            if len(self.albedos) > 0: save_img_u8(self.albedos[idx].clamp(0., 1.).permute(1,2,0).cpu().numpy(), os.path.join(albedo_path, '{0:05d}'.format(idx) + ".png"))

            # if viewpoint_cam.gt_alpha_mask is not None:
            #     save_img_u8(viewpoint_cam.gt_alpha_mask.squeeze().cpu().numpy(), os.path.join(mask_path, '{0:05d}'.format(idx) + ".png"))

            np.save(os.path.join(vis_path, 'depth_{0:05d}'.format(idx) + ".npy"), self.depthmaps[idx][0].cpu().numpy())
            np.save(os.path.join(vis_path, 'normal_{0:05d}'.format(idx) + ".npy"), self.normals[idx].cpu().numpy())
            np.save(os.path.join(vis_path, 'depth_normal_{0:05d}'.format(idx) + ".npy"), self.depth_normals[idx].cpu().numpy())
            # save_img_f32(self.depthmaps[idx][0].cpu().numpy(), os.path.join(vis_path, 'depth_{0:05d}'.format(idx) + ".tiff"))
            # save_img_u8(self.normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(vis_path, 'normal_{0:05d}'.format(idx) + ".png"))
            # save_img_u8(self.depth_normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(vis_path, 'depth_normal_{0:05d}'.format(idx) + ".png"))
