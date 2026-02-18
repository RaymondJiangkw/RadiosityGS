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

import os
import cv2
import glob
import imageio
import trimesh
import open3d as o3d
from tqdm import tqdm
from argparse import ArgumentParser, Namespace

import torch
import numpy as np

from scene import *
from dnnlib import EasyDict
from gaussian_renderer import *
from PIL import Image
from scipy.spatial import cKDTree as KDTree
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh

#----------------------------------------------------------------------------

def compute_trimesh_chamfer(gt_points, gen_mesh, num_mesh_samples=500000):
    gen_points_sampled = trimesh.sample.sample_surface(gen_mesh, num_mesh_samples)[0]

    # only need numpy array of points
    gt_points_np = gt_points.vertices
    # gt_points_np = trimesh.sample.sample_surface(gt_points, num_mesh_samples)[0]

    # one direction
    gen_points_kd_tree = KDTree(gen_points_sampled)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points_np)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points_np)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points_sampled)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_gen_chamfer, gen_to_gt_chamfer

#----------------------------------------------------------------------------

def read_cfg(path: str):
    assert os.path.exists(os.path.join(path, 'cfg_args'))
    with open(os.path.join(path, 'cfg_args')) as f:
        string = f.read()
    args = eval(string)
    return EasyDict(**vars(args).copy())

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    pipeline = PipelineParams(parser)
    parser.add_argument("--model_path", "-m", type=str, required=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_novel", action="store_true")
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--skip_metrics", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--caching", action="store_true")
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')
    parser.add_argument('--num_walks', type=int, default=128)
    parser.add_argument('--nop', type=int, default=500000)
    parser.add_argument('--max_dist', type=float, default=0.1)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    dataset = read_cfg(args.model_path)

    iteration, pipe = args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset)
    light_sources = LightModel(dataset)
    light_sources.create_from_env_map(init_intensity=1.)
    scene = Scene(dataset, gaussians, light_sources, load_iteration=iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    train_dir = os.path.join(args.model_path, 'train', "ours_{}".format(scene.loaded_iter))
    test_dir = os.path.join(args.model_path, 'test', "ours_{}".format(scene.loaded_iter))
    novel_dir = os.path.join(args.model_path, 'novel', "ours_{}".format(scene.loaded_iter))
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)
    
    if not args.skip_train:
        print("export training images ...")
        os.makedirs(train_dir, exist_ok=True)
        views = scene.getTrainCameras()
        if 'TensoIRSynthetic' in dataset.source_path:
            for view in views:
                view.original_albedo_image = torch.from_numpy(
                    np.array(Image.open(view.image_path.replace("rgba.png", f"albedo.png")))
                ).to(torch.float32).cuda().permute(2, 0, 1)[:3] / 255.
        elif 'Synthetic4Relight' in dataset.source_path:
            for view in views:
                image = torch.from_numpy(
                    np.array(Image.open(view.image_path.replace("rgba.png", f"albedo.png")))
                ).to(torch.float32).cuda().permute(2, 0, 1) / 255.
                view.original_albedo_image = image[:3] * image[3:]
        gaussExtractor.reconstruction(views)
        gaussExtractor.reconstructionGI(views, light_sources, pipe, background, args.num_walks, caching=args.caching)
        gaussExtractor.export_image(train_dir)
    
    if (not args.skip_test) and (len(scene.getTestCameras()) > 0):
        print("export rendered testing images ...")
        os.makedirs(test_dir, exist_ok=True)
        views = scene.getTestCameras()
        if 'TensoIRSynthetic' in dataset.source_path:
            for view in views:
                view.original_albedo_image = torch.from_numpy(
                    np.array(Image.open(view.image_path.replace("rgba.png", f"albedo.png")))
                ).to(torch.float32).cuda().permute(2, 0, 1)[:3] / 255.
        elif 'Synthetic4Relight' in dataset.source_path:
            for view in views:
                image = torch.from_numpy(
                    np.array(Image.open(view.image_path.replace("rgba.png", f"albedo.png")))
                ).to(torch.float32).cuda().permute(2, 0, 1) / 255.
                view.original_albedo_image = image[:3] * image[3:]
        gaussExtractor.reconstruction(views)
        gaussExtractor.reconstructionGI(views, light_sources, pipe, background, args.num_walks, caching=args.caching)
        gaussExtractor.export_image(test_dir)
    
    if (not args.skip_novel):
        if 'Stanford-ORB' in dataset.source_path and os.path.exists(os.path.join(dataset.source_path, 'transforms_novel.json')):
            print("export rendered novel images ...")
            os.makedirs(novel_dir, exist_ok=True)
            with open(os.path.join(dataset.source_path, 'transforms_novel.json'), 'r') as f:
                transforms_novel = json.load(f)['frames']
            blender_HDR_path = os.path.abspath(os.path.join(dataset.source_path, ".."))
            gt_env_map_path = blender_HDR_path.replace("blender_HDR", "ground_truth")
            assert os.path.exists(blender_HDR_path)
            assert os.path.exists(gt_env_map_path)
            gaussExtractor.reconstructionGINovel(dataset, transforms_novel, blender_HDR_path, gt_env_map_path, pipe, background, args.num_walks)
            gaussExtractor.export_image(novel_dir)
        
        if 'TensoIRSynthetic' in dataset.source_path:
            os.makedirs(novel_dir, exist_ok=True)
            with open(os.path.join(dataset.source_path, 'transforms_test.json'), 'r') as f:
                transforms_test = json.load(f)['frames']
            blender_HDR_path = os.path.abspath(os.path.join(dataset.source_path, "..", "Environment_Maps", "high_res_envmaps_1k"))
            assert os.path.exists(blender_HDR_path)
            test_cameras = scene.getTestCameras()
            
            for blender_HDR_name in ["bridge", "city", "fireplace", "forest", "night", "sunset"]:
                print(f"Dealing with Environment map {blender_HDR_name} ...")
                for view in test_cameras:
                    view.original_image = torch.from_numpy(
                        np.array(Image.open(view.image_path.replace("rgba.png", f"rgba_{blender_HDR_name}.png")))
                    ).to(torch.float32).cuda().permute(2, 0, 1)[:3] / 255.
                    view.original_albedo_image = None
                out_dir = os.path.join(novel_dir, blender_HDR_name)
                ls = LightModel(dataset)
                hdr_path = os.path.join(blender_HDR_path, blender_HDR_name + ".hdr")
                envmap = imageio.v3.imread(hdr_path)
                envmap = torch.from_numpy(envmap).permute(2, 0, 1).cuda()
                ls.create_from_env_map(envmap / 20.0, convention='blender')
                gaussExtractor.reconstructionGI(test_cameras, ls, pipe, background, args.num_walks, clean_albedo=True, caching=args.caching)
                gaussExtractor.export_image(out_dir)
        
        if 'Synthetic4Relight' in dataset.source_path:
            os.makedirs(novel_dir, exist_ok=True)
            with open(os.path.join(dataset.source_path, 'transforms_test.json'), 'r') as f:
                transforms_test = json.load(f)['frames']
            blender_HDR_path = os.path.abspath(os.path.join(dataset.source_path, "..", "EnvironmentMap"))
            assert os.path.exists(blender_HDR_path)
            test_cameras = scene.getTestCameras()
            
            for blender_HDR_name in ["envmap6", "envmap12"]:
                print(f"Dealing with Environment map {blender_HDR_name} ...")
                for view in test_cameras:
                    path = view.image_path.replace("/test", "/test_rli")
                    image = torch.from_numpy(
                        np.array(Image.open(os.path.join(os.path.dirname(path), blender_HDR_name + "_" + os.path.basename(path).replace("_rgba.png", ".png"))))
                    ).to(torch.float32).cuda().permute(2, 0, 1) / 255.
                    view.original_image = image[:3] * image[3:]
                    view.original_albedo_image = None
                out_dir = os.path.join(novel_dir, blender_HDR_name)
                ls = LightModel(dataset)
                hdr_path = os.path.join(blender_HDR_path, blender_HDR_name + ".exr")
                envmap = imageio.v3.imread(hdr_path)
                envmap = torch.from_numpy(envmap).squeeze().permute(2, 0, 1).cuda()[:3]
                ls.create_from_env_map(envmap / 20.0, convention='blender')
                gaussExtractor.reconstructionGI(test_cameras, ls, pipe, background, args.num_walks, clean_albedo=True, caching=args.caching)
                gaussExtractor.export_image(out_dir)
    
    if not args.skip_mesh:
        print("export mesh ...")
        os.makedirs(train_dir, exist_ok=True)
        # set the active_sh to 0 to export only diffuse texture
        gaussExtractor.gaussians.active_sh_degree = 0
        gaussExtractor.reconstruction(scene.getTrainCameras())
        # extract the mesh and save
        if args.unbounded:
            name = 'fuse_unbounded.ply'
            mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
        else:
            name = 'fuse.ply'
            depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0  else args.depth_trunc
            voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
            sdf_trunc = 4.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
            mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
        
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)
        print("mesh saved at {}".format(os.path.join(train_dir, name)))
        # post-process the mesh and save, saving the largest N clusters
        try:
            mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
            o3d.io.write_triangle_mesh(os.path.join(train_dir, name.replace('.ply', '_post.ply')), mesh_post)
            print("mesh post processed saved at {}".format(os.path.join(train_dir, name.replace('.ply', '_post.ply'))))
        except:
            pass

        if not args.skip_metrics:
            if 'Stanford-ORB' in dataset.source_path:
                input_dir = os.path.join(test_dir, "vis")
                mask_dir = os.path.join(dataset.source_path, "test_mask")
                
                gt_mesh = trimesh.load(os.path.join(dataset.source_path.replace("blender_HDR", "ground_truth"), 'mesh_blender', 'mesh.obj'))
                mesh = trimesh.load(os.path.join(train_dir, 'fuse.ply'))
                gt_to_gen_chamfer, gen_to_gt_chamfer = compute_trimesh_chamfer(gt_mesh, mesh)
                shape_metric = (gt_to_gen_chamfer + gen_to_gt_chamfer) / 2.

                print("Shape:", f"{shape_metric} ({gt_to_gen_chamfer}, {gen_to_gt_chamfer})")
                with open(os.path.join(test_dir, "geometry_metrics.txt"), "w") as f:
                    f.write(f"Shape: {shape_metric}")