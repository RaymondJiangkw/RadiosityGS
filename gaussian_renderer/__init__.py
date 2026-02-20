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
import math
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer # type: ignore
from scene.gaussian_model import GaussianModel
from utils.point_utils import depth_to_normal
from submodules.radiosity_solver import *
from submodules.radiosity_solver.auto_diff import *

def tone_mapper(radiance_values: torch.Tensor) -> torch.Tensor:
    radiance_values = torch.log(radiance_values + 1.)
    return torch.where(radiance_values > 0.0031308, 
                       (1 + 0.055) * radiance_values.clamp_min(0.0031308).pow(1./2.4) - 0.055, 
                       12.92 * radiance_values)

def renderGI(
    viewpoint_camera, 
    pc, 
    ls, 
    pipe, 
    bg_color: torch.Tensor, 
    override_radiosities = None, 
    override_solver_settings = {}
):
    # Light sources primitives are appended to the last.
    num_elements_ls = len(ls.get_xyz)
    means3D = torch.cat((pc.get_xyz, ls.get_xyz))
    means2D = torch.zeros((len(means3D), 4), dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    try:
        means2D.retain_grad()
    except:
        pass
    geovalues = torch.cat((pc.get_geovalue, ls.get_geovalue))

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = torch.cat((pc.get_covariance(), ls.get_covariance()))
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:
        scales = torch.cat((pc.get_scaling, ls.get_scaling))
        rotations = torch.cat((pc.get_rotation, ls.get_rotation))
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None
    
    # Primitives in `pc` do not emit.
    emissions = torch.cat((torch.zeros((len(pc.get_xyz), *ls.get_emissions.shape[1:]), dtype=ls.get_emissions.dtype, device=ls.get_emissions.device), ls.get_emissions))
    
    # Light sources should not have BRDF. However, we treat light sources as special
    # Gaussian surfels which help unify and simplify the framework.
    brdf_coeffs = pc.get_brdf_coeffs
    brdf_coeffs = torch.cat((brdf_coeffs, torch.ones((len(ls.get_xyz), *brdf_coeffs.shape[1:]), dtype=torch.float32, device=brdf_coeffs.device) * 1e-8))
    
    # Explicitly specify primitives represent the light sources.
    is_light_source = torch.cat((torch.zeros((len(pc.get_xyz), 1), dtype=torch.bool, device=pc.get_xyz.device), ls.get_is_light_source))

    norm_factors = torch.cat((pc.get_norm_factor, ls.get_norm_factor))

    if override_radiosities is None:
        radiosity_solver = RadiosityPropagater(RadiosityPropagationSettings(solver_type=pipe.solver_type, debug=pipe.debug, active_sh_degree=pc.active_sh_degree, max_sh_degree=pc.max_sh_degree, directional_light_source=ls.is_directional_light, **override_solver_settings))

        radiosities = radiosity_solver(means3D, geovalues, scales, rotations, norm_factors, torch.clamp_min(emissions, 0.), brdf_coeffs, is_light_source)
    else:
        radiosities = torch.cat((override_radiosities, ls.get_emissions))

    shs = radiosities
    colors_precomp = None

    # We don't render point/directional light sources.
    render_pack = render(viewpoint_camera, means3D[:-num_elements_ls], rotations[:-num_elements_ls], geovalues[:-num_elements_ls], scales[:-num_elements_ls], shs[:-num_elements_ls], pc.active_sh_degree, pipe, bg_color, colors_precomp=colors_precomp)
    
    render_pack['radiosity'] = radiosities[:-num_elements_ls]

    if pipe.use_tone_mapper:
        render_pack['render'] = tone_mapper(render_pack['render'])
    
    return render_pack

def render(viewpoint_camera, 
           means3D, rotations, geovalues, scales, shs, active_sh_degree, 
           pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, colors_precomp = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(rotations, dtype=rotations.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        # pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points
    cov3D_precomp = None
    
    rendered_image, radii, count_accum, weight_accum, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        geovalues = geovalues,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rets =  {"render": rendered_image,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "count_accum": count_accum, 
            # "weight_accum": weight_accum, 
    }


    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    # render_depth_expected = (render_depth_expected / render_alpha)
    # render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

    render_depth_expected = (render_depth_expected / render_alpha.clamp_min(1E-8))
    
    # get depth distortion map
    render_dist = allmap[6:7]

    render_back_alpha = allmap[7:8]

    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1; 
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()

    rets.update({
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_back_alpha': render_back_alpha, 
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal
    })

    return rets