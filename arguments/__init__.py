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

from argparse import ArgumentParser, Namespace
import sys
import os
import math
import torch

footprint_activation = lambda x: 1. - torch.exp(-0.03279 * torch.pow(x, 3.4))
inverse_footprint_activation = lambda y: math.pow(-math.log(1 - y) / 0.03279, 1 / 3.4)

S_activation = lambda x: 0.03279 * torch.pow(x.clamp_min(0.), 3.4)
inverse_S_activation = lambda y: torch.pow(y.clamp_min(0.) / 0.03279, 1 / 3.4)

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 9
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.geovalue_mul = 5.0
        self.max_scaling = 0.05
        self.max_lambda = 1000.
        self.max_inverse_falloff = 1.
        self.intensity_norm = 1.

        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.depth_ratio = 0.0
        self.debug = False

        # 'MC' or 'hybrid' for global illumination, 'PR' for direct illumination
        self.solver_type = 'hybrid'
        self.not_use_cluster = False
        self.use_tone_mapper = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.shininess_lr = 0.0025
        self.blending_lr = 0.0025
        self.diffuse_albedo_lr = 0.0025
        self.specular_albedo_lr = 0.0025
        self.norm_factor_lr = 0.0025
        self.geovalue_lr_init = 0.01
        self.geovalue_lr_final = 0.05
        self.geovalue_lr_max_steps = 7_000
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.001
        self.lambda_dssim = 0.2
        self.lambda_dist = 0.0
        self.lambda_normal = 0.05
        self.lambda_alpha = 0.1

        self.init_sh_degree = 1
        self.init_env_map = 1.
        self.env_map_lr = 0.001
        self.lambda_env_map = 0.05

        self.geovalue_init = inverse_footprint_activation(0.1)
        self.geovalue_cull = inverse_footprint_activation(0.05)

        self.ndc_start_iteration = 500

        self.densification_interval = 300
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.densify_absgrad_threshold = 0.0004
        self.densify_absgrad_start_iteration = 0
        self.densify_clone_only_from_iter = 8000

        self.mc_init_walks = 0
        self.mc_final_walks = 64
        self.mc_start_steps = 500
        self.mc_decay_steps = 7000

        self.min_decay_init = 1e-3
        self.min_decay_final = 1e-4
        self.min_decay_start_steps = 2000
        self.min_decay_decay_steps = 4000
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
