# !/usr/bin/python3
# coding: utf-8
# author: lixiang
# date: 2022-10-17

from pyhocon.config_tree import ConfigTree

class Train_Params():
    def __init__(self):
        super(Train_Params, self).__init__()

        self.learning_rate = 5e-4
        self.learning_rate_alpha = 0.05
        self.end_iter = 120000

        self.batch_size = 512
        self.validate_resolution_level = 4
        self.warm_up_end = 5000
        self.anneal_end = 0
        self.use_white_bkgd = False

        self.save_freq = 10000
        self.val_freq = 2500
        self.val_mesh_freq = 10000
        self.report_freq = 1000

        self.igr_weight = 0.1
        self.mask_weight = 0.1

class Model_Params():
    def __init__(self):
        super(Model_Params, self).__init__()

    def nerf_params(self):
        nerf_conf = ConfigTree()
        
        nerf_conf.put("D", 8)
        nerf_conf.put("d_in", 4)
        nerf_conf.put("d_in_view", 3)
        nerf_conf.put("W", 256),
        nerf_conf.put("multires", 10)
        nerf_conf.put("multires_view", 4)
        nerf_conf.put("output_ch", 4)
        nerf_conf.put("skips", [4])
        nerf_conf.put("use_viewdirs", True)

        return nerf_conf
    
    def sdf_network_params(self):
        sdf_net_conf = ConfigTree()

        sdf_net_conf.put("d_out", 257)
        sdf_net_conf.put("d_in", 3)
        sdf_net_conf.put("d_hidden", 256)
        sdf_net_conf.put("n_layers", 8)
        sdf_net_conf.put("skip_in", [4])
        sdf_net_conf.put("multires", 6)
        sdf_net_conf.put("bias", 0.5)
        sdf_net_conf.put("scale", 1.0)
        sdf_net_conf.put("geometric_init", True)
        sdf_net_conf.put("weight_norm", True)

        return sdf_net_conf
        
    
    def variance_network_params(self):
        var_net_conf = ConfigTree()
        var_net_conf.put("init_val", 0.3)

        return var_net_conf
    
    def rendering_network_params(self):
        render_net_conf = ConfigTree()
        
        render_net_conf.put("d_feature", 256)
        render_net_conf.put("mode", "idr")
        render_net_conf.put("d_in", 9)
        render_net_conf.put("d_out", 3)
        render_net_conf.put("d_hidden", 256)
        render_net_conf.put("n_layers", 4)
        render_net_conf.put("weight_norm", True)
        render_net_conf.put("multires_view", 4)
        render_net_conf.put("squeeze_out", True)

        return render_net_conf
    
    def neus_renderer_params(self):
        neus_conf = ConfigTree()
        
        neus_conf.put("n_samples", 64)
        neus_conf.put("n_importance", 64)
        neus_conf.put("n_outside", 0)
        neus_conf.put("up_sample_steps", 4)
        neus_conf.put("perturb", 1.0)

        return neus_conf