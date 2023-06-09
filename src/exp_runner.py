# !/usr/bin/python3
# coding: utf-8
# author: lixiang
# data: 2022-10-13

import os
import logging
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from tqdm import tqdm
from .models.dataset import Dataset
from .models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from .models.renderer import NeuSRenderer
from .models.texture_v1 import train_texture
from .dataset.imgs2npz import imgs2poses

from .configs.wmask_conf import Train_Params, Model_Params


class Runner:
    def __init__(self, processed_data_dir, save_dir, mode='train', is_continue=False, checkpoint=None):
        self.device = torch.device('cuda')
        self.processed_data_dir = processed_data_dir
        self.save_dir = save_dir
        self.checkpoint = checkpoint
        self.train_params = Train_Params()
        self.model_params = Model_Params()

        # Configuration
        self.nerf_conf = self.model_params.nerf_params()
        self.sdf_net_conf = self.model_params.sdf_network_params()
        self.var_net_conf = self.model_params.variance_network_params()
        self.render_net_conf = self.model_params.rendering_network_params()
        self.neus_render_conf = self.model_params.neus_renderer_params()

        self.base_exp_dir = os.path.join(self.save_dir, 'Exp')
        if not os.path.exists(self.base_exp_dir):
            os.makedirs(self.base_exp_dir, exist_ok=True)

        self.dataset = Dataset(self.processed_data_dir)
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.train_params.end_iter
        self.save_freq = self.train_params.save_freq
        self.report_freq = self.train_params.report_freq
        self.val_freq = self.train_params.val_freq
        self.val_mesh_freq = self.train_params.val_mesh_freq
        self.batch_size = self.train_params.batch_size
        self.validate_resolution_level = self.train_params.validate_resolution_level
        self.learning_rate = self.train_params.learning_rate
        self.learning_rate_alpha = self.train_params.learning_rate_alpha
        self.use_white_bkgd = self.train_params.use_white_bkgd
        self.warm_up_end = self.train_params.warm_up_end
        self.anneal_end = self.train_params.anneal_end


        # Weights
        self.igr_weight = self.train_params.igr_weight
        self.mask_weight = self.train_params.mask_weight
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.nerf_outside = NeRF(**self.nerf_conf).to(self.device)
        self.sdf_network = SDFNetwork(**self.sdf_net_conf).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.var_net_conf).to(self.device)
        self.color_network = RenderingNetwork(**self.render_net_conf).to(self.device)
        
        print('=> Print Network Structure ...')
        print('-' * 50)
        #x_sdf = torch.rand(39, 3)
        #x_nef = torch.rand(84, 3)
        x_red = torch.rand(289, 3)
        #torch.onnx.export(self.sdf_network, x_sdf, "sdf.onnx")
        #torch.onnx.export(self.nerf_outside, x_sdf, x_nef, "nerf.onnx")
        torch.onnx.export(self.color_network, x_red, "render.onnx")
        print('-' * 50)

        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     **self.neus_render_conf)

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]
            #print("model_name:", latest_model_name)
        
        if self.checkpoint is not None:
            latest_model_name = self.checkpoint.split('/')[-1]
    
        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        # if self.mode[:5] == 'train':
        #     self.file_backup()

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()

        for iter_i in tqdm(range(res_step)):
            data = self.dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)], self.batch_size)
            rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3])

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)

            mask_sum = mask.sum() + 1e-5
            render_out = self.renderer.render(rays_o, rays_d, near, far,
                                              background_rgb=background_rgb,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio())

            color_fine = render_out['color_fine']
            s_val = render_out['s_val']
            cdf_fine = render_out['cdf_fine']
            gradient_error = render_out['gradient_error']
            weight_max = render_out['weight_max']
            weight_sum = render_out['weight_sum']

            # Loss
            color_error = (color_fine - true_rgb) * mask
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
            psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

            eikonal_loss = gradient_error

            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

            loss = color_fine_loss +\
                   eikonal_loss * self.igr_weight +\
                   mask_loss * self.mask_weight

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1

            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
            self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            # if self.iter_step % self.val_freq == 0:
            #     self.validate_image()

            # if self.iter_step % self.val_mesh_freq == 0:
            #     self.validate_mesh()

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    # def file_backup(self):
    #     dir_lis = self.conf['general.recording']
    #     os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
    #     for dir_name in dir_lis:
    #         cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
    #         os.makedirs(cur_dir, exist_ok=True)
    #         files = os.listdir(dir_name)
    #         for f_name in files:
    #             if f_name[-3:] == '.py':
    #                 copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

    #     copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_image(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            # del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           np.concatenate([img_fine[..., i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)]))
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           normal_img[..., i])

    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            # del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        return img_fine

    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        
        os.makedirs(os.path.join(self.base_exp_dir, 'init_meshes'), exist_ok=True)
        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        export_path = os.path.join(self.base_exp_dir, 'init_meshes', '{:0>6d}.obj'.format(self.iter_step))

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(export_path)

        logging.info('End')

    def interpolate_view(self, img_idx_0, img_idx_1):
        images = []
        n_frames = 60
        for i in range(n_frames):
            print(i)
            images.append(self.render_novel_image(img_idx_0,
                                                  img_idx_1,
                                                  np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                          resolution_level=4))
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(os.path.join(video_dir,
                                             '{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)),
                                fourcc, 30, (w, h))

        for image in images:
            writer.write(image)

        writer.release()


class AutoReconstructPackage():
    def __init__(self, root_dir, out_dir, checkpoint=None):
        super(AutoReconstructPackage, self).__init__()

        self.root_dir            = root_dir
        self.out_dir             = out_dir
        self.checkpoint          = checkpoint
        
        self.match_type          = 'exhaustive_matcher'         # type of matcher used. Valid options: [exhaustive_matcher sequential_matcher]
        self.mode                = 'train'
        
        self.mcube_threshold     = 0.0  
        self.is_continue         = True

        self.processed_out_dir   = os.path.join(self.root_dir, 'preprocessed')
    
    def find_newest_obj(self, m_folder):
        return max([os.path.join(m_folder, d) for d in os.listdir(m_folder)], key=os.path.getmtime)
    
    def gen_poses(self):                                                        
        imgs2poses(self.root_dir, self.match_type)

    def mesh_train(self):
        if self.mode == 'train':
            runner = Runner(self.processed_out_dir, self.out_dir, self.mode, False)     
            runner.train()
    
    def mesh_extract(self):                                  
        if self.mode == 'validate_mesh':
            runner = Runner(self.processed_out_dir, self.out_dir, self.mode, self.is_continue, self.checkpoint)
            runner.validate_mesh(world_space=True, resolution=512, threshold=self.mcube_threshold)

    def gen_texture(self):                                   
        #base_mesh = os.path.join(self.processed_out_dir, 'meshes', '{}.obj'.format(self.case))
        base_mesh = self.find_newest_obj(os.path.join(self.out_dir, 'Exp', 'init_meshes'))
        train_texture(base_mesh, self.processed_out_dir, self.out_dir)
    
    def auto_reconstruct(self):
        ## step 01
        self.gen_poses()
        ## step 02
        self.mode = 'train'
        self.mesh_train()
        ## step 03
        self.mode = 'validate_mesh'
        self.mesh_extract()
        ## step 04
        self.gen_texture()
