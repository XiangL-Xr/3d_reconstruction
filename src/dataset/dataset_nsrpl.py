# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import glob

import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from ..render import util
from .dataset import Dataset
from .poses_utils.colmap_read_model import read_cameras_binary, read_images_binary, read_points3d_binary

def _load_mask(fn):
    img = torch.tensor(util.load_image(fn), dtype=torch.float32)
    if len(img.shape) == 2:
        img = img[..., None].repeat(1, 1, 3)
    return img

def _load_img(fn):
    img = util.load_image_raw(fn)
    if img.dtype != np.float32: # LDR image
        img = torch.tensor(img / 255, dtype=torch.float32)
        img[..., 0:3] = util.srgb_to_rgb(img[..., 0:3])
    else:
        img = torch.tensor(img, dtype=torch.float32)
    return img

def _resize_data(base_dir, dst_size):
    imgs_list = glob.glob(os.path.join(base_dir, "images", "*.*"))
    mask_list = glob.glob(os.path.join(base_dir, "masks", "*.*"))
    
    imgs_resize_dir = os.path.join(base_dir, "resized", "images")
    mask_resize_dir = os.path.join(base_dir, "resized", "masks")
    if not os.path.exists(imgs_resize_dir):
        os.makedirs(imgs_resize_dir, exist_ok=False)
    if not os.path.exists(mask_resize_dir):
        os.makedirs(mask_resize_dir, exist_ok=False)
    
    for i_path in imgs_list:
        img = Image.open(i_path)
        resizeImg = img.resize(dst_size)
        resizeImg.save(os.path.join(imgs_resize_dir, os.path.basename(i_path)), quality=95)
    for m_path in mask_list:
        msk = Image.open(m_path)
        resizeMsk = msk.resize(dst_size)
        resizeMsk.save(os.path.join(mask_resize_dir, os.path.basename(m_path)), quality=95)
    
    print("=> Resize images and masks to {}".format(dst_size))
    print('-' * 60)

def get_center(pts):
    center = pts.mean(0)
    dis = (pts - center[None,:]).norm(p=2, dim=-1)
    mean, std = dis.mean(), dis.std()
    q25, q75 = torch.quantile(dis, 0.25), torch.quantile(dis, 0.75)
    valid = (dis > mean - 1.5 * std) & (dis < mean + 1.5 * std) & (dis > mean - (q75 - q25) * 1.5) & (dis < mean + (q75 - q25) * 1.5)
    pts = pts[valid]
    center = pts.mean(0)
    return center, pts


def normalize_poses(poses, pts):
    center, pts = get_center(pts)

    z = F.normalize((poses[...,3] - center).mean(0), dim=0)
    y_ = torch.as_tensor([z[1], -z[0], 0.])
    x = F.normalize(y_.cross(z), dim=0)
    y = z.cross(x)

    Rc = torch.stack([x, y, z], dim=1)
    tc = center.reshape(3, 1)

    R, t = Rc.T, -Rc.T @ tc

    poses_homo = torch.cat([poses, torch.as_tensor([[[0.,0.,0.,1.]]]).expand(poses.shape[0], -1, -1)], dim=1)
    inv_trans = torch.cat([torch.cat([R, t], dim=1), torch.as_tensor([[0.,0.,0.,1.]])], dim=0)

    poses_norm = (inv_trans @ poses_homo)[:,:3] # (N_images, 4, 4)
    scale = poses_norm[...,3].norm(p=2, dim=-1).min()
    poses_norm[...,3] /= scale

    pts = (inv_trans @ torch.cat([pts, torch.ones_like(pts[:,0:1])], dim=-1)[...,None])[:,:3,0]
    pts = pts / scale

    return poses_norm, pts


###############################################################################
# LLFF datasets (real world camera lightfields)
###############################################################################
class DatasetNSRPL(Dataset):
    def __init__(self, base_dir, FLAGS, examples=None):
        self.FLAGS = FLAGS
        #self.base_dir = base_dir
        self.examples = examples
        self.use_mask = True
        
        # Load camera poses
        # poses_bounds = np.load(os.path.join(base_dir, 'poses_bounds.npy'))
        self.root_dir = base_dir
        
        # self.base_dir = base_dir
        # Load resized images and masks
        self.base_dir = os.path.join(base_dir, "resized")

        # Enumerate all image files and get resolution
        all_img = [f for f in sorted(glob.glob(os.path.join(self.base_dir, "images", "*"))) if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg')]
        self.resolution = _load_img(all_img[0]).shape[0:2]

        self.steup()

        if self.FLAGS.local_rank == 0:
            print("Load DatasetNSRPL: %d images with shape [%d, %d]" % (len(all_img), self.resolution[0], self.resolution[1]))
            # print("DatasetLLFF: auto-centering at %s" % (center.cpu().numpy()))

    
    def steup(self):
        camdata = read_cameras_binary(os.path.join(self.root_dir, 'sparse/0/cameras.bin'))
        H = int(camdata[1].height)
        W = int(camdata[1].width)
        F = camdata[1].params[0]

        w, h = self.resolution
        assert round(W / w * h) == H
        self.w, self.h = w, h
        self.factor = w / W

        imdata = read_images_binary(os.path.join(self.root_dir, 'sparse/0/images.bin'))
        names  = [imdata[k].name for k in imdata]
        perms  = np.argsort(names)

        mask_dir = os.path.join(self.root_dir, 'masks')
        self.use_mask = os.path.exists(mask_dir) and self.use_mask
        
        all_c2w, self.all_images, self.all_fg_masks = [], [], []
        bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])
        for i, d in enumerate(imdata.values()):
            R = d.qvec2rotmat()
            t = d.tvec.reshape([3, 1])
            # c2w = torch.from_numpy(np.concatenate([np.concatenate([R.T, -R.T@t], axis=1), bottom], axis=0)).float() # modify by @lixiang
            c2w = np.concatenate([np.concatenate([R.T, -R.T@t], axis=1), bottom], axis=0)                             # modify by @lixiang
            # c2w[:,1:3] *= -1. # COLMAP => OpenGL   
            all_c2w.append(c2w)
            img_path = os.path.join(self.root_dir, 'images', d.name)
            img = Image.open(img_path)
            img = img.resize(self.resolution, Image.BICUBIC)
            img = TF.to_tensor(img).permute(1, 2, 0)[...,:3]
            if self.use_mask:
                mask_paths = [os.path.join(mask_dir, d.name), os.path.join(mask_dir, d.name[3:])]
                mask_paths = list(filter(os.path.exists, mask_paths))
                assert len(mask_paths) == 1
                mask = Image.open(mask_paths[0]).convert('L') # (H, W, 1)
                mask = mask.resize(self.resolution, Image.BICUBIC)
                mask = TF.to_tensor(mask)[0]
            else:
                mask = torch.ones_like(img[...,0])
            
            self.all_fg_masks.append(mask) # (h, w)
            self.all_images.append(img)
        
        # all_c2w = np.linalg.inv(np.stack(all_w2c, 0))
        self.all_c2w = torch.from_numpy(np.stack(all_c2w, 0)).float()

        pts3d = read_points3d_binary(os.path.join(self.root_dir, 'sparse/0/points3D.bin'))
        pts3d = torch.from_numpy(np.array([pts3d[k].xyz for k in pts3d])).float()                        # modify by @lixiang
        # pts3d = np.array([pts3d[k].xyz for k in pts3d])                                                # modify by @lixiang            
 
        self.all_c2w, pts3d = normalize_poses(self.all_c2w[:, :3, :4], pts3d)
        
        hwf   = np.array([H, W, F]).reshape([3, 1])
        
        poses = self.all_c2w[:, :3, :4].numpy().transpose([1, 2, 0])
        poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1, 1, poses.shape[-1]])], 1)
        poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :],
                                poses[:, 3:4, :], poses[:, 4:5, :]], 1)
        
        ## save and load poses
        poses_arr = []
        for i in perms:
            poses_arr.append(poses[..., i].ravel())
            
        i_poses = np.array(poses_arr).reshape([-1, 3, 5]).transpose([1, 2, 0])
        i_poses = np.concatenate([i_poses[:, 1:2, :], -i_poses[:, 0:1, :], i_poses[:, 2:, :]], 1)       # Taken from nerf, swizzles from LLFF to expected coordinate system
        i_poses = np.moveaxis(i_poses, -1, 0).astype(np.float32)
        
        lcol        = np.array([0, 0, 0, 1], dtype=np.float32)[None, None, :].repeat(i_poses.shape[0], 0)
        self.imvs   = torch.tensor(np.concatenate((i_poses[:, :, 0:4], lcol), axis=1), dtype=torch.float32)
        self.aspect = self.w / self.h
        self.fovy   = util.focal_length_to_fovy(i_poses[:, 2, 4], i_poses[:, 0, 4])
        
        self.preloaded_data = []
        for i in range(self.imvs.shape[0]):
            self.preloaded_data += [self._parse_frame(i)]

    def _parse_frame(self, idx):
        all_img  = [f for f in sorted(glob.glob(os.path.join(self.base_dir, "images", "*"))) if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg')]
        all_mask = [f for f in sorted(glob.glob(os.path.join(self.base_dir, "masks", "*"))) if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg')]
        assert len(all_img) == self.imvs.shape[0] and len(all_mask) == self.imvs.shape[0]

        # Load image+mask data
        img  = _load_img(all_img[idx])
        mask = _load_mask(all_mask[idx])
        img  = torch.cat((img, mask[..., 0:1]), dim=-1)
        # print('img shape: ', img.shape)
        # print('img value: ', img)

        # Setup transforms
        proj   = util.perspective(self.fovy[idx, ...], self.aspect, self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])
        mv     = torch.linalg.inv(self.imvs[idx, ...])
        campos = torch.linalg.inv(mv)[:3, 3]
        mvp    = proj @ mv

        return img[None, ...], mv[None, ...], mvp[None, ...], campos[None, ...] # Add batch dimension

    def __len__(self):
        return self.imvs.shape[0] if self.examples is None else self.examples

    def __getitem__(self, itr):
        if self.FLAGS.pre_load:
            img, mv, mvp, campos = self.preloaded_data[itr % self.imvs.shape[0]]
        else:
            img, mv, mvp, campos = self._parse_frame(itr % self.imvs.shape[0])

        return {
            'mv' : mv,
            'mvp' : mvp,
            'campos' : campos,
            'resolution' : self.resolution,
            'spp' : self.FLAGS.spp,
            'img' : img
        }
