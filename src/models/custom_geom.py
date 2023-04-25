# !/usr/bin/python3
# coding: utf-8
# author: lixiang

import torch
import xatlas
import numpy as np

from ..render import mesh
from ..render import render
from ..render import regularizer
from ..render import material
from ..render import texture
from ..render import mlptexture

###############################################################################
# UV - map geometry & convert to a mesh
###############################################################################
@torch.no_grad()
def xatlas_uvmap(glctx, geometry, mat, FLAGS):
    
    m_verts, m_faces = geometry.InitMesh()
    v_pos = m_verts.detach().cpu().numpy()
    t_pos_idx = m_faces.detach().cpu().numpy()
    
    vmapping, indices, uvs = xatlas.parametrize(v_pos, t_pos_idx)
    indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)
    
    uvs = torch.tensor(uvs, dtype=torch.float32, device='cuda')
    uvs_idx = torch.tensor(indices_int64, dtype=torch.int64, device='cuda')

    new_mesh = mesh.Mesh(m_verts, m_faces, v_tex=uvs, t_tex_idx=uvs_idx, material=mat)
    
    mask, kd, ks, normal = render.render_uv(glctx, new_mesh, FLAGS.texture_res, mat['kd_ks_normal'])
    
    if FLAGS.layers > 1:
        kd = torch.cat((kd, torch.rand_like(kd[...,0:1])), dim=-1)

    kd_min, kd_max = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda')
    ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')
    nrm_min, nrm_max = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device='cuda')

    new_mesh.material = material.Material({
        'bsdf'   : mat['bsdf'],
        'kd'     : texture.Texture2D(kd, min_max=[kd_min, kd_max]),
        'ks'     : texture.Texture2D(ks, min_max=[ks_min, ks_max]),
        'normal' : texture.Texture2D(normal, min_max=[nrm_min, nrm_max])
    })

    return new_mesh
###############################################################################


###############################################################################
# Utility functions for material
###############################################################################
def initial_guess_material(geometry, FLAGS):
    kd_min, kd_max = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda')
    ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')
    nrm_min, nrm_max = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device='cuda')
    
    mlp_min = torch.cat((kd_min[0:3], ks_min, nrm_min), dim=0)
    mlp_max = torch.cat((kd_max[0:3], ks_max, nrm_max), dim=0)
    mlp_map_opt = mlptexture.MLPTexture3D(geometry.getAABB(), channels=9, min_max=[mlp_min, mlp_max])
    mat =  material.Material({'kd_ks_normal' : mlp_map_opt})
    mat['bsdf'] = 'pbr'

    return mat
###############################################################################

class Init_Geom():
    def __init__(self, base_mesh):
        super(Init_Geom, self).__init__()

        self.verts = torch.tensor(base_mesh.vertices, dtype=torch.float32, device='cuda')
        self.faces = torch.tensor(base_mesh.faces, dtype=torch.long, device='cuda')
    
    @torch.no_grad()
    def getAABB(self):
        return torch.min(self.verts, dim=0).values, torch.max(self.verts, dim=0).values
    
    def InitMesh(self):
        return self.verts, self.faces
        

class CustomMesh():
    def __init__(self, initial_guess, FLAGS):
        super(CustomMesh, self).__init__()

        self.FLAGS = FLAGS

        self.initial_guess = initial_guess
        self.mesh          = initial_guess.clone()
        print("=> Base mesh has %d triangles and %d vertices." % (self.mesh.t_pos_idx.shape[0], self.mesh.v_pos.shape[0]))

    @torch.no_grad()
    def getAABB(self):
        return mesh.aabb(self.mesh)

    def getMesh(self, material):
        self.mesh.material = material
        imesh = mesh.Mesh(base=self.mesh)
        
        # Compute normals and tangent space
        imesh = mesh.auto_normals(imesh)
        imesh = mesh.compute_tangents(imesh)
        return imesh

    def render(self, glctx, target, lgt, opt_material, bsdf=None):
        opt_mesh = self.getMesh(opt_material)
        return render.render_mesh(glctx, opt_mesh, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], 
                                    num_layers=self.FLAGS.layers, msaa=True, background=target['background'], bsdf=bsdf)

    def tick(self, glctx, target, lgt, opt_material, loss_fn, iteration):
        
        buffers = self.render(glctx, target, lgt, opt_material)
        # t_iter = iteration / self.FLAGS.iter

        # Image-space loss, split into a coverage component and a color component
        color_ref = target['img']
        img_loss = torch.nn.functional.mse_loss(buffers['shaded'][..., 3:], color_ref[..., 3:]) 
        img_loss += loss_fn(buffers['shaded'][..., 0:3] * color_ref[..., 3:], color_ref[..., 0:3] * color_ref[..., 3:])

        reg_loss = torch.tensor([0], dtype=torch.float32, device="cuda")        

        # Albedo (k_d) smoothnesss regularizer
        # reg_loss += torch.mean(buffers['kd_grad'][..., :-1] * buffers['kd_grad'][..., -1:]) * 0.03 * min(1.0, iteration / 500)
        reg_loss += torch.mean(buffers['kd_grad'][..., :-1] * buffers['kd_grad'][..., -1:]) * 0.003 * min(1.0, iteration / 500)

        # Visibility regularizer
        #reg_loss += torch.mean(buffers['occlusion'][..., :-1] * buffers['occlusion'][..., -1:]) * 0.001 * min(1.0, iteration / 500)

        # Light white balance regularizer
        # reg_loss = reg_loss + lgt.regularizer() * 0.01
        reg_loss = reg_loss + lgt.regularizer() * 0.03

        return img_loss, reg_loss
