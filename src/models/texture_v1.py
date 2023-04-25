import os
import time
from tkinter import N

import numpy as np
import torch
import nvdiffrast.torch as dr
import trimesh

from ..configs.texture_conf import Texture_Hyparams
from ..dataset.dataset_llff import DatasetLLFF, _resize_data
from ..dataset.dataset_nerf import DatasetNERF
from ..dataset.dataset_nsrpl import DatasetNSRPL
from ..render import renderutils as ru
from ..render import obj
from ..render import util
from ..render import light

from .custom_geom import Init_Geom, CustomMesh, initial_guess_material, xatlas_uvmap

###########################################################################################
# Loss setup
###########################################################################################
@torch.no_grad()
def createLoss(FLAGS):
    if FLAGS.loss == "smape":
        return lambda img, ref: ru.image_loss(img, ref, loss='smape', tonemapper='none')
    elif FLAGS.loss == "mse":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='none')
    elif FLAGS.loss == "logl1":
        return lambda img, ref: ru.image_loss(img, ref, loss='l1', tonemapper='log_srgb')
    elif FLAGS.loss == "logl2":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='log_srgb')
    elif FLAGS.loss == "relmse":
        return lambda img, ref: ru.image_loss(img, ref, loss='relmse', tonemapper='none')
    else:
        assert False

###########################################################################################
# Mix background into a dataset image
###########################################################################################
@torch.no_grad()
def prepare_batch(target, bg_type='black'):
    assert len(target['img'].shape) == 4, "Image shape should be [n, h, w, c]"
    if bg_type == 'checker':
        background = torch.tensor(util.checkerboard(target['img'].shape[1:3], 8), dtype=torch.float32, device='cuda')[None, ...]
    elif bg_type == 'black':
        background = torch.zeros(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    elif bg_type == 'white':
        background = torch.ones(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    elif bg_type == 'reference':
        background = target['img'][..., 0:3]
    elif bg_type == 'random':
        background = torch.rand(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    else:
        assert False, "Unknown background type %s" % bg_type
    target['mv'] = target['mv'].cuda()
    target['mvp'] = target['mvp'].cuda()
    target['campos'] = target['campos'].cuda()
    target['img'] = target['img'].cuda()
    target['background'] = background
    target['img'] = torch.cat((torch.lerp(background, target['img'][..., 0:3], target['img'][..., 3:4]), target['img'][..., 3:4]), dim=-1)

    return target

###########################################################################################
# Validation & testing
###########################################################################################
def validate_itr(glctx, target, geometry, opt_material, lgt, FLAGS):
    result_dict = {}
    with torch.no_grad():
        lgt.build_mips()
        if FLAGS.camera_space_light:
            lgt.xfm(target['mv'])

        buffers = geometry.render(glctx, target, lgt, opt_material)

        result_dict['ref'] = util.rgb_to_srgb(target['img'][...,0:3])[0]
        result_dict['opt'] = util.rgb_to_srgb(buffers['shaded'][...,0:3])[0]
        result_image = torch.cat([result_dict['opt'], result_dict['ref']], axis=1)

        if FLAGS.display is not None:
            white_bg = torch.ones_like(target['background'])
            for layer in FLAGS.display:
                if 'latlong' in layer and layer['latlong']:
                    if isinstance(lgt, light.EnvironmentLight):
                        result_dict['light_image'] = util.cubemap_to_latlong(lgt.base, FLAGS.display_res)
                    result_image = torch.cat([result_image, result_dict['light_image']], axis=1)
                elif 'relight' in layer:
                    if not isinstance(layer['relight'], light.EnvironmentLight):
                        layer['relight'] = light.load_env(layer['relight'])
                    img = geometry.render(glctx, target, layer['relight'], opt_material)
                    result_dict['relight'] = util.rgb_to_srgb(img[..., 0:3])[0]
                    result_image = torch.cat([result_image, result_dict['relight']], axis=1)
                elif 'bsdf' in layer:
                    buffers = geometry.render(glctx, target, lgt, opt_material, bsdf=layer['bsdf'])
                    if layer['bsdf'] == 'kd':
                        result_dict[layer['bsdf']] = util.rgb_to_srgb(buffers['shaded'][0, ..., 0:3])  
                    elif layer['bsdf'] == 'normal':
                        result_dict[layer['bsdf']] = (buffers['shaded'][0, ..., 0:3] + 1) * 0.5
                    else:
                        result_dict[layer['bsdf']] = buffers['shaded'][0, ..., 0:3]
                    result_image = torch.cat([result_image, result_dict[layer['bsdf']]], axis=1)
   
        return result_image, result_dict

def validate(glctx, geometry, opt_material, lgt, dataset_validate, out_dir, FLAGS):
    mse_values = []
    psnr_values = []
    dataloader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size=1, collate_fn=dataset_validate.collate)
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, 'metrics.txt'), 'w') as fout:
        fout.write('ID, MSE, PSNR\n')
        print("Running validation")
        for it, target in enumerate(dataloader_validate):
            # Mix validation background
            target = prepare_batch(target, FLAGS.background)
            result_image, result_dict = validate_itr(glctx, target, geometry, opt_material, lgt, FLAGS)
           
            # Compute metrics
            opt = torch.clamp(result_dict['opt'], 0.0, 1.0) 
            ref = torch.clamp(result_dict['ref'], 0.0, 1.0)

            mse = torch.nn.functional.mse_loss(opt, ref, size_average=None, reduce=None, reduction='mean').item()
            mse_values.append(float(mse))
            psnr = util.mse_to_psnr(mse)
            psnr_values.append(float(psnr))

            line = "%d, %1.8f, %1.8f\n" % (it, mse, psnr)
            fout.write(str(line))

            for k in result_dict.keys():
                np_img = result_dict[k].detach().cpu().numpy()
                util.save_image(out_dir + '/' + ('val_%03d_%s.png' % (it, k)), np_img)

        avg_mse = np.mean(np.array(mse_values))
        avg_psnr = np.mean(np.array(psnr_values))
        line = "AVERAGES: %1.4f, %2.3f\n" % (avg_mse, avg_psnr)
        fout.write(str(line))
        print("MSE,      PSNR")
        print("%1.8f, %2.3f" % (avg_mse, avg_psnr))
    
    return avg_psnr

###########################################################################################
# Main shape fitter function / optimization loop
###########################################################################################
class Trainer(torch.nn.Module):
    def __init__(self, glctx, geometry, lgt, mat, optimize_light, image_loss_fn, FLAGS):
        super(Trainer, self).__init__()
        
        self.glctx = glctx
        self.geometry = geometry
        self.light = lgt
        self.material = mat
        self.optimize_light = optimize_light
        self.image_loss_fn = image_loss_fn
        self.FLAGS = FLAGS

        if not self.optimize_light:
            with torch.no_grad():
                self.light.build_mips()

        self.params = list(self.material.parameters())
        self.params += list(self.light.parameters()) if optimize_light else []

    def forward(self, target, it):
        if self.optimize_light:
            self.light.build_mips()
            if self.FLAGS.camera_space_light:
                self.light.xfm(target['mv'])

        return self.geometry.tick(self.glctx, target, self.light, self.material, self.image_loss_fn, it)

def optimize_mesh(
        glctx,
        geometry,
        opt_material,
        lgt,
        dataset_train,
        FLAGS,
        warmup_iter=0,
        log_interval=10,
        pass_idx=0,
        optimize_light=True,
    ):

    learning_rate = FLAGS.learning_rate[pass_idx] if isinstance(FLAGS.learning_rate, list) or isinstance(FLAGS.learning_rate, tuple) else FLAGS.learning_rate
    learning_rate_mat = learning_rate[1] if isinstance(learning_rate, list) or isinstance(learning_rate, tuple) else learning_rate

    def lr_schedule(iter, fraction):
        if iter < warmup_iter:
            return iter / warmup_iter 
        return max(0.0, 10**(-(iter - warmup_iter)*0.0002)) # Exponential falloff from [1.0, 0.1] over 5k epochs.    

    
    ## Image loss
    image_loss_fn = createLoss(FLAGS)
    trainer_noddp = Trainer(glctx, geometry, lgt, opt_material, optimize_light, image_loss_fn, FLAGS)

    ## Single GPU training mode
    trainer = trainer_noddp

    optimizer = torch.optim.Adam(trainer_noddp.params, lr=learning_rate_mat)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_schedule(x, 0.9)) 

    ## Training loop
    #img_cnt = 0
    img_loss_vec = []
    reg_loss_vec = []
    iter_dur_vec = []

    dataloader_train    = torch.utils.data.DataLoader(dataset_train, batch_size=FLAGS.batch, collate_fn=dataset_train.collate, shuffle=True)
    for it, target in enumerate(dataloader_train):
        # Mix randomized background into dataset image
        target = prepare_batch(target, 'random')
        iter_start_time = time.time()
        optimizer.zero_grad()

        img_loss, reg_loss = trainer(target, it)
        total_loss = img_loss + reg_loss

        img_loss_vec.append(img_loss.item())
        reg_loss_vec.append(reg_loss.item())

        total_loss.backward()
        if hasattr(lgt, 'base') and lgt.base.grad is not None and optimize_light:
            lgt.base.grad *= 64
        if 'kd_ks_normal' in opt_material:
            opt_material['kd_ks_normal'].encoder.params.grad /= 8.0

        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            if 'kd' in opt_material:
                opt_material['kd'].clamp_()
            if 'ks' in opt_material:
                opt_material['ks'].clamp_()
            if 'normal' in opt_material:
                opt_material['normal'].clamp_()
                opt_material['normal'].normalize_()
            if lgt is not None:
                lgt.clamp_(min=0.0)

        torch.cuda.current_stream().synchronize()
        iter_dur_vec.append(time.time() - iter_start_time)

        if it % log_interval == 0 and FLAGS.local_rank == 0:
            img_loss_avg = np.mean(np.asarray(img_loss_vec[-log_interval:]))
            reg_loss_avg = np.mean(np.asarray(reg_loss_vec[-log_interval:]))
            iter_dur_avg = np.mean(np.asarray(iter_dur_vec[-log_interval:]))
            
            remaining_time = (FLAGS.iter-it)*iter_dur_avg
            print("iter=%5d, img_loss=%.6f, reg_loss=%.6f, lr=%.5f, time=%.1f ms, rem=%s" % 
                (it, img_loss_avg, reg_loss_avg, optimizer.param_groups[0]['lr'], iter_dur_avg*1000, util.time_to_text(remaining_time)))

    return geometry, opt_material


###########################################################################################
# -- Main function.
###########################################################################################
def train_texture(base_mesh, ref_mesh, out_dir):

    FLAGS = Texture_Hyparams(base_mesh, ref_mesh, out_dir)
    if FLAGS.display_res is None:
        FLAGS.display_res = FLAGS.train_res

    tex3D_results = os.path.join(FLAGS.out_dir, "rebuild_results")
    if not os.path.exists(tex3D_results):
        os.makedirs(tex3D_results, exist_ok=True)
    glctx = dr.RasterizeGLContext()

    if os.path.isfile(os.path.join(FLAGS.ref_mesh, 'poses_bounds.npy')):
        # images and masks resize
        print("=> Start data resize ...")
        _resize_data(FLAGS.ref_mesh, FLAGS.train_res)
        dataset_train    = DatasetNSRPL(FLAGS.ref_mesh, FLAGS, examples=(FLAGS.iter+1)*FLAGS.batch)
        dataset_validate = DatasetNSRPL(FLAGS.ref_mesh, FLAGS)
    elif os.path.isfile(os.path.join(FLAGS.ref_mesh, 'transforms_train.json')):
        # images and masks resize
        #print("=> Start data resize ...")
        #_resize_data(FLAGS.ref_mesh, FLAGS.train_res)
        dataset_train    = DatasetNERF(os.path.join(FLAGS.ref_mesh, 'transforms_train.json'), FLAGS, examples=(FLAGS.iter+1)*FLAGS.batch)
        dataset_validate = DatasetNERF(os.path.join(FLAGS.ref_mesh, 'transforms_test.json'), FLAGS)
    

    ## Create env light with trainable parameters
    if FLAGS.learn_light:
        lgt = light.create_trainable_env_rnd(512, scale=0.0, bias=0.5)
    else:
        lgt = light.load_env(FLAGS.envmap, scale=FLAGS.env_scale)

    # initializate geometry by input mesh
    init_geometry = Init_Geom(trimesh.load(FLAGS.base_mesh))

    # Setup textures, make initial guess from reference if possible
    mat = initial_guess_material(init_geometry, FLAGS)

    # Create textured mesh from result
    print('=> Initialize the uvs by xatlas...')
    new_mesh = xatlas_uvmap(glctx, init_geometry, mat, FLAGS)
        
    # Free temporaries / cached memory 
    torch.cuda.empty_cache()
    mat['kd_ks_normal'].cleanup()
    del mat['kd_ks_normal']

    lgt = lgt.clone()
    geometry = CustomMesh(new_mesh, FLAGS)
    geometry, mat = optimize_mesh(glctx, geometry, new_mesh.material, lgt, dataset_train, FLAGS, 
                pass_idx=0, optimize_light=FLAGS.learn_light)

    ## Dump output
    final_mesh = geometry.getMesh(mat)
    os.makedirs(os.path.join(tex3D_results, "mesh"), exist_ok=True)
    obj.write_obj(os.path.join(tex3D_results, "mesh/"), final_mesh)
    light.save_env_map(os.path.join(tex3D_results, "mesh/probe.hdr"), lgt)
    
    ## Validate
    if FLAGS.validate and FLAGS.local_rank == 0:
        validate(glctx, geometry, mat, lgt, dataset_validate, os.path.join(tex3D_results, "render_images"), FLAGS)
###########################################################################################
