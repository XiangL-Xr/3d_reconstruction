# !/usr/bin/python3
# coding: utf-8
# @Author: lixiang
# @Time  : 2022-10-31

import os,sys
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import time
import argparse

import src.exp_runner as Exp_Runner
from shutil import rmtree

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default=None,
                    help='input images directory')
parser.add_argument('--init_state', type=str, default='')

args = parser.parse_args()

## 设置多视图数据存放路径
ROOT_DIR = './data/'

## 设置三维重建项目输出结果存放路径
OUT_DIR = './Final_out/'

########################################################################################################### 
### -- 此函数用于重建中断时，从指定阶段重启3D重建过程
### -------------------------------------------------------------------------------------------------------
def reconstruct_restart(init_state, exp_runner):
    if init_state == 'train':
        ## step 02 -----------------------------
        #log_callback(uuid, "开始模型训练")
        exp_runner.mode = 'train'
        exp_runner.mesh_train()
                    
        ## step 03 -----------------------------
        #log_callback(uuid, "开始mesh提取")
        exp_runner.mode = 'validate_mesh'
        exp_runner.mesh_extract()
                    
        ## step 04 -----------------------------
        #log_callback(uuid, "开始贴图生成")
        exp_runner.gen_texture()

    ## 初始阶段为extract, 设置模式(mode=='validate_mesh')开始生成mesh网格模型及对应贴图
    elif init_state == 'extract':
        ## step 03 -----------------------------
        #log_callback(uuid, "开始mesh提取")
        exp_runner.mode = 'validate_mesh'
        exp_runner.mesh_extract()
                    
        ## step 04 -----------------------------
        #log_callback(uuid, "开始贴图生成")
        exp_runner.gen_texture()

    ## 初始阶段为texture, 开始生成mesh网格模型对应的贴图
    elif init_state == 'texture':
        ## step 04 -----------------------------
        #log_callback(uuid, "开始贴图生成")
        exp_runner.gen_texture()

    else:
        print('=> Init State Select Error!, Please select from [train, extract, texture].')    


########################################################################################################### 
### -- 自动化3D重建函数
### -------------------------------------------------------------------------------------------------------
def auto_3Drebuild(exp_runner):
    ## step 01 -----------------------------
    exp_runner.gen_poses()
                
    ## step 02 -----------------------------
    exp_runner.mode = 'train'
    exp_runner.mesh_train()
                
    ## step 03 -----------------------------
    exp_runner.mode = 'validate_mesh'
    exp_runner.mesh_extract()
                
    ## step 04 -----------------------------
    exp_runner.gen_texture()

###################################################################################################### END!

if __name__ == '__main__':
    
    print('=' * 90)
    data_name = args.data_name 
    init_state = args.init_state 
    data_time = time.strftime('%Y-%m')
        
    ## 设置数据集加载路径
    data_folder = os.path.join(ROOT_DIR, data_time, data_name)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder, exist_ok=False)
                   
    ## 设置重建结果存放路径
    out_folder = os.path.join(OUT_DIR, data_time, data_name)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder, exist_ok=False)
    
    ## 引入3D重建包
    exp_runner = Exp_Runner.AutoReconstructPackage(data_folder, out_folder)

    ## 进行自动化三维重建
    if len(init_state) > 0:
        reconstruct_restart(init_state, exp_runner)
    else: 
        auto_3Drebuild(exp_runner)

    print("=> 3D modeling End!!")
    print('=' * 90)