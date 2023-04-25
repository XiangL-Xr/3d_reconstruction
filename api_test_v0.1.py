# !/usr/bin/python3
# coding: utf-8
# author: lixiang
# data: 2022-10-13

import os
import time
import subprocess
import zipfile
import threading
import requests

import src.exp_runner as Exp_Runner
from flask import Flask, request, send_from_directory, make_response, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

## 设置上传数据保存文件夹,即多视图三维重建项目加载数据路径
app.config['UPLOAD_FOLDER'] = './data/'

## 设置三维重建项目输出结果存放路径
app.config['RESULTS_FOLDER'] = './Final_out/'

## 设置请求接口地址
app.config['SAVE_LOG_URL'] = "http://172.18.16.107:91/api/technology/savelog"      # 时间日志打点

## 设置允许上传的文件格式
ALLOW_EXTENSIONS = ['zip',]

## 判断文件后缀是否在列表中
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[-1] in ALLOW_EXTENSIONS

## 检查static/目录, 不存在则创建, 存在则清空
def check_folder(folder):
    if os.path.exists(folder) and len(os.listdir(folder)) > 0:
        FNULL = open(os.devnull, 'w')
        subprocess.call(f"rm {folder}/*.*", shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
    else:
        os.makedirs(folder, exist_ok=True)

## 将上传的图片移动到指定目录下
def run_copy(src_img, dst_img):
    strcmd = 'cp ' + src_img + ' ' + dst_img
    subprocess.call(strcmd, shell=True)

## 打包zip文件
def make_zip(filepath, source_dir):
    zipf = zipfile.ZipFile(source_dir, 'w')
    pre_len = len(os.path.dirname(filepath))
    for parent, dirnames, filenames in os.walk(filepath):
        for filename in filenames:
            pathfile = os.path.join(parent, filename)
            arcname = pathfile[pre_len:].strip(os.path.sep)
            zipf.write(pathfile, arcname)
    
    zipf.close()

## 解压zip文件
def ext_zip(filepath, dst_dir):
    zipf = zipfile.ZipFile(filepath, 'r')
    for file in zipf.namelist():
        zipf.extract(file, dst_dir)
    zipf.close()


## 查找指定目录中最新的文件夹
def find_newest_folder(m_folder):
    return max([os.path.join(m_folder, d) for d in os.listdir(m_folder)], key=os.path.getmtime)

## 日志打点回调函数
def log_callback(uuid, label):
    processing_time = time.strftime('%Y-%m-%d %H:%M:%S')
    processing_json = {
        "uuid": uuid,
        "data": [{
            "remark": str(label),
            "log_time": processing_time
        }]
    }
    processing_res = requests.post(app.config['SAVE_LOG_URL'], json=processing_json)
    print('=> %s, 请求结果:'%(label), processing_res.text)


@app.route('/')
def index():
    return make_response(render_template('index_v0.1.html'))

###########################################################################################################
### -- 自动化三维重建接口，通过多线程实现异步执行
### -------------------------------------------------------------------------------------------------------
@app.route('/api/auto_reconstruction', methods=['POST', 'GET'])
def auto_reconstruct():
    if request.method == 'POST':
        print('=' * 90)  
        ## 接收'POST'请求的参数
        file_data  = request.files.get('file')
        init_state = request.form.get('init_state')
        file_name  = file_data.filename
        task_uuid  = "60cbb303d247e957564751d25a4749f6"
        #date_name  = time.strftime('%Y-%m-%d')
        date_name  = "2022-10-25"
        
        ## 设置重建结果存放路径
        out_folder = os.path.join(app.config['RESULTS_FOLDER'], date_name, "my_results")

        ## 设置数据集加载路径
        data_folder = os.path.join(app.config['UPLOAD_FOLDER'], date_name)
        if not os.path.exists(data_folder):
            os.makedirs(data_folder, exist_ok=True)

        ## 将加载的数据解压到指定目录下
        # log_callback(task_uuid, "开始上传数据")
        if file_data and allowed_file(file_name):
            ext_zip(file_data, data_folder)                             ## 将上传的压缩包文件解压到数据集加载目录下
        else:
            return {"code": '404', "data": "NOT FOUND", "message": "The requested zipfile does not upload!"}
        
        data_folder = os.path.join(data_folder, file_name.split('.')[0])
        
        ## 引入3D重建包
        exp_runner = Exp_Runner.AutoReconstructPackage(data_folder, out_folder)

        ## =========================================================================================
        ## -- 定义子线程函数reconstruct_work()，实现自动化三维重建的功能，待数据上传成功后，自动执行此线程
        ## -- 进行自动化三维重建，可通过指定init_state参数选择重启的起始阶段
        ## -----------------------------------------------------------------------------------------
        def reconstruct_work(): 
            print('-' * 80)                                                 
            # log_callback(task_uuid, "开始3D建模")

            ## init_state不为空时，代表从指定阶段重新启动3D重建过程, 否则进行自动化三维重建
            if len(init_state) > 0: 
                reconstruct_restart(init_state, task_uuid, exp_runner)
            else:                    
                auto_3Drebuild(task_uuid, exp_runner)    

            ## reconstruction end -----------------------------
            # log_callback(task_uuid, "3D建模完成")
            print('=' * 90)
            
            ### 最终重建结果的存储目录，可通过post请求进行返回
            callback_folder = os.path.join(out_folder, 'rebuild_results')

        ## ==================================================================================== end!

        if len(os.listdir(data_folder)) > 0:
            thread = threading.Thread(name='t0', target=reconstruct_work)
            thread.start()
            return {"code": '200', "data": "OK", "message": "{} upload successful!".format(file_name)}
        else:
            return {"code": '404', "data": "NOT FOUND", "message": "{} upload failed!".format(file_name)}

    else:
        return {"code": '503', "data": "NOT SUPPORT", "message": "only support post method!"}
###########################################################################################################


########################################################################################################### 
### -- 此函数用于重建中断时，从指定阶段重启3D重建过程
### -------------------------------------------------------------------------------------------------------
def reconstruct_restart(init_state, uuid, exp_runner):
    if init_state == 'train':
        ## step 02 -----------------------------
        # log_callback(uuid, "开始模型训练")
        exp_runner.mode = 'train'
        exp_runner.mesh_train()
                    
        ## step 03 -----------------------------
        # log_callback(uuid, "开始mesh提取")
        exp_runner.mode = 'validate_mesh'
        exp_runner.mesh_extract()
                    
        ## step 04 -----------------------------
        # log_callback(uuid, "开始贴图生成")
        exp_runner.gen_texture()

    ## 初始阶段为extract, 设置模式(mode=='validate_mesh')开始生成mesh网格模型及对应贴图
    elif init_state == 'extract':
        ## step 03 -----------------------------
        # log_callback(uuid, "开始mesh提取")
        exp_runner.mode = 'validate_mesh'
        exp_runner.mesh_extract()
                    
        ## step 04 -----------------------------
        # log_callback(uuid, "开始贴图生成")
        exp_runner.gen_texture()

    ## 初始阶段为texture, 开始生成mesh网格模型对应的贴图
    elif init_state == 'texture':
        ## step 04 -----------------------------
        # log_callback(uuid, "开始贴图生成")
        exp_runner.gen_texture()

    else:
        print('=> Init State Select Error!, Please select from [train, extract, texture].')    

########################################################################################################### 
### -- 自动化3D重建函数
### -------------------------------------------------------------------------------------------------------
def auto_3Drebuild(uuid, exp_runner):
    ## step 01 -----------------------------
    # log_callback(uuid, "开始colmap位姿估计")
    exp_runner.gen_poses()
                
    ## step 02 -----------------------------
    # log_callback(uuid, "开始模型训练")
    exp_runner.mode = 'train'
    exp_runner.mesh_train()
                
    ## step 03 -----------------------------
    # log_callback(uuid, "开始mesh提取")
    exp_runner.mode = 'validate_mesh'
    exp_runner.mesh_extract()
                
    ## step 04 -----------------------------
    # log_callback(uuid, "开始贴图生成")
    exp_runner.gen_texture()


###########################################################################################################
### -- 结果文件下载接口
###--------------------------------------------------------------------------------------------------------
@app.route('/api/auto_reconstruction/download_file', methods=['GET', 'POST'])
def download_file():
    if request.method == 'POST':
        download_folder = request.form.get('download_path')
        
        ## 定义zip压缩包存放路径
        zip_folder = download_folder.replace('rebuild_results', 'zip_folder')
        if not os.path.exists(zip_folder):
            os.makedirs(zip_folder, exist_ok=True)
        
        ## 将重建结果打包为zip文件，便于下载
        if len(os.listdir(download_folder)) > 0:
            zip_name = download_folder.split('/')[-1] + '.zip'
            make_zip(download_folder, os.path.join(zip_folder, zip_name))

        ## 判断待下载文件是否存在，存在则下载文件
        if not os.path.exists(os.path.join(zip_folder, zip_name)):
            return {"code": '404', "data": "NOT FOUND", "message": "The requested resource does not exist!"}     
        else:
            return make_response(send_from_directory(zip_folder, zip_name, as_attachment=True)) 
    
    else:
        return {"code": '503', "data": "NOT SUPPORT", "message": "only support post method!"}
###################################################################################################### END!

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10423, debug=True)
    #app.run(debug=True)