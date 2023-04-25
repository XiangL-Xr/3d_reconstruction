# !/usr/bin/python3
# coding: utf-8

import os
import subprocess

from distutils.core import setup
from Cython.Build import cythonize
from shutil import rmtree, copy


APP_NAME = "rebuild_pkg"
BUILD_DIR_NAME = "src"
GIT_UPLOAD_DIR = "3d_reconstruction_api"

PIP_REQUIREMENTS = "requirements.txt"

CPP_EXTENSION = "c_src"
_C_SO = "_C.cpython-38-x86_64-linux-gnu.so"

# 查看当前目录下是否存在build目录，如果存在则删掉
if os.path.isdir("build"):
    rmtree("build")

if os.path.isdir(f"{GIT_UPLOAD_DIR}/{BUILD_DIR_NAME}"):
    rmtree(f"{GIT_UPLOAD_DIR}/{BUILD_DIR_NAME}")

def folder_copy(src_dir, dst_dir):
    strcmd = 'cp -r ' + src_dir + ' ' + dst_dir
    subprocess.call(strcmd, shell=True)

def load_files():
    file_path_list = []
    for a, _, c in os.walk(BUILD_DIR_NAME):
        if len(c) == 0: 
            continue

        if "_del" in a:     # 文件夹名称中带有 “_del” 的不进行转化
            continue
        for p in c:
            if p.endswith(".pyc"):
                continue
            if p.endswith(".py"):
                path = "{}/{}".format(a, p)
                # print(path)
                file_path_list.append(path)
            if p.endswith('.bin'):
                bin_path = "{}/{}".format(a, p)

    return file_path_list, bin_path

# 获取需要编译为so文件的py文件
py_to_c_files, bin_path = load_files()

setup(
    name = f'{APP_NAME}.app',
    version="1.0",
    ext_modules = cythonize(py_to_c_files),
    script_args=["build_ext",]
)

# 删除编译过程中生成的.c文件
del_c_files = [path.replace(".py", ".c") for path in py_to_c_files]
for c_p in del_c_files:
    if os.path.isfile(c_p):
        os.remove(c_p)


for tn in os.listdir("build"):
    if "temp" in tn:
        rmtree(f"build/{tn}")
    if "lib" in tn:
        # 重命名lib文件夹
        os.rename(f"build/{tn}", f"build/lib_{APP_NAME}")

# copy bsdf file to build folder
bin_file = bin_path.split('/')[-1]
copy(bin_path, f"build/lib_{APP_NAME}/src/dataset/{bin_file}")

# 复制build之后的工程到接口封装目录3d_reconstruction_api
folder_copy(f"build/lib_{APP_NAME}/{BUILD_DIR_NAME}", f"{GIT_UPLOAD_DIR}/{BUILD_DIR_NAME}")
folder_copy(f"{BUILD_DIR_NAME}/render/renderutils/{CPP_EXTENSION}", f"{GIT_UPLOAD_DIR}/{BUILD_DIR_NAME}/render/renderutils/{CPP_EXTENSION}")
copy(f"{BUILD_DIR_NAME}/render/tinycudann/{_C_SO}", f"{GIT_UPLOAD_DIR}/{BUILD_DIR_NAME}/render/tinycudann/{_C_SO}")

copy(PIP_REQUIREMENTS, f"{GIT_UPLOAD_DIR}/{PIP_REQUIREMENTS}")
# copy(DOCKER_FILE, f"{GIT_UPLOAD_DIR}/{DOCKER_FILE}")