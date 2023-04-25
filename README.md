#### 基于神经渲染的自动化三维重建项目

##### 1.下载代码
  ```shell
  git clone http://git.moviebook.cn/algorithm/reconstruction3d/3d_reconstruction.git --recursive
  ```
##### 2.配置环境
  ```shell
  cd 3d_reconstruction
  git submodule add http://git.moviebook.cn/algorithm/Reconstruction3D/ceres-solver.git src/submodule/ceres-solver  ## 用于搭建项目运行环境,不搭建环境时无需下载
  git submodule add http://git.moviebook.cn/algorithm/Reconstruction3D/colmap.git src/submodule/colmap              ## 用于搭建项目运行环境,不搭建环境时无需下载
  conda create -n 3d_modeling python=3.8
  conda activate 3d_modeling
  sh env_config.sh
  ```

##### 3.运行测试代码进行三维建模
  ```shell
  python exp_test.py --data_name shoes_test
  ```

##### 4.安装Flask库
  ```shell
  pip install Flask
  pip install requests
  ```

##### 5.启动Flask服务
  ```shell
  python api_test_v0.1.py
  ```

##### 6.运行build编译(API封装时使用)
  ```shell
  python setup.py
  ```
  运行完成后会自动生成API工程目录(3d_reconstruction_api)，用于后续的部署。