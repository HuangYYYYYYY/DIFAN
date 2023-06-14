# DIFAN
Simple Lens Imaging ,Large-FOV
## DIFAN: Iterative Filter Adaptive Network for Simple Lens Imaging System

This repo contains training and evaluation code for the following paper:

> [**Iterative Filter Adaptive Network for Simple Lens Imaging System**]
>  
> *Optics Communication


<p align="left">
  <a href="https://junyonglee.me/projects/IFAN">
    <img width=85% src="./assets/DIFAN .png"/>
  </a><br>
</p>

## Getting Started
### Prerequisites

*Tested environment*

![Python](https://img.shields.io/badge/Python-3.9.5-green.svg?style=plastic)
![PyTorch](https://img.shields.io/badge/PyTorch-1.7.1%20&%201.8.0%20&%201.9.0%20&%201.10.2%20&%201.11.0-green.svg?style=plastic)
![CUDA](https://img.shields.io/badge/CUDA-11.1%20&%2011.3%20&%2011.7-green.svg?style=plastic)

#### 1. Environment setup
* Option 1. install from scratch
    ```bash
    $ git clone https://github.com/codeslake/IFAN.git
    $ cd DIFAN

    ## for CUDA11.7
    $ conda create -y --name DIFAN python=3.9 && conda activate DIFAN
    $ sh install_CUDA11.7.sh

    ## for CUDA11.1 or CUDA11.3
    $ conda create -y --name DIFAN python=3.8 && conda activate DIFAN
    # CUDA11.1
    $ sh install_CUDA11.1.sh

* Option 2. docker
    ```bash
    $ nvidia-docker run --privileged --gpus=all -it --name DIFAN --rm codeslake/IFAN /bin/hy
    $ git clone https://github.com/codeslake/DIFAN.git
    $ cd IFAN

    # for CUDA11.1
    $ coda activate IFAN_CUDA11.1

    # for CUDA11.3 (for amp)
    $ coda activate IFAN_CUDA11.3 
    
   # for CUDA11.7 
    $ coda activate IFAN_CUDA11.7
    ```

#### 2. Datasets
Download and unzip datasets under `[DATASET_ROOT]`:
* DPDD dataset: [Google Drive](https://drive.google.com/open?id=1Mq7WtYMo9mRsJ6I6ccXdY1JJQvwBuMuQ&authuser=codeslake%40gmail.com&usp=drive_fs) | [Dropbox](https://www.dropbox.com/s/w9urn5m4mzllrwu/DPDD.zip?dl=1)
* DPDD-SL dataset:

```
[DATASET_ROOT]
 ├── DPDD
 ├── DPDD-SL
```
> `[DATASET_ROOT]` can be modified with [`config.data_offset`](https://github.com/codeslake/IFAN/blob/main/configs/config.py#L48-L49) in `./configs/config.py`.

#### 3. Pre-trained models
Download and unzip pretrained weights ([Google Drive](https://drive.google.com/open?id=1MnoyTZgHgG7vwZYuloLeQgegwa6O9A7O&authuser=codeslake%40gmail.com&usp=drive_fs) | [Dropbox](https://www.dropbox.com/s/y2s8pmukgzkqpky/DIFAN_TEST.pytorch?dl=0) under `./ckpt/`:

```
.
├── ...
├── ./ckpt
│   ├── IFAN.pytorch
│   ├── ...
│   └── IFAN_dual.pytorch
└── ...
```

## Testing models

```shell
## Table 2 in the main paper
# Our final model used for comparison
 python run.py --mode IFAN --network IFAN --config config_IFAN --data DBCI --ckpt_abs_name ckpt/IFAN_03220.pytorch --data_offset ./DATASET_ROOT --output_offset ./output


  python run.py --mode IFAN --network IFAN --config config_IFAN --data DPDD --ckpt_abs_name ckpt/IFAN.pytorch --data_offset /data_offset --output_offset ./output

#

> Testing results will be saved in `[LOG_ROOT]/IFAN_CVPR2021/[mode]/result/quanti_quali/[mode]_[epoch]/[data]/`.

> `[LOG_ROOT]` can be modified with [`config.log_offset`](https://github.com/codeslake/IFAN/blob/main/configs/config.py#L65) in `./configs/config.py`.

#### Options
* `--data`: The name of a dataset to evaluate. `DPDD-SL`  `random`. Default: `DPDD-SL`
    * The folder structure can be modified in the function [`set_eval_path(..)`](https://github.com/codeslake/IFAN/blob/main/configs/config.py#L114-L139) in `./configs/config.py`.
    * `random` is for testing models with any images, which should be placed as `[DATASET_ROOT]/random/*.[jpg|png]`.

## Wiki
* [Logging](https://github.com/codeslake/IFAN/wiki/Log-Details)
* [Training and testing details](https://github.com/codeslake/IFAN/wiki/Training-&-Testing-Details).

## Contact
Open an issue for any inquiries.
You may also have contact with (e-mail:huang2020bit@163.com)


## Citation
If you find this code useful, please consider citing:

```
@InProceedings{DIFAN2023,
    author    = {HUANG},
    title     = {Iterative Filter Adaptive Network for Simple Lens Imaging System},
    booktitle = {Optics Communications},
    year      = {2023}
}
```

