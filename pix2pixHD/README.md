<br><br>

# Evaluation of virtual staining for high-throughput screening

<img src='imgs/Fig1.png' align="center" width=480>


### [Paper](https://ieeexplore.ieee.org/abstract/document/10230501?casa_token=NEyrUDqvFfIAAAAA:tklGisf9BEKWVjoZ6pgryKvLbF6JyurOu5Jrgoia1QQLpAMdCSlP9gMa02f3w37PvVjdiWCvFhA) <br>
Pytorch implementation of adapted pix2pixHD method for high-resolution (e.g. 1080x1080) virtual staining via image-to-image translation.
## Prerequisites
- Linux or macOS
- Python 3
- NVIDIA GPU (11G memory or larger) + CUDA cuDNN

## Getting Started
To create a conda environment from the `pix2pixHDCUDA11_environment.yml` file, follow these instructions:
1. Open a terminal or command prompt.
2. Navigate to the directory where the `pix2pixHDCUDA11_environment.yml` file is located. 
3. Run the following command to create the conda environment:
  ```bash
  conda env create -f pix2pixHDCUDA11_environment.yml
  ```
  This command will read the `pix2pixHDCUDA11_environment.yml` file and create a new conda environment with the specified dependencies.
4. Wait for the environment creation process to complete. Conda will download and install all the necessary packages.
5. Once the environment is created, activate it by running the following command:
  ```bash
  conda activate <environment_name>
  ```
  Replace `<environment_name>` with the name you want to give to the environment.
6. You can now use the conda environment with all the installed dependencies for your project.
Remember to replace `<environment_name>` with a suitable name for your environment. You can choose any name you like.
7. Clone this repo:
```bash
git clone git@github.com:Tonks684/GANs_MI2I.git
cd pix2pixHD
```

### Training
```bash
python ../pix2pixHD/train_dlmbl.py --dataroot ../ --data_type 16 --batchSize 4 --checkpoints_dir ../../results/dlmbl_vscyto --label_nc 0 --name dlmbl_vscyto --no_instance  --resize_or_crop none --input_nc 1 --output_nc 1 --seed 42 --no_vgg_loss  --nThreads 1 --loadSize 256 --ndf 32 --norm instance --use_dropout  --fp16 --gpu_ids 1
```
- To view training results, please launch `tensorboard --logdir opt.checpoints_dir`

### Multi-GPU training
```bash
python ../pix2pixHD/train_dlmbl.py --dataroot ../ --data_type 16 --batchSize 4 --checkpoints_dir ../../results/dlmbl_vscyto --label_nc 0 --name dlmbl_vscyto --no_instance  --resize_or_crop none --input_nc 1 --output_nc 1 --seed 42 --no_vgg_loss  --nThreads 1 --loadSize 256 --ndf 32 --norm instance --use_dropout  --fp16 --gpu_ids 1,2,3
```
### Training with Automatic Mixed Precision (AMP) for faster speed
- To train with mixed precision support, please first install apex from: https://github.com/NVIDIA/apex
- You can then train the model by adding `--fp16`. For example,
```bash
python ../pix2pixHD/train_dlmbl.py --dataroot ../ --data_type 16 --batchSize 4 --checkpoints_dir ../../results/dlmbl_vscyto --fp16
```
### Testing
```bash

python test.py --results_dir ../results/ --dataroot ../ --data_type 16 --batchSize 1 --checkpoints_dir ../../results/dlmbl_vscyto
```
The test results will be saved to a html file here: `./results/


## Acknowledgments
This code borrows heavily from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [pix2pixHD] (https://github.com/NVIDIA/pix2pixHD)
