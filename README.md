# GANs_MI2I — Virtual Fluorescence Staining with Pix2PixHD

[![CI](https://github.com/Tonks684/GANs_MI2I/actions/workflows/ci.yml/badge.svg)](https://github.com/Tonks684/GANs_MI2I/actions/workflows/ci.yml)

Pix2PixHD-based conditional GAN for translating phase-contrast microscopy images into virtual fluorescent stains (nuclei & cytoplasm channels), adapted from [Wang et al. 2018](https://arxiv.org/abs/1711.11585) for 16-bit single-channel TIFF microscopy data.

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/Tonks684/GANs_MI2I.git
cd GANs_MI2I

# Option A — conda (recommended for GPU)
conda env create -f 04_image_translation_phd.yml
conda activate 04_image_translation_phd

# Option B — pip
pip install -e ".[dev]"       # includes test/lint extras
pip install -e ".[wandb]"     # adds W&B support
```

### 2. Download data

```bash
python download_and_split_dataset.py --output_image_folder ./data --crop_size 512
# Creates ./data/{input,nuclei,cyto}/{train,val}/
```

### 3. Train

```bash
# Edit hyperparameters in config/train.yaml, then:
cd pix2pixHD
python train_dlmbl.py $(python ../command_gen.py)

# With W&B logging:
python train_dlmbl.py $(python ../command_gen.py) --use_wandb --wandb_project my-project
```

### 4. Inference

```bash
cd pix2pixHD
# Standard inference (all validation images):
python -c "
from options.test_options import TestOptions
from models.models import create_model
from data.data_loader_dlmbl import CreateDataLoader
from test_dlmbl import inference
opt = TestOptions().parse()
model = create_model(opt)
dataset = CreateDataLoader(opt, phase='val').load_data()
inference(dataset, opt, model)
"

# Variational sampling (first N images, multiple stochastic draws):
# add: --variational_inf_runs 10 --max_samples 20
```

### 5. Evaluate

```bash
python segmentation_scores.py  # see function gen_segmentation_scores() for API
```

### 6. View training curves

```bash
tensorboard --logdir checkpoints/dlmbl_vscyto/tensorboard
```

---

## Docker

```bash
# Build once
docker compose build

# Download data
docker compose run --rm download

# Train (requires NVIDIA Docker runtime)
docker compose run --rm train

# Run tests (CPU, no data needed)
docker compose run --rm test

# TensorBoard at http://localhost:6006
docker compose up tensorboard
```

Set `WANDB_API_KEY` in a `.env` file at the repo root to enable W&B inside Docker.

---

## Configuration

All training hyperparameters live in [config/train.yaml](config/train.yaml). Edit that file instead of modifying `command_gen.py`. Key options:

| Key | Default | Description |
|-----|---------|-------------|
| `name` | `dlmbl_vscyto` | Experiment name (checkpoint subdirectory) |
| `dataroot` | `../../data` | Path to dataset root |
| `target` | `nuclei` | Target channel (`nuclei` or `cyto`) |
| `batchSize` | `4` | Training batch size |
| `fp16` | `false` | Enable AMP (automatic mixed precision) |
| `seed` | `42` | RNG seed (enables cuDNN determinism) |
| `use_wandb` | _(unset)_ | Enable W&B logging |

---

## Training Speed Tips

- `--fp16` enables AMP via `torch.cuda.amp` (~1.5–2× faster on Ampere+ GPUs)
- `torch.compile` is applied automatically on PyTorch ≥ 2.0 (~10–30% throughput gain)
- `cudnn.benchmark=True` is set when no seed is fixed (finds fastest convolution algorithm)
- `zero_grad(set_to_none=True)` is used throughout to reduce memory overhead

---

## Project Structure

```
GANs_MI2I/
├── config/
│   └── train.yaml              # All hyperparameters — edit this
├── pix2pixHD/
│   ├── train_dlmbl.py          # Training loop (AMP, torch.compile, W&B, TensorBoard)
│   ├── test_dlmbl.py           # Inference + variational sampling
│   ├── encode_features.py      # Feature encoding + KMeans clustering
│   ├── models/                 # Generator, discriminator, losses
│   ├── data/                   # 16-bit TIFF paired dataset loader
│   ├── options/                # CLI argument definitions
│   └── util/                   # Visualisation utilities
├── tests/
│   ├── test_segmentation_scores.py  # IoU, F1, PSNR unit tests
│   ├── test_data_loader.py          # Dataset smoke tests
│   └── test_model_forward.py        # Generator/discriminator shape tests
├── download_and_split_dataset.py
├── segmentation_scores.py
├── command_gen.py              # Reads config/train.yaml → CLI flags
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## Introduction to Generative Modelling
In this part of the exercise, we will tackle the same supervised image-to-image translation task but use an alternative approach. Here we will explore a generative modelling approach, specifically a conditional Generative Adversarial Network (cGAN). <br>

The previous regression-based method learns a deterministic mapping from phase contrast to fluorescence. This results in a single virtual staining prediction to the image translation task which often leads to blurry results. Virtual staining is an ill-posed problem; given the phase contrast image, with inherent noise and lack of contrast between the background and the structure of interest, it can be very challenging to virtually stain from the phase contrast image alone. In fact, there is a distribution of possible virtual staining solutions that could come from the phase contrast.

cGANs learn to map from the phase contrast domain to a distirbution of virtual staining solutions. This distribution can then be sampled from to produce virtual staining predictions that are no longer a compromise between possible solutions which can lead to improved sharpness and realism in the generated images. Despite these improvements, cGANs can be prown to 'hallucinations' in which the network instead of making a compromise when it does not know something (such as a fine-grain detail of the nuclei shape) it makes something up that looks very sharp and realistic. These hallucinations can appear very plausible, but in many cases to predict such details from the phase contrast is extremely challenging. This is why determining reliable evaluation criteria for the task at hand is very important when dealing with cGANs .<br>
<br>
<br>
![Overview of cGAN](https://github.com/Tonks684/dlmbl_material/blob/main/imgs/GAN.jpg?raw=true)
<br>
<br>

At a high-level a cGAN has two networks; a generator and a discriminator. The generator is a fully convolutional network that takes the source image as input and outputs the target image. The discriminator is also a fully convolutional network that takes as input the source image concatentated with a real or fake image and outputs the probabilities of whether the real fluorescence image is real or whether the fake virtual stain image is fake as shown in the figure above.<br>

The generator is trained to fool the discriminator into predicting a high probability that its generated outputs are real, and the discriminator is trained to distinguish between real and fake images. Both networks are trained using an adversarial loss in a min-max game, where the generator tries to minimize the probability of the discriminator correctly classifying its outputs as fake, and the discriminator tries to maximize this probability. It is typically trained until the discriminator can no longer determine whether or not the generated images are real or fake better than a random guess (p(0.5)).<br>

We will be exploring [Pix2PixHD GAN](https://arxiv.org/abs/1711.11585) architecture, a high-resolution extension of a traditional cGAN adapted for our recent [virtual staining works](https://ieeexplore.ieee.org/abstract/document/10230501?casa_token=NEyrUDqvFfIAAAAA:tklGisf9BEKWVjoZ6pgryKvLbF6JyurOu5Jrgoia1QQLpAMdCSlP9gMa02f3w37PvVjdiWCvFhA). Pix2PixHD GAN improves upon the traditional cGAN by using a coarse-to-fine generator, a multi-scale discrimator and additional loss terms. The "coarse-to-fine" generator is composed of two sub-networks, both ResNet architectures that operate at different scales. As shown below the first sub-network (G1) generates a low-resolution image, which is then upsampled and concatenated with the source image to produce a higher resolution image. The multi-scale discriminator is composed of 3 networks that operate at different scales, each network is trained to distinguish between real and fake images at that scale using the same convolution kernel size. This leads to the convolution having a much wider field of view when the inputs are downsampled. The generator is trained to fool the discriminator at each scale. 
<br>
<br>
![Pix2PixGAN ](https://github.com/Tonks684/dlmbl_material/blob/main/imgs/Pix2pixHD_1.jpg?raw=true)
<br>
<br>
The additional loss terms include a feature matching loss (as shown below), which encourages the generator to produce images that are perceptually similar to the real images at each scale. As shown below for each of the 3 discriminators, the network takes seperaetly both phase concatenated with virtual stain and phase concatenated with fluorescence stain as input and as they pass through the network the feature maps obtained for each ith layer are extracted. We then minimize the loss which is the mean L1 distance between the feature maps obtained across each of the 3 discriminators and each ith layer. <br>
![Feature Matching Loss Pix2PixHD GAN](https://github.com/Tonks684/dlmbl_material/blob/main/imgs/Pix2pixHD_2.jpg?raw=true)

All of the discriminator and generator loss terms are weighted the same.
