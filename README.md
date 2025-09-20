<div align="center">
<h1> LENS-Net: Low-Energy Spiking Neural Network for Remote Sensing Saliency </h1>
</div>

## 🎈 News

[2025.9.15] Training and testing code released.



## ⭐ Abstract

Salient object detection in optical remote sensing images (ORSI-SOD) is vital for applications such as urban planning and disaster monitoring. Yet, existing deep models remain energy-intensive and unsuitable for edge deployment. 
We tackle this challenge by proposing  a Low-Energy Spiking Network Neural for Remote Sensing Saliency(LENS-Net), the first fully spiking neural network for ORSI-SOD. LENS-Net employs a Spike-driven Transformer v3 encoder to extract multi-scale features with low energy cost, and a Spike Multi-scale Attention Decoder that fuses contextual cues via spike-driven channel attention and up-convolution, ensuring effective saliency representation under sparse computation. 
Moreover, a {sigmoid-based soft surrogate gradient} replaces hard truncation, stabilizing training and enhancing boundary recognition in complex scenes. 
Across three benchmark datasets (ORSSD, EORSSD, and ORI-4199), LENS-Net demonstrates outstanding performance while maintaining high energy efficiency, outperforming all lightweight ANN counterparts. For instance, on the ORSSD dataset under a timestep T=4 during inference, it achieves a $S_{\alpha}$ of 92.79\%, an $MAE$ of 0.0109, and consumes only 11.48 mJ of energy. These results establish an efficient, low-energy solution for practical ORSI-SOD deployment.
The source code is available at https://github.com/7dra/LENS-Net.

## 🚀 Introduction

<div align="center">
    <img width="800" alt="image" src="LENS-Net.png?raw=true">
</div>

The challenges: .

## 📻 Overview

<div align="center">
<img width="800" alt="image" src="figures/network.png?raw=true">
</div>

Illustration of the overall architecture.


## 📆 TODO

- [x] Release code

## 🎮 Getting Started

### 1. Install Environment

```
conda create -n Net python=3.8
conda activate Net
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs PyWavelets
```

### 2. Prepare Datasets

- Download datasets: ISIC2018 from this [link](https://challenge.isic-archive.com/data/#2018).


- Folder organization: put datasets into ./data/datasets folder.

### 3. Train the Net

```
python train.py
```

### 3. Test the Net

```
python inference.py
python metric.py
```

### 3. Code example

```
python Test/test_example.py
```

## 🖼️ Visualization

<div align="center">
<img width="800" alt="image" src="figures/com_pic.png?raw=true">
</div>

<div align="center">
We compare our method against 13 state-of-the-art methods. The red box indicates the area of incorrect predictions.
</div>

## ✨ Quantitative comparison

<div align="center">
<img width="800" alt="image" src="figures/com_tab.png?raw=true">
</div>

<div align="center">
Performance comparison with ten SOTA methods on 5 datasets.
</div>

## 🖼️ Visualization of Ablation Results

<div align="center">
<img width="800" alt="image" src="figures/aba.png?raw=true">
</div>



## 🖼️ Convergence Analysis

<div align="center">
<img width="800" alt="image" src="figures/curve.png?raw=true">
</div>




## 🎫 License

The content of this project itself is licensed under Your License url.
