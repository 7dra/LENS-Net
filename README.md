<div align="center">
<h1> LENS-Net: Low-Energy Spiking Neural Network for Remote Sensing Saliency </h1>
</div>

## 🎈 News

[2025.9.15] Training and testing code released.



## ⭐ Abstract

Salient Object Detection in Optical Remote Sensing Images (ORSI-SOD) is vital for applications such as urban planning and disaster monitoring. Yet, existing deep models remain energy-intensive and unsuitable for edge deployment. 
We tackle this challenge by proposing  a Low-Energy Spiking Network Neural for Remote Sensing Saliency(LENS-Net), the first fully spiking neural network for ORSI-SOD. LENS-Net employs a Spike-driven Transformer v3 encoder to extract multi-scale features with low energy cost, and a Spike Multi-scale Attention Decoder that fuses contextual cues via spike-driven channel attention and up-convolution, ensuring effective saliency representation under sparse computation. 
Moreover, a {sigmoid-based soft surrogate gradient} replaces hard truncation, stabilizing training and enhancing boundary recognition in complex scenes. 
Across three benchmark datasets (ORSSD, EORSSD, and ORI-4199), LENS-Net demonstrates outstanding performance while maintaining high energy efficiency, outperforming all lightweight ANN counterparts. For instance, on the ORSSD dataset under a timestep T=4 during inference, it achieves a $S_{\alpha}$ of 92.79\%, an $MAE$ of 0.0109, and consumes only 11.48 mJ of energy. These results establish an efficient, low-energy solution for practical ORSI-SOD deployment.
The source code is available at https://github.com/7dra/LENS-Net.

## 🚀 Contribution

- **First SNN-based ORSI-SOD Method**  
  We present the first SNN-based method for ORSI-SOD, filling an important gap and establishing a strong baseline for future neuromorphic research in remote sensing.
- **Novel Multi-scale Spiking Decoder (SpikeMAD)**  
  We design a novel multi-scale spiking decoder termed as SpikeMAD, which incorporates membrane potential residual connections and a multi-scale fusion strategy to achieve efficient and effective feature integration in remote sensing imagery.
- **Soft-Clip Spike Firing Approximation**  
  We propose a soft-clip spike firing approximation backpropagation function that ensures smooth gradient transitions, eliminates discontinuities from hard truncation, and enhances boundary recognition in salient object detection.
- **Extensive Experimental Validation**  
  Extensive experiments on three benchmark datasets demonstrate that LENS-Net achieves superior accuracy compared to all lightweight ANN counterparts, while consuming significantly less energy (e.g., only 11.61 mJ on ORSSD with T=4).

<div align="center">
    <img width="800" alt="image" src="figures/LENS-Net.png?raw=true">
</div>
<div align="center">
    Illustration of our LENS-Net
</div>


## 📆 TODO

- [x] Release code

## 🎮 Getting Started

### 1. Install Environment

```
conda create -n LENS_Net python=3.8
conda activate LENS_Net
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 -c pytorch
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


## 🖼️ Comparative experiment of LENS-Net on three datasets.

<div align="center">
<img width="700" alt="image" src="figures/compriment_experiment.png?raw=true">
</div>
<div align="center">
<img width="300" alt="image" src="figures/compriment_experiment2.png?raw=true">
</div>
<div align="center">
We compare our method against 6 state-of-the-art methods.
</div>

## 🖼️ Ablation experiment of LENS-Net on ORSSD dataset.

<div align="center">
<img width="300" alt="image" src="figures/ABLATION.png?raw=true">
</div>

## ✨ Visualization of saliency prediction maps by different methods on the EORSSD dataset 

<div align="center">
<img width="700" alt="image" src="figures/comparative.png?raw=true">
</div>

<div align="center">
Performance comparison with ten SOTA methods on 5 datasets.
</div>

## ✨ Visualization ablation studies on the ORSSD dataset

<div align="center">
<img width="300" alt="image" src="figures/ablatio_study.png?raw=true">
</div>

##  💡 Layer-wise average spiking firing rates of LENS-Net on the EORSSD test dataset(600 images)

<div align="center">
<img width="400" alt="image" src="figures/fire_rate.png?raw=true">
</div>

## 🎫 License

The content of this project itself is licensed under Your License url.
