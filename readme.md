# Fourier Convolution Block with global receptive field for MRI reconstruction
PyTorch implementation.

### Abstract
Reconstructing images from under-sampled Magnetic Resonance Imaging (MRI) signals significantly reduces scan time and improves clinical practice. However, Convolutional Neural Network (CNN)-based methods, while demonstrating great performance in MRI reconstruction, may face limitations due to their restricted receptive field (RF), hindering the capture of global features. This is particularly crucial for reconstruction, as aliasing artifacts are distributed globally. Recent advancements in Vision Transformers have further emphasized the significance of a large RF. In this study, we proposed a novel global Fourier Convolution Block (FCB) with whole image RF and low computational complexity by transforming the regular spatial domain convolutions into frequency domain. Visualizations of the effective RF and trained kernels demonstrated that FCB improves the RF of reconstruction models in practice. The proposed FCB was evaluated on four popular CNN architectures using brain and knee MRI datasets. Models with FCB achieved superior PSNR and SSIM than baseline models and exhibited more details and texture recovery.

### Architecture


### Environment
This project has the following dependencies:

- PyTorch (version 1.12.0 or later)
- NumPy (version 1.24.4 or later)
- Scikit-Image (version 0.21.0 or later)

### Datasets
The dataset used in this project is a subset of [FastMRI](https://github.com/facebookresearch/fastMRI). 
Pre-processing can be done by running the following commands. 

```bash
python preprocess/gen_slices.py
python preprocess/gen_smap.py
```
- Please ensure to modify the paths for loading and saving data within the two scripts.
- Please ensure to update the data paths in the `utils.py` file to match yours.

### Training
```bash
python train.py -c ./config/UNet_ccs_8_brain.yaml
python train.py -c ./config/FUNet_ccs_8_brain.yaml
```
The `repara: True` setting in the configuration file indicates that the proposed re-parameterization method is being used. You should specify a baseline model for re-parameterization by setting the `repara_path` in the configuration file. Alternatively, you can set the `LR_repara` and `weight_decay_repara` parameters.

## Acknowledgements

This repository was built on the following resources:

- Data and Pre-processing: [FastMRI](https://github.com/facebookresearch/fastMRI)
- Baseline Models:
  - [MoDL_PyTorch](https://github.com/bo-10000/MoDL_PyTorch)
  - [E2EVar](https://github.com/facebookresearch/fastMRI)
  - [VS-Net](https://github.com/j-duan/VS-Net)

## Citing Our Work
If you find this code useful in your research, we kindly ask you to cite our work. Here is the citation information:

```bibtex
@article{SUN2025103349,
title = {Fourier Convolution Block with global receptive field for MRI reconstruction},
journal = {Medical Image Analysis},
volume = {99},
pages = {103349},
year = {2025},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2024.103349},
url = {https://www.sciencedirect.com/science/article/pii/S1361841524002743},
author = {Haozhong Sun and Yuze Li and Zhongsen Li and Runyu Yang and Ziming Xu and Jiaqi Dou and Haikun Qi and Huijun Chen},
}
