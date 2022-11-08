# Joint Privacy Enhancement and Quantization in Federated Learning
<!--Created by Natalie Lang and Nir Shlezinger from Ben-Gurion University.-->

![image](https://user-images.githubusercontent.com/55830582/200581053-3eb814d9-2342-4ae7-89b7-b9647c886e15.png)

## Introduction
In this work we propose a method for joint privacy enhancement and quantization (JoPEQ), unifying lossy compression and privacy enhancement for federated learning. This repository contains a basic PyTorch implementation of JoPEQ. Please refer to our [paper](https://arxiv.org/abs/2208.10888) for more details.

## Usage
This code has been tested on Python 3.7.3, PyTorch 1.8.0 and CUDA 11.1.

### Prerequisite
1. PyTorch=1.8.0: https://pytorch.org
2. scipy
3. tqdm
4. matplotlib
5. torchinfo
6. TensorboardX: https://github.com/lanpa/tensorboardX

### Training
```
python main.py --exp_name=jopeq --quntization --lattice_dim 2 --R 1 --privacy --privacy_noise jopeq_vector --epsilon 4 --sigma_squared 0.2 --nu 4
```

### Testing
```
python main.py --exp_name=jopeq --eval 
```
