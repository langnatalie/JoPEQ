# JoPEQ: Joint Privacy Enhancement and Quantization in Federated Learning
<!--Created by Natalie Lang and Nir Shlezinger from Ben-Gurion University.-->

![image](https://user-images.githubusercontent.com/55830582/151803310-6bf83637-f606-4e28-a8ab-802985a8c879.png)

## Introduction
In this work we propose a method for joint privacy enhancement and quantization (JoPEQ), unifying lossy compression and privacy enhancement for FL. This repository contains a basic PyTorch implementation of JoPEQ. Please refer to our [paper](https://drive.google.com/file/d/1AhirvDbA-43B0XKVa9RTpzVbhsxO5anW/view?usp=sharing) for more details.

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
python main.py --exp_name=jopeq --quntization --R 1 --privacy --epsilon 4 --privacy_noise PPN
```

### Testing
```
python main.py --exp_name=jopeq --eval 
```
