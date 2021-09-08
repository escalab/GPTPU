# GPTPU: General-Purpose Computing on (Edge) Tensor Processing Units

Welcome to the repository of ESCAL @ UCR's GPTPU project! We aim at demonstrating the power of matrix processing units (MXUs) that are now ubiquitous in all types of computing platforms. This project chooses Google's Edge TPU -- a "relatively" open archtecture that allows everyone to purchase and integrate into their systems. In our preliminary results, we achieve 2.46x speedup over one single high-end CPU core. You may reference our arXiv paper 
https://arxiv.org/pdf/2107.05473.pdf or the paper coming up in SC21 for more information.

[![DOI](https://zenodo.org/badge/392392831.svg)](https://zenodo.org/badge/latestdoi/392392831)


# Hardware installation
You will need an M.2 version of the edge TPU (recommeded)
https://coral.ai/docs/m2/get-started/
or a USB edge TPU accelerator to installed in your system. 

Once you have the Edge TPUs, please follow Google's document to install their drivers and toolchains before installing our GPTPU framework.
https://coral.ai/docs/m2/get-started/#2-install-the-pcie-driver-and-edge-tpu-runtime

You may also reference Section 3.1 of our arXiv paper to build a multi-Edge-TPU machine (a lot cheaper) or purchase ASUS's 8x Edge TPU PCIe card https://iot.asus.com/products/AI-accelerator/AI-Accelerator-PCIe-Card/

# Install GPTPU library (Our contribution)
## Compile all benchmarks
```
$ make 
```
## Run all benchmarks
```
$ make run
// each benchmark shows its RMSE and error rate as mentioned in paper. Some may involve experimental features.
```
## gptpu library is pre-compiled as libgptpu.so and linked by Makefile. 

## Compile the gptpu library
```
// rune the Makefile_gptpu, while it requires sudo permission
// sc21 is simply an demo account without sudo permission
``` 
 ### Prerequisites
 tensorflow 1.13.1 // Python-based model creation creates the template for the first time if not exist
 bazel 2.0.0
 cnpy ([https://github.com/rogersce/cnpy](https://github.com/rogersce/cnpy))
 cmake
 python3
 numpy
 apex driver
 gesket driver
 cblas (for comparison only)
 ```
 $ sudo apt-get install libopenblas-dev
 ```
 ## Set PATH
 ```
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
```
 
## Note about GEMM
GEMM is fundamental and it's our very first benchmark. It also includes exact mode as experimental feature. Our exact mode is still in progress while it adopts blocking algorithm in a block size of 256 to avoid uint8_t overflow. In this demo, we show the floating point approximation result with small RMSE and error rate as mentioned in paper.

## Multi-tpu scheme
GPTPU library allows enabling multiple TPUs for parallel computing. This following device initialization API
```
open_devices(int opening_order, int wanted_dev_cnt)
```
has two arguments ***opening_order*** and ***wanted_dev_cnt***.

1. opening_order:  0: open device(s) sequentially starting from first device (index 0). 1: open device(s) sequentially starting from a random number device. (You can extend this argument with more opening policies)
2. wanted_dev_cnt: number of devices you want to open. (constrained by maximum number of devices available)

## openctpu usage
Please refer to the example source code : **./src/openctpu.cc** in detail.
