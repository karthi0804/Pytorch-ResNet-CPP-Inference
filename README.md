# CPPND: Capstone Hello World Repo

The Capstone Project gives you a chance to integrate what you've learned throughout this program. This project will become an important part of your portfolio to share with current and future colleagues and employers.

In this project, you can build your own C++ application starting with this repo, following the principles you have learned throughout this Nanodegree Program. This project will demonstrate that you can independently create applications using a wide range of C++ features.

## Dependencies for Running Locally
* cmake >= 3.7
  * [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 
  * make is installed by default on most Linux distros
* gcc/g++ >= 5.4
  * gcc / g++ is installed by default on most Linux distros
* OpenCV >= 3.0
  * [click here for installation instructions](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html)
* LibTorch 
  * wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
  * unzip libtorch-shared-with-deps-latest.zip
* Python Torch >=1.9
  * pip install torch torchvision

## Basic Build Instructions

1. Clone this repo.
2. `cd src` and `python model.py`
3. Make a build directory in the top level directory: `mkdir build && cd build`
4. Compile: `cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch .. && make`
5. Run it: `./Pytorch-CNN-classifier`.