# CPPND: ResNet based Image classification using Pytorch C++

The project provides the model which is trained on ImageNet and can be used to classify the images. The project uses multi-thread approach to calculate the model predictions.  

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

## Basic Instructions

1. Clone this repo.
2. `cd src` and `python model.py`
3. Make a build directory in the top level directory: `mkdir build && cd build`
4. Compile: `cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch .. && make`
5. Run it: `./Pytorch-CNN-classifier`.
6. Modify the `pic/` folder to add custom images.

## Model Output
```
Input the num of workers: 2
Spawning workers...
Collecting results...
from worker : 140511188563712 : Top-1 Prediction with prob  71.2% of    ../pic/turtle.jpg: Label: box turtle, box tortoise
from worker : 140511196956416 : Top-1 Prediction with prob  97.1% of       ../pic/dog.jpg: Label: beagle
from worker : 140511188563712 : Top-1 Prediction with prob  23.9% of     ../pic/dog-1.jpg: Label: Cardigan, Cardigan Welsh corgi
from worker : 140511196956416 : Top-1 Prediction with prob  67.7% of     ../pic/shark.jpg: Label: tiger shark, Galeocerdo cuvieri
Inference took 2214 milliseconds
```

## Code Structure

* `main.cpp` : contains the main code to create and call the class `Inference`.
* `Inference` : This class encapsulates the Torch Script module of ResNet along with other necessary fucntions like `predict` and `display` in `inference.cpp`
  *  `predict` : splits the dataset and spawns multiple threads with each batch.
  *  `display` : collects the model output from the threads and prints the Top-K predictions along with their probability.
* `model.py` : to generate torch script file.

## Rubric Info

* Loops, Functions, I/O :
  * C++ functions and control structures. : demonstarted in many places. for example `inference.cpp 104:114`   
  * The project reads data from a file and process the data, or the program writes data to a file. : `inference.cpp 104:114` 
  * The project accepts user input and processes the input. : `main.cpp:25` to get the number of threads.
* Object Oriented Programming:
  * Classes encapsulate behavior, Templates, Class constructors utilize member initialization lists, Classes encapsulate behavior : `inference.h`
* Memory Management:
  * The project makes use of references in function declarations. : `inference.cpp:90`
  * The project follows the Rule of 5. : `inference.cpp 27:88`
  * The project uses move semantics to move data, instead of copying it, where possible. `inference.h 144`
  * The project uses smart pointers instead of raw pointers.: `inference.h 62`
* Concurrency:
  * The project uses multithreading.: `inference.cpp:159`
  * A mutex or lock is used in the project. : `inference.cpp: 128:130`
  * A condition variable is used in the project. : `inference.cpp: 10:19`
