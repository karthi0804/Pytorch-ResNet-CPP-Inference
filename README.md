# ResNet based Image classification using Pytorch C++

The project provides the model which is trained on ImageNet and can be used to classify the images. The project uses multi-thread approach to calculate the model predictions.  

## Run Using Docker (tested with Docker 20.10.7)

 * Clone this repo.
 * Pull the docker image environment using `docker pull karthi0804/pytorch-resnet-cpp:deploy` - it takes around 10 mins! 
 * Run the docker contianer using `docker run --rm -it -v /absolute/path/to/repo:/classifier karthi0804/pytorch-resnet-cpp:deploy`
 * If the docker container is up successfully, you fill find the git code under the directory `classifier` inside the container. Please check by using `root@XXXXXXXXXXXXXX:/# ls`
 * Go to project root using `root@XXXXXXXXXXXXXX:/# cd /classifier` inside the docker container.
 * Make a build directory in the top level directory: `mkdir build && cd build` inside the docker container.
 * Compile: `cmake -DCMAKE_PREFIX_PATH=/libtorch .. && make` inside the docker container.
 * Run it: `./Pytorch-CNN-classifier` inside the docker container.
 * Modify the `pic/` folder to add custom images.
 
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
