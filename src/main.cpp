#include <vector>
#include <string>
#include <chrono>
#include <iostream>
#include "inference.h"

std::vector<std::string> dataset()
{
 	std::vector<std::string> samples;
    samples.push_back("../pic/dog.jpg");
    samples.push_back("../pic/shark.jpg");
  	samples.push_back("../pic/turtle.jpg");
  	samples.push_back("../pic/dog-1.jpg"); 
  	return samples;
}

int main()
{   // start time measurement
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	Inference predictor;
  	std::vector<std::string> samples = dataset();
  	int sample_count = samples.size();
  	int num_workers;
  	std::cout << "Input the num of workers: ";
  	std::cin >> num_workers;
  	predictor.predict(std::move(samples), num_workers);
  	predictor.display(sample_count);
  	// stop time measurement and print execution time
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
  	std::cout << "Inference took " << duration <<" milliseconds" << std::endl;
  	return 0;
}