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
    bool valid_num_workers;
    do{
	std::cout << "Input the num of workers: ";
	std::cin >> num_workers;
	if(num_workers > 0){
		valid_num_workers = sample_count % num_workers == 0;
		std::cout<<"Cannot do equal data split among the workers. please give a number such that the total samples count "<< sample_count
		       	<<" is divisible and less than total sample count " << sample_count <<std::endl;
	}
	else
	{
		std::cout<<"Please give a non-zero positive number such that the total samples count "<< sample_count
		       	<<" is divisible and less than total sample count " << sample_count <<std::endl;
		valid_num_workers = false;
	}
    }
    while(!valid_num_workers);
    predictor.predict(std::move(samples), num_workers);
    predictor.display(sample_count);
    // stop time measurement and print execution time
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
    std::cout << "Inference took " << duration <<" milliseconds" << std::endl;
    return 0;
}
