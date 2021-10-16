#include <iostream>
#include <chrono>
#include <algorithm>
#include "inference.h"


template <typename T>
void MessageQueue<T>::send(T &&msg)
{
  std::lock_guard<std::mutex> lck(_mutex);
  _queue.emplace_back(msg);
  _condition.notify_one();
}

template <typename T>
T MessageQueue<T>::receive()
{
  std::unique_lock<std::mutex> lck(_mutex);
  _condition.wait(lck, [this]{return !_queue.empty();});
  T msg = std::move(_queue.front());
  _queue.pop_front();
  return msg;
}


// Constructor
Inference::Inference() : _num_samples(0)
{
  	_module = std::make_shared<torch::jit::script::Module>(torch::jit::load("../resnet18.pt"));
    LoadImageNetLabel("../label.txt"); 
}

// Destructor
Inference::~Inference()
{	
    // set up thread barrier before this object is destroyed
    std::for_each(_threads.begin(), _threads.end(), [](std::thread &t) {t.join();});
}

// Copy Constructor
Inference::Inference(const Inference & source)
{
    _labels = source._labels;
  	_module = source._module;
 	_num_samples = 0;
 	_threads.clear();
}

// Copy Assign Constructor
Inference & Inference::operator=(const Inference & source)
{
  	if (this == &source){
      return *this;
    }
    _labels = source._labels;
  	_module = source._module;
 	_num_samples = 0;
 	_threads.clear();
  	return *this;
}

// Move Constructor
Inference::Inference(Inference && source)
{
    _labels = source._labels;
  	_module = std::move(source._module);
 	_num_samples = 0;
 	_threads.clear();
  	source._module = nullptr;
 	source._num_samples = 0;
 	source._threads.clear();
}

// Move Assign Constructor
Inference & Inference::operator=(Inference && source)
{
  	if (this == &source){
      return *this;
    }
    _labels = source._labels;
  	_module = std::move(source._module);
 	_num_samples = 0;
 	_threads.clear();
  	source._module = nullptr;
 	source._num_samples = 0;
 	source._threads.clear();
  	return *this;
}

void Inference::LoadImage(std::string file_name, cv::Mat &image) {
  image = cv::imread(file_name);  // CV_8UC3
  if (image.empty() || !image.data) {
    throw std::runtime_error("Can't load the image, please check your path!");
  }
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  // scale image to fit
  cv::Size scale(kIMAGE_SIZE, kIMAGE_SIZE);
  cv::resize(image, image, scale);
  // convert [unsigned int] to [float]
  image.convertTo(image, CV_32FC3, 1.0f / 255.0f);
}


void Inference::LoadImageNetLabel(std::string file_name) {
 	 std::ifstream ifs(file_name);  
    if (!ifs)
    {
      throw std::runtime_error("Please check your label file path!");
    }
  	std::string line;
  	while (std::getline(ifs, line)) {
  	  _labels.push_back(line);
  	}
}


void Inference::worker(std::vector<std::string> file_names)
{
  for (auto & file_name : file_names)
  {	
  	cv::Mat image;
  	LoadImage(file_name, image);
    auto input_tensor = torch::from_blob(image.data, {1, kIMAGE_SIZE, kIMAGE_SIZE, kCHANNELS});
    input_tensor = input_tensor.permute({0, 3, 1, 2});
    input_tensor[0][0] = input_tensor[0][0].sub_(0.485).div_(0.229);
    input_tensor[0][1] = input_tensor[0][1].sub_(0.456).div_(0.224);
    input_tensor[0][2] = input_tensor[0][2].sub_(0.406).div_(0.225);
    std::unique_lock<std::mutex> lck(_mutex);    
    torch::Tensor out_tensor = _module->forward({input_tensor}).toTensor();
    lck.unlock();
    auto results = out_tensor.sort(-1, true); 
    auto softmaxs = std::get<0>(results)[0].softmax(0);
    auto indexs = std::get<1>(results)[0];
  	std::vector<prediction_t> top_outputs;  	
  	for (int i = 0; i < kTOP_K; ++i) 
    {
      prediction_t output;
      output.idx = indexs[i].item<int>();
      output.prob = softmaxs[i].item<float>();
      output.file_name = file_name;
      output.thread_id = std::this_thread::get_id();
      top_outputs.emplace_back(output);
    }  	
  	_predictions.send(std::move(top_outputs));
  }
}

void Inference::predict(std::vector<std::string> && file_names, int num_workers)
{	
  	std::cout << "Spawning workers..." << std::endl;
  	int num_sample_per_worker =  file_names.size()/num_workers;
  	for (int worker_i = 0; worker_i < num_workers; worker_i++)
    {
      std::vector<std::string> samples_per_worker;
      for (int sample_i =0; sample_i<num_sample_per_worker; sample_i++)
      {
        samples_per_worker.push_back(file_names[(worker_i*num_sample_per_worker)+sample_i]);
      }
      _threads.emplace_back(std::thread(&Inference::worker, this, samples_per_worker));
    }
}
  

void Inference::display(int total_num_samples)
{
    std::cout << "Collecting results..." << std::endl;
  	while(_num_samples<total_num_samples)
    {
  	std::vector<prediction_t> top_outputs = _predictions.receive();
    for (int i = 0; i < kTOP_K; ++i) 
    {
      std::cout << "from worker : " << top_outputs[i].thread_id << " : Top-" << i+1  
        <<" Prediction with prob "<< std::setw(5) << std::setfill(' ')  << std::setprecision(3) << top_outputs[i].prob * 100.0 
        <<"% of "<< std::setw(20) << std::setfill(' ')  << top_outputs[i].file_name  
        <<": Label: " << _labels[top_outputs[i].idx] << std::endl;
    } 
      ++_num_samples;
    }
}