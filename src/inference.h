#ifndef INFERENCE_H
#define INFERENCE_H

#include <mutex>
#include <typeinfo>
#include <queue>
#include <vector>
#include <thread>
#include <memory>
#include <condition_variable>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <torch/script.h>


#define kTOP_K 1
#define kIMAGE_SIZE 224
#define kCHANNELS 3

template <class T>
class MessageQueue
{
	public:
		void send(T && msg);
        T receive();
        int get_count();
    private:
    	std::deque<T> _queue;
        std::mutex _mutex;
        std::condition_variable _condition;
};


struct prediction_t
{
	int idx;
   	float prob;
    std::string file_name;
    std::thread::id thread_id;
};


class Inference
{
	public:
    	Inference();  // constructor
        ~Inference();  // destructor
		Inference(const Inference &);   // copy const
        Inference & operator=(const Inference &);   // copy assign const
        Inference(Inference &&);  // move const
        Inference & operator=(Inference &&);  // move assign const
        
        void predict(std::vector<std::string> &&file_names, int workers);
        void display(int num_samples);
    
    private:
    	std::vector<std::thread> _threads; // holds all threads that have been launched within this object
    	MessageQueue<std::vector<prediction_t>> _predictions;
        std::vector<std::string> _labels;
        std::shared_ptr<torch::jit::script::Module> _module;
        int _num_samples;
        std::mutex _mutex;
        
        void worker(std::vector<std::string> file_names);        
        void LoadImageNetLabel(std::string file_name);
        void LoadImage(std::string file_name, cv::Mat &image);
 };
#endif