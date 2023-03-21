// Example to run a model using one Edge TPU.
// It depends only on tflite and edgetpu.h

// Copyright 2020 Google LLC
// Modified by Nam Vu 2020

#include <algorithm>
#include <chrono>  // NOLINT
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <ostream>
#include <regex>
#include <string>

//#include "edgetpu.h"
#include "model_utils.h"
//#include "tensorflow/lite/interpreter.h"
//#include "tensorflow/lite/model.h"

typedef std::chrono::time_point<std::chrono::high_resolution_clock> timing;
typedef std::chrono::high_resolution_clock clk;

int main(int argc, char* argv[]) {

    if (argc != 6) {
        std::cout << "Usage: " << argv[0] << " <edgetpu model> <input size> <output size> <iter> <enable_log>" << std::endl;
        return 1;
    }
    
    model_utils::DeviceHandler device_handler;

    int total_dev_cnt = device_handler.list_devices(1/*verbose*/);
    device_handler.open_device(0/*tpu_id*/, 1/*verbose*/);

    const std::string model_path = argv[1];
    int input_size = atoi(argv[2]);
    int output_size = atoi(argv[3]);
    int iter = atoi(argv[4]);
    int enable_log = atoi(argv[5]);
    
    // Build model.
    device_handler.build_model(model_path, 0/*model_id*/);

    // Build interpreter.
    device_handler.build_interpreter(0/*tpu_id*/, 0/*model_id*/);

    // random input data generation
    uint8_t* input = (uint8_t*) malloc(input_size * sizeof(uint8_t));
    for(int i = 0 ; i < input_size ; i++){
    	input[i] = (uint8_t)(rand()%256);
    }
    device_handler.populate_input(input, input_size, 0/*model_id*/);

    // Inference.
    timing start = clk::now();
    device_handler.model_invoke(0/*model_id*/, iter/*iter*/);
    timing end = clk::now();
    
    // Get output.
    uint8_t* output = (uint8_t*) malloc(output_size * sizeof(uint8_t));
    uint8_t zero_point;
    float scale;

    device_handler.get_raw_output(output, output_size, 0/*model_id*/, zero_point, scale);

    // default data checking
    int print_count = 10;
    if(output_size < print_count){
    	print_count = output_size;
    }
    std::cout << __func__ << ": zero_point: " 
	      << (unsigned)zero_point 
	      << ", scale: " 
	      << scale << std::endl;
    for(int i = 0 ; i < print_count ; i++){
	    std::cout << "input[" << i << "]: " 
		      << (unsigned)input[i] 
		      << ", raw output[" << i << "]: " 
		      << (unsigned)output[i] << std::endl;
    }

    double kernel_time = 
	    (std::chrono::duration_cast<std::chrono::nanoseconds>
            (end - start).count()/1e6)/iter;
    std::cout << "kernel time: " << kernel_time << " (ms), average over " 
              << iter << " time(s)." << std::endl;

    if(enable_log){
        std::ofstream outfile;
        outfile.open("minimal_log.csv", std::ios_base::app);
        outfile << model_path << ",avg kernel time (ms)," << std::to_string(kernel_time)
        << ",iter," << std::to_string(iter) << std::endl;
    }

    return 0;
}
