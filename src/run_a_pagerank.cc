#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <gptpu.h>

int main(int argc, char* argv[]){
  if(argc != 4){
    std::cout << "argc = " << argc << std::endl;
    std::cout << "Usage: ./run_a_pagerank [model_path] [iter] [input_size]" << std::endl;
    return 1;
  }
  int idx = 1;
  std::string model_path = argv[idx++]; // operation name
  int iter               = atoi(argv[idx++]); // iteration
  int input_size         = atoi(argv[idx++]);
  run_a_pagerank(model_path, iter, input_size);  
  return 0;
}
