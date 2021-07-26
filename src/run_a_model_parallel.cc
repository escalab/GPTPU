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
    std::cout << "Usage: ./run_a_model_parallel [model_path] [iter] [dev_cnt]" << std::endl;
    return 1;
  }
  int idx = 1;
  std::string model_path = argv[idx++]; // operation name
  int iter               = atoi(argv[idx++]); // iteration
  int dev_cnt            = atoi(argv[idx++]);
  run_a_model_parallel(model_path, iter, dev_cnt);  
  return 0;
}
