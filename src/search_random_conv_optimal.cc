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
    std::cout << "Usage: ./search_random_conv_optimal [in_dir] [iter] [0-index section]" << std::endl;
    return 1;
  }
  int idx = 1;
  std::string in_dir = argv[idx++]; // operation name
  int iter           = atoi(argv[idx++]); // iteration
  int sec            = atoi(argv[idx++]); // 0-index secton with size predefined as 50000
  search_random_conv_optimal(in_dir, iter, sec);  
  return 0;
}
