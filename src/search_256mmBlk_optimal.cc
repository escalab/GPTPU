#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <gptpu.h>

int main(int argc, char* argv[]){
  if(argc != 3){
    std::cout << "argc = " << argc << std::endl;
    std::cout << "Usage: ./search_256mmBlk_optimal [in_dir] [iter]" << std::endl;
    return 1;
  }
  int idx = 1;
  std::string in_dir = argv[idx++]; // operation name
  int iter           = atoi(argv[idx++]); // iteration
  search_256mmblk_optimal(in_dir, iter);  
  return 0;
}
