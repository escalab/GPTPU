#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>
#include <chrono>
#include "cnpy.h"
#include "make_temp.h"
#include "offset.h"
//##include <dense.h> // created within edgetpu/
#include <complex>
#include <float.h>
#include <sys/sysinfo.h>
#include <math.h>
#include <omp.h>

//#define MAX(a, b) ((a) > (b) ? (a) : (b))

unsigned long long int get_file_size(const std::string& file_path){
  std::fstream f(file_path, std::ios::binary|std::ios::in|std::ios::ate);
  unsigned long long int file_size;
  if(f.is_open()){
    file_size = f.tellg();
  }else{
    std::cout << __func__ << ": file open fail. " << file_path << std::endl;
  }
  f.close();
  assert(file_size >= 0);
  return file_size;
}

int make_mm2conv_temp(const std::string& in_path, const std::string& out_path, int blk_A, int blk_B, int blk_C){
  std::cout << __func__ << ": in_path: " << in_path << ", out_path: " << out_path << std::endl;
  std::fstream in, out;
  unsigned long long int file_size = get_file_size(in_path);
  in.open(in_path, std::ios::binary|std::ios::in|std::ios::out|std::ios::ate);
  out.open(out_path, std::ios::binary|std::ios::in|std::ios::out|std::ios::trunc);
  if(!in.is_open()){
    std::cout << __func__ << ": file open fail. " << in_path << std::endl;
    exit(0);
  }
  if(!out.is_open()){
    std::cout << __func__ << ": file open fail. " << out_path << std::endl;
    exit(0);
  }
  unsigned long long int before_data, after_data;
  if(blk_A == blk_B && blk_B == blk_C && blk_C == 1024){
    before_data = offset::mm2conv::oneKs::before_data;
    after_data  = offset::mm2conv::oneKs::after_data;
  }else if(blk_A == 1 && blk_B == blk_C && blk_C == 1024){
    before_data = offset::mm2conv::oneKmv::before_data;
    after_data  = offset::mm2conv::oneKmv::after_data;
  }else if(blk_A == blk_B && blk_B == blk_C && blk_C == 256){
    before_data = offset::mm2conv::mm_256::before_data;
    after_data  = offset::mm2conv::mm_256::after_data;
  }else if(blk_A == blk_B && blk_B == blk_C && blk_C == 128){
    before_data = offset::mm2conv::mm_128::before_data;
    after_data  = offset::mm2conv::mm_128::after_data;
  }else if(blk_A == 4096 && blk_B == 256 && blk_C == 4096){
    before_data = offset::mm256conv::b16::before_data;
    after_data  = offset::mm256conv::b16::after_data;
  }else if(blk_A == 2048 && blk_B == 256 && blk_C == 2048){
    before_data = offset::mm256conv::b8::before_data;
    after_data  = offset::mm256conv::b8::after_data;
  }else{
    std::cout << __func__ << ": the shape: " << blk_A << "x" << blk_B << "x" << blk_C << " is not supported yet." << std::endl;
    exit(0);
  }

  unsigned long long int max_len = MAX(before_data, file_size - after_data);
  char *buf = new char[max_len];
  // copy first section before data section
  in.seekg(0);
  out.seekp(0);
  in.read(&buf[0], before_data);
  out.write(&buf[0], before_data);
  // copy section after data section
  in.seekg(after_data);
  out.seekp(before_data);
  in.read(&buf[0], file_size - after_data);
  out.write(&buf[0], file_size - after_data);
  
  delete [] buf;
  in.close();
  out.close();
  return 0;
}

int make_dense_temp(const std::string& in_path, const std::string& out_path){
  std::fstream in, out;
  unsigned long long int file_size = get_file_size(in_path);
  in.open(in_path, std::ios::binary|std::ios::in|std::ios::out|std::ios::ate);
  out.open(out_path, std::ios::binary|std::ios::in|std::ios::out|std::ios::trunc);
  if(!in.is_open()){
    std::cout << "make_temp: file open fail. " << in_path << std::endl;
  }
  if(!out.is_open()){
    std::cout << "make_temp: file open fail. " << out_path << std::endl;
  }

  // get data size
  uint32_t size_buffer[2];
  in.seekg(file_size - offset::dense_meta);
  in.read(reinterpret_cast<char *>(&size_buffer[0]), sizeof(size_buffer));
  // need buffer
  std::cout << "making template...: " << size_buffer[0] << ", " << size_buffer[1] << std::endl;
  int input_size  = size_buffer[1];
  int output_size = size_buffer[0];

  // special offset for size_buufer[0] == 1 case:
  unsigned long long int dense_data, dense_4out, dense_bias;
  if(size_buffer[0] == 1){
    dense_data = offset::dense_data + 0x4;
    dense_4out = offset::dense_4out + 0x4;
    dense_bias = offset::dense_bias + 0x4;
  }else{
    dense_data = offset::dense_data;
    dense_4out = offset::dense_4out;
    dense_bias = offset::dense_bias;
  }

  unsigned long long int data_size = (unsigned long long int)(size_buffer[0] * size_buffer[1]);
  // remain size depends on offset distribution, vary from op to op
  unsigned long long int remain_size = file_size - dense_bias - output_size * 4 - data_size;
  unsigned long long int max_len = MAX(dense_data, remain_size);

//  std::cout << "dense_data: " << std::to_string(offset::dense_data) << ", addition: " << std::to_string((size_buffer[0] == 1)?(0x4):(0x0)) << "sum up: " << std::to_string(offset::dense_data + (size_buffer[0] == 1)?(0x4):(0x0)) << std::endl;

  char *buf = new char[max_len];
  // copy first section before data section
  in.seekg(0);
  out.seekp(0);
  in.read(&buf[0], dense_data);
  out.write(&buf[0], dense_data);
  // copy section between actual data and actual bias
  in.seekg(dense_data + data_size);
  out.seekp(dense_data);
  unsigned long long int data2bias_size = dense_4out - (dense_data);
  in.read(&buf[0], data2bias_size);
  out.write(&buf[0], data2bias_size);
  // copy remaining section after bias data
  in.seekg(file_size - remain_size);
  out.seekp(dense_bias);
  in.read(&buf[0], remain_size);
  out.write(&buf[0], remain_size);
  // modify data size (the meta) as zero
  unsigned long long int out_file_size = file_size - data_size - 4 * output_size;
  char* zeros = new char[2*sizeof(float)](); // initialized as zero
  out.seekp(out_file_size - offset::dense_meta);
  out.write(&zeros[0], 2*sizeof(float));  
  // modify bias size (the meta) as zero
  out.seekp(out_file_size - offset::dense_bias_size);
  out.write(&zeros[0], sizeof(float));
  // modify flatten size (the meta) as zero
  out.seekp(out_file_size - offset::dense_flatten);
  out.write(&zeros[0], sizeof(float));
  // modify Matmul size (the meta) as zero
  out.seekp(out_file_size - offset::dense_matmul);
  out.write(&zeros[0], sizeof(float));

  delete [] buf;
  delete [] zeros;
  in.close();
  out.close();
  return 0;
}

int make_temp(std::string model_name, const std::string& in_path, const std::string& out_path){
  std::fstream in, out;
  unsigned long long int file_size = get_file_size(in_path);
  in.open(in_path, std::ios::binary|std::ios::in|std::ios::out|std::ios::ate);
  out.open(out_path, std::ios::binary|std::ios::in|std::ios::out|std::ios::trunc);
  if(!in.is_open()){
    std::cout << "make_temp: file open fail. " << in_path << std::endl;
  }
  if(!out.is_open()){
    std::cout << "make_temp: file open fail. " << out_path << std::endl;
  }
  // copy the whole file first
  char *buf = new char[file_size];
  in.seekg(0);
  out.seekp(0);
  in.read(&buf[0], file_size);
  out.write(&buf[0], file_size);
  // modify sizes (the meta) as zero  
  char* zeros = new char[sizeof(uint32_t)](); // initialized as zero
  unsigned long long int* offsets;
  int size = 0;
  std::cout << "model_name: " << model_name << std::endl;
  if(model_name == "crop_model"){
    offsets = offset::crop_offsets;
    size = CROP_OFFSETS_CNT;
  }else if(model_name == "sub_model"){
    offsets = offset::sub_offsets;
    size = SUB_OFFSETS_CNT;
  }
  for(int i = 0 ; i < size ; i++){
    out.seekp(offsets[i]);
    out.write(&zeros[0], sizeof(uint32_t));  
  }

  delete [] zeros;
  delete [] buf;
  in.close();
  out.close();
  return 0;
}


