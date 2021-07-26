#include <algorithm>
#include <cmath>
#include <chrono>  // NOLINT
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <sys/mman.h>
#include <ostream>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <string>
#include <time.h>
#include <iomanip>
#include "dense.h"
#include "src/cpp/examples/Mymodel_utils.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <mutex>
#define BREAKDOWN 1

typedef std::chrono::time_point<std::chrono::high_resolution_clock> timing;
typedef std::chrono::high_resolution_clock clk;
struct Itpr{ // This turns out to be unnecessary
  int tpu_id;
  int model_id;
  std::shared_ptr<tflite::Interpreter>  interpreter;
};
std::mutex mylock;
std::mutex itpr_lock;
//std::vector<std::shared_ptr<tflite::Interpreter>>             interpreter;

std::vector<Itpr>                                             Interpreter;
std::vector<std::shared_ptr<tflite::FlatBufferModel>>         model;
std::vector<edgetpu::EdgeTpuManager::DeviceEnumerationRecord> enumerate_edgetpu;
std::vector<std::shared_ptr<edgetpu::EdgeTpuContext>>         edgetpu_context;
std::vector<uint8_t> input;
std::vector<uint8_t> input2;
//std::vector<int*>    result; // dicarded, only used by run_model()

void edgetpu_cleanup(){
  Interpreter.clear();
  model.clear();
}

void set_verbose(int verbose){
  edgetpu::EdgeTpuManager::GetSingleton()->SetVerbosity(verbose);
}

void dense_set_chunk_size(int chunk_size){
  coral::util_set_chunk_size(chunk_size);
}

int dense_get_chunk_size(){
  return coral::util_get_chunk_size();
}

long long int ListTheDevices(int verbose, int& count){
  timing start = clk::now();
  if(verbose >= 0){
    std::cerr << "===== Select edgetpu device... ========" << std::endl;
  }
  enumerate_edgetpu = edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();  
  edgetpu_context.resize(enumerate_edgetpu.size());
  //result.resize(enumerate_edgetpu.size());
  count = enumerate_edgetpu.size();
  if(verbose >= 0){
    std::cerr << "Enumerated edgetpu: " << std::endl;
    for (auto it = enumerate_edgetpu.begin(); it != enumerate_edgetpu.end(); it++){
      std::cerr << it->path << std::endl;
    }
  }
  timing end   = clk::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

void close_device(int tpuid, int verbose){
  if(tpuid >= (int)edgetpu_context.size()){
    std::cout << "tpuid: " << tpuid << " is not exist, can't close." << std::endl;
    exit(0);
  }
  edgetpu_context[tpuid].reset();
}

long long int open_device(int tpuid, int verbose){
  timing start = clk::now();
  if (tpuid >= (int)enumerate_edgetpu.size()){
    std::cout << "tpuid \'" << tpuid<< "\' is out of scope." << std::endl;
    exit(0);
  }
  edgetpu_context[tpuid] = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice(enumerate_edgetpu[tpuid].type, enumerate_edgetpu[tpuid].path);
  if(verbose >= 0){
    std::cout << "Opened device:\ntype: " << enumerate_edgetpu[tpuid].type << ", path: " << enumerate_edgetpu[tpuid].path << std::endl;
  }
  timing end   = clk::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

//int interpreter_initialization(int model_cnt){
//  timing s = clk::now();
////  Interpreter.resize(model_cnt);
////  model.resize(model_cnt);
//  timing e = clk::now();
//  return std::chrono::duration_cast<std::chrono::nanoseconds>(e - s).count();
//}

int read_input(const std::string& input_path, int input_mode, int model_id){
  timing start = clk::now();
  const auto& required_shape = coral::GetInputShape(*Interpreter[model_id].interpreter, 0);
  int length = required_shape[0];
  if(input_mode == 0){
    input.resize(length);
  }else{
    input2.resize(length);
  }
  int fd = open(input_path.c_str(), O_RDWR);
  struct stat st;
  if(fd < 0){
    std::cout << "check_input: in file opening fail." << input_path << std::endl;
  }
  fstat(fd, &st);
  char* map;
  map = static_cast<char*>(mmap(NULL, length*sizeof(char), PROT_READ, MAP_SHARED, fd , 0));
  assert(map != MAP_FAILED);
  for(int i = 0 ; i < length ; i++){
    if(input_mode == 0){
      input[i]  = (uint8_t)map[i];
    }else if(input_mode == 1){
      input2[i] = (uint8_t)map[i];
    }
  }
  timing end   = clk::now();
  //delete buf;
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

long long int build_model_from_buffer(char* buf, size_t size, int model_id){
  timing start = clk::now();
  std::shared_ptr<tflite::FlatBufferModel> local_model_tmp;
  local_model_tmp = tflite::FlatBufferModel::BuildFromBuffer(buf, size);
  if (local_model_tmp == nullptr) {
    std::cerr << "Fail to build FlatBufferModel[" << model_id << "] from buffer with size: " << size << std::endl;
    std::abort();
  }
  if(model.size() < model_id+1){
    model.resize(model_id+1);
  }
  model[model_id] = local_model_tmp;
  timing end   = clk::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

}

long long int build_model(const std::string& model_path, int model_id){
  timing start = clk::now();
  std::shared_ptr<tflite::FlatBufferModel> local_model_tmp;
  local_model_tmp = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  if (local_model_tmp == nullptr) {
    std::cerr << "Fail to build FlatBufferModel from file: " << model_path << std::endl;
    std::abort();
  }
  if(model.size() < model_id+1){
    model.resize(model_id+1);
  }
  model[model_id] = local_model_tmp;
  timing end   = clk::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

long long int build_interpreter(int tpuid, int model_id){
  timing start = clk::now();
  Itpr tmp;
  tmp.interpreter = coral::BuildEdgeTpuInterpreter(*model[model_id], edgetpu_context[tpuid].get());
  if(tmp.interpreter == nullptr){
    std::cerr << "Fail to build interpreter ( tpu_id: " << tpuid << ", model_id: " << model_id << ")" << std::endl;
    std::abort();
  }
  tmp.tpu_id      = tpuid;
  tmp.model_id    = model_id;
  if(Interpreter.size() < model_id+1){
    Interpreter.resize(model_id+1);
  }
  Interpreter[model_id] = tmp;
  timing end   = clk::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

//int run_model(int iter, int* output_size, int model_id, int verbose){
//  timing start = clk::now();
//  *output_size = coral::MyRunInference(&result[0], input, iter, Interpreter[model_id].interpreter.get(), verbose);
//  timing end   = clk::now();
//  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
//}

int run_modelV2(int* b, int size, int iter, int& output_size, int* per_result, int tpu_id, int model_id, const std::string& data_type, int w_chunk_idx, int in_chunk_idx, int verbose, long long int& mem_ns, long long int& run_ns, long long int& pop_ns){
  timing mem_start = clk::now();
  uint8_t* input = Interpreter[model_id].interpreter.get()->typed_input_tensor<uint8_t>(0);
  int mean_value = (!data_type.compare("uint8"))?0:((!data_type.compare("int8"))?128:0/*unexpected type*/);
  int chunk_size = dense_get_chunk_size();
  int chunk_mask = (~(0xffffffff << chunk_size)) << (in_chunk_idx * chunk_size);
  int temp;
//  std::cout << "chunk size: " << chunk_size << std::endl;
  for(int i = 0 ; i < size ; i++){
    temp = b[i] & chunk_mask;
    temp >>= (in_chunk_idx * chunk_size);
    assert(temp >= 0 || temp < UCHAR_MAX);
    input[i] = (uint8_t)temp + mean_value;
//    if(i < 10)std::cout << "input[" << i << "]: " << (unsigned)input[i] << ", b: " << b[i] << ", tmep: " << temp << std::endl;
  }
  timing mem_end = clk::now();
//  std::cout << __func__ << ": mdoel_id: " << model_id << std::endl;
  output_size = coral::MyRunInferenceV2(per_result, iter, Interpreter[model_id].interpreter.get(), w_chunk_idx, in_chunk_idx, verbose, run_ns, pop_ns, true);
  mem_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(mem_end - mem_start).count();
  return 1;
}

int populate_input(int *b, int size, int model_id){
  uint8_t* input = Interpreter[model_id].interpreter.get()->typed_input_tensor<uint8_t>(0);
  std::cout << __func__ << ": start populating..., size = " << size << std::endl;
  timing mem_start = clk::now();
  for(int i = 0 ; i < size ; i++){
    assert(b[i] >= 0 || b[i] < UCHAR_MAX);
    input[i] = (uint8_t)b[i] + 128; // when create_model.py uses --mean-value=128
  }
  timing mem_end = clk::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(mem_end - mem_start).count();
}

int populate_input_16x8(int *b, int size, int model_id){ // experimental
  int16_t* input = Interpreter[model_id].interpreter.get()->typed_input_tensor<int16_t>(0);
  std::cout << __func__ << ": start populating..." << std::endl;
  timing mem_start = clk::now();
  for(int i = 0 ; i < size ; i++){
    assert(b[i] >= 0 || b[i] < UCHAR_MAX);
    input[i] = (int16_t)b[i] + 128; // when create_model.py uses --mean-value=128
  }
  timing mem_end = clk::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(mem_end - mem_start).count();
}

int populate_input_uint8(uint8_t *b, int size, int model_id){
  uint8_t* input = Interpreter[model_id].interpreter.get()->typed_input_tensor<uint8_t>(0);
  std::cout << __func__ << ": start populating..." << std::endl;
  timing mem_start = clk::now();
//TODO: optimize
  for(int i = 0 ; i < size ; i++){
    input[i] = (uint8_t)b[i] /*+ 128*/; // when create_model.py uses --mean-value=128
  }
  timing mem_end = clk::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(mem_end - mem_start).count();
}

// can be called in parallel since data preparation doesn't matter
int populate_input_exact(int *b, int size_A, int size_C, int chunk_num, int model_id, const std::string& data_type){
  timing mem_start = clk::now();
  uint8_t* input = Interpreter[model_id].interpreter.get()->typed_input_tensor<uint8_t>(0);
  int chunk_size = dense_get_chunk_size();
  if(chunk_size != 1) std::cout << __func__ << ": warning: chunk_size is: " << chunk_size << " for exact mode" << std::endl;
  int chunk_mask;
  int mean_value = (!data_type.compare("uint8"))?0:((!data_type.compare("int8"))?128:0/*unexpected type*/);
  int temp;
  int cnt = 0;
  for(int chunk_idx = 0 ; chunk_idx < chunk_num ; chunk_idx++){
    chunk_mask = (~(0xffffffff << chunk_size)) << (chunk_idx * chunk_size);
    for(int i = 0 ; i < size_A*size_C ; i++){
  /*
      do chunking proces on input data, filter out non chunking bits for this chunking_idx
  */
      temp = b[cnt] & chunk_mask;
      temp >>= (chunk_idx * chunk_size);  // 0-based chunk_dix starts from LSB
  //    temp = (b[i] & chunk_mask) >> (chunk_idx * chunk_size);
      assert(temp >= 0 || temp < UCHAR_MAX);
      input[cnt] = (uint8_t)temp + mean_value; // when create_model.py uses --mean-value=128
//      if(temp != 0){std::cout << __func__ << ": b[" << cnt << "]: " << b[cnt] << ", input: " << (unsigned)input[cnt] << ", temp: " << temp << std::endl;}
      cnt++;
    }
  }
  timing mem_end = clk::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(mem_end - mem_start).count();
}
int populate_input_chunking(int *b, int size, int model_id, int chunk_idx, const std::string& data_type){
  timing mem_start = clk::now();
  uint8_t* input = Interpreter[model_id].interpreter.get()->typed_input_tensor<uint8_t>(0);
  int chunk_size = dense_get_chunk_size();
  int chunk_mask = (~(0xffffffff << chunk_size)) << (chunk_idx * chunk_size);
  int mean_value = (!data_type.compare("uint8"))?0:((!data_type.compare("int8"))?128:0/*unexpected type*/);
  int temp;
  for(int i = 0 ; i < size ; i++){
/*
    do chunking proces on input data, filter out non chunking bits for this chunking_idx
*/
//    if(i == 0)std::cout << "b[0]: " << b[i] << ", chunk_idx: " << chunk_idx << ", chunk_size: " << chunk_size << std::endl;
    temp = b[i] & chunk_mask;
    temp >>= (chunk_idx * chunk_size);  // 0-based chunk_dix starts from LSB
//    temp = (b[i] & chunk_mask) >> (chunk_idx * chunk_size);
    assert(temp >= 0 || temp < UCHAR_MAX);
    input[i] = (uint8_t)temp + mean_value; // when create_model.py uses --mean-value=128
//    if(i == 0)std::cout << __func__ << ", input[" << i << "]: " << (unsigned)input[i] << ", b[" << i << "]: " << b[i] << ", chunk_idx: " << chunk_idx << std::endl;
  }
  timing mem_end = clk::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(mem_end - mem_start).count();
}

long long int invoke_model(int model_id, int iter){
  double us = 0;
  timing ms, me;
  timing run_s = clk::now();
  Interpreter[model_id].interpreter.get()->Invoke(); 
  timing run_e = clk::now();
  if(iter > 1){
    for(int i = 0 ; i < iter -1 ;i++){
      ms = clk::now();
      Interpreter[model_id].interpreter.get()->Invoke(); 
      me = clk::now();
      us += std::chrono::duration_cast<std::chrono::nanoseconds>(me - ms).count();
//std::cout << "i:" << i << ", us: " << us << std::endl;
    }
  }
  if(iter == 1){
    return std::chrono::duration_cast<std::chrono::nanoseconds>(run_e - run_s).count();
  }else{
    return us + std::chrono::duration_cast<std::chrono::nanoseconds>(run_e - run_s).count();
  }
}

//out-dated
//int populate_output(int* partial_result, int A, int C, int blk_A, int blk_C, int i, int k, int model_id){
//  timing pop_s = clk::now();
//  coral::My_populate_output(partial_result, A, C, blk_A, blk_C, i, k, Interpreter[model_id].interpreter.get());
//  timing pop_e = clk::now();
//  return std::chrono::duration_cast<std::chrono::nanoseconds>(pop_e - pop_s).count();
//}

int simple_populate_output(int* result, int model_id, int verbose){
  timing s = clk::now();
  coral::My_simple_output(result, Interpreter[model_id].interpreter.get(), verbose);
  timing e = clk::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(e-s).count();
}

int populate_output_exact(int* partial_result, int A, int C, int blk_A, int blk_C, int i, int k, int model_id, int chunk_num,float SCALE){
  timing pop_s = clk::now();
  coral::My_populate_output_exact(partial_result, A, C, blk_A, blk_C, i, k, Interpreter[model_id].interpreter.get(), chunk_num, SCALE);
  timing pop_e = clk::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(pop_e - pop_s).count();
}

int populate_output_chunking(int* partial_result, int A, int C, int blk_A, int blk_C, int i, int k, int model_id, int in_chunk_idx, int w_chunk_idx, float SCALE){
  timing pop_s = clk::now();
  coral::My_populate_output_chunking(partial_result, A, C, blk_A, blk_C, i, k, Interpreter[model_id].interpreter.get(), in_chunk_idx, w_chunk_idx, SCALE);
  timing pop_e = clk::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(pop_e - pop_s).count();
}

int run_modelV3(int* b, int size, int iter, int& output_size, int* partial_result, int A, int C, int blk_A, int blk_C, int i/*ROW_BLK_CNT idx*/, int k/*COL_BLK_CNT idx*/, int tpu_id,  int model_id, int verbose, long long int& mem_ns, long long int& run_ns, long long int& pop_ns){
  uint8_t* input = Interpreter[model_id].interpreter.get()->typed_input_tensor<uint8_t>(0);
  timing mem_start = clk::now();
  for(int i = 0 ; i < size ; i++){
    assert(b[i] >= 0 || b[i] < UCHAR_MAX);
    input[i] = (uint8_t)b[i] + 128; // when create_model.py uses --mean-value=128
  }
  timing mem_end = clk::now();
  output_size = coral::MyRunInferenceV3(partial_result, A, C, blk_A, blk_C, i, k, iter, Interpreter[model_id].interpreter.get(), verbose, run_ns, pop_ns);
  mem_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(mem_end - mem_start).count();
  return 1;
}

int run_element_wise_modelV2(int* a, int* b, int size, int iter, int& output_size, int* per_result, int tpu_id,  int model_id, int verbose, long long int& mem_ns, long long int& run_ns, long long int& pop_ns){
  uint8_t* input = Interpreter[model_id].interpreter.get()->typed_input_tensor<uint8_t>(0);
  timing mem_start = clk::now();
  for(int i = 0 ; i < size ; i++){
    assert(b[i] >= 0 || b[i] < UCHAR_MAX);
    input[i] = (uint8_t)a[i];
  }
  for(int i = 0 ; i < size ; i++){
    assert(b[i] >= 0 || b[i] < UCHAR_MAX);
    input[size+i] = (uint8_t)b[i]; /*appending second matrix*/
  }
  timing mem_end = clk::now();
  output_size = coral::MyRunInferenceV2(per_result, iter, Interpreter[model_id].interpreter.get(), 0, 0, verbose, run_ns, pop_ns, false);
  mem_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(mem_end - mem_start).count();
  return 1;
}

int run_element_wise_modelV3(int* a, int* b, int size, int iter, int& output_size, int* per_result, int tpu_id, int model_id, int w_chunk_idx, int in_chunk_idx, int verbose, long long int& mem_ns, long long int& run_ns, long long int& pop_ns){
  uint8_t* input = Interpreter[model_id].interpreter.get()->typed_input_tensor<uint8_t>(0);
  timing mem_start = clk::now();
  for(int i = 0 ; i < size ; i++){
    assert(a[i] >= 0 || a[i] < UCHAR_MAX);
    input[i] = (uint8_t)a[i];
  }
  for(int i = 0 ; i < size ; i++){
    assert(b[i] >= 0 || b[i] < UCHAR_MAX);
    input[size+i] = (uint8_t)b[i]; /*appending second matrix*/
  }
  timing mem_end = clk::now();
  output_size = coral::MyRunInferenceV2(per_result, iter, Interpreter[model_id].interpreter.get(), w_chunk_idx, in_chunk_idx, verbose, run_ns, pop_ns, true);
  mem_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(mem_end - mem_start).count();
  return 1;
}

int populate_element_wise_input_chunking(int* a, int* b, int size, int xi, int yi, int chunk_size, int model_id, std::string& model_name, long long int& mem_ns){
  uint8_t* input = Interpreter[model_id].interpreter.get()->typed_input_tensor<uint8_t>(0);
  timing mem_start = clk::now();
  if(!model_name.compare("mul_model") || !model_name.compare("add_model")){
    int mask1 = (~(0xffffffff << chunk_size)) << (xi * chunk_size);
    int mask2 = (~(0xffffffff << chunk_size)) << (yi * chunk_size);
    for(int i = 0 ; i < size ;i++){
      input[i]      = (uint8_t)((a[i] & mask1) >> (xi * chunk_size));
      input[size+i] = (uint8_t)((b[i] & mask2) >> (yi * chunk_size));
    }
  }else{
    for(int i = 0 ; i < size ; i++){
      assert(a[i] >= 0 || a[i] < UCHAR_MAX);
      input[i] = (uint8_t)a[i];
    }
    for(int i = 0 ; i < size ; i++){
      assert(b[i] >= 0 || b[i] < UCHAR_MAX);
      input[size+i] = (uint8_t)b[i]; /*appending second matrix*/
    }
  }
  timing mem_end = clk::now();
  mem_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(mem_end - mem_start).count();
  return 1;
}

int populate_mm2mul_input_chunking(int* a, int blk_A, int blk_B, unsigned long long int offset, int* b, int idx, int i , int j, int ROW_BLK_CNT, int COL_BLK_CNT, int size, int xi, int yi, int chunk_size, int model_id, std::string& model_name, long long int& mem_ns){
  uint8_t* input = Interpreter[model_id].interpreter.get()->typed_input_tensor<uint8_t>(0);
  timing mem_start = clk::now();
  unsigned long long int global_size = blk_A*ROW_BLK_CNT*blk_B*COL_BLK_CNT;
  unsigned long long int new_offset  = idx*(blk_B*COL_BLK_CNT)+offset;
  if(!model_name.compare("mul_model") || !model_name.compare("add_model")){
    int mask1 = (~(0xffffffff << chunk_size)) << (xi * chunk_size);
    int mask2 = (~(0xffffffff << chunk_size)) << (yi * chunk_size);
    for(int i = 0 ; i < size ;i++){
//TODO: rotate input a 
      new_offset = (idx*(blk_B*COL_BLK_CNT)+offset+i)%(global_size); 
      input[i]      = (uint8_t)((a[new_offset] & mask1) >> (xi * chunk_size));
      input[size+i] = (uint8_t)((b[i         ] & mask2) >> (yi * chunk_size));
    }
  }else{
    for(int i = 0 ; i < size ; i++){
      assert(a[i] >= 0 || a[i] < UCHAR_MAX);
      input[i] = (uint8_t)a[i];
    }
    for(int i = 0 ; i < size ; i++){
      assert(b[i] >= 0 || b[i] < UCHAR_MAX);
      input[size+i] = (uint8_t)b[i]; /*appending second matrix*/
    }
  }
  timing mem_end = clk::now();
  mem_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(mem_end - mem_start).count();
  return 1;
}

int run_element_wise_modelV4(int* a, int* b, int size, int iter, int& output_size, int* per_result, int tpu_id, int model_id, int w_chunk_idx, int in_chunk_idx, int verbose, long long int& mem_ns, long long int& run_ns, long long int& pop_ns){
  uint8_t* input = Interpreter[model_id].interpreter.get()->typed_input_tensor<uint8_t>(0);
  timing mem_start = clk::now();
  for(int i = 0 ; i < size ; i++){
    assert(a[i] >= 0 || a[i] < UCHAR_MAX);
    input[i] = (uint8_t)a[i];
  }
  for(int i = 0 ; i < size ; i++){
    assert(b[i] >= 0 || b[i] < UCHAR_MAX);
    input[size+i] = (uint8_t)b[i]; /*appending second matrix*/
  }
  timing mem_end = clk::now();
  output_size = coral::MyRunInferenceV2(per_result, iter, Interpreter[model_id].interpreter.get(), w_chunk_idx, in_chunk_idx, verbose, run_ns, pop_ns, false);
  mem_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(mem_end - mem_start).count();
  return 1;
}

long long int populate_element_wise_output(int* partial_result, int size, int model_id, float SCALE){
  timing pop_s = clk::now();
  coral::My_populate_element_wise_output(partial_result, size, Interpreter[model_id].interpreter.get(), SCALE);
  timing pop_e = clk::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(pop_e - pop_s).count();
}
long long int populate_element_wise_output_chunking(int* partial_result, int size, int model_id, int in_chunk_idx, int w_chunk_idx, float SCALE){
  timing pop_s = clk::now();
  coral::My_populate_element_wise_output_chunking(partial_result, size, Interpreter[model_id].interpreter.get(), in_chunk_idx, w_chunk_idx, SCALE);
  timing pop_e = clk::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(pop_e - pop_s).count();
}
