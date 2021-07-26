#include <errno.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
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
#include "gptpu.h"
#include "make_temp.h"
#include "utils.h"
#include "offset.h"
#include "fifo.h"
#include "make_model.h"
#include <dense.h> // created within edgetpu/
#include <complex>
#include <float.h>
#include <sys/sysinfo.h>
#include <math.h>
#include <omp.h>
#include <pthread.h>
#include <thread>
#include <mutex>
#include <glob.h>
#include <stdarg.h>

#define itoa(x) std::to_string(x)
#define ASSERT(condition, message) do {\
if(!(condition)){printf("%s", message);}\
assert((condition)); } while(false)
#define sub_cnt(size, chunk_size) ((size/chunk_size)+((size%chunk_size!=0)?1:0))

inline void SET_BLK(openctpu_buffer* ret, int A, int B, int C, int& ROW_BLK_CNT, int& INN_BLK_CNT, int& COL_BLK_CNT, int& ROW_BLK_REM, int& INN_BLK_REM, int& COL_BLK_REM){   
 ret->set_BLK_CNTs(A, B, C);
 ret->get_BLK_CNTs(ROW_BLK_CNT, INN_BLK_CNT, COL_BLK_CNT, ROW_BLK_REM, INN_BLK_REM, COL_BLK_REM);
}

typedef unsigned long long int long_int; // TODO: can be renamed if needed
typedef std::chrono::time_point<std::chrono::high_resolution_clock> timing; 
typedef std::chrono::high_resolution_clock clk; 

// ========== runtime ==========
std::vector<Flatbufs> flatbufs;

pthread_mutex_t pmtx;
pthread_cond_t in_CV, out_CV, end_CV;
int qin, qout, queue_cnt = 0;
volatile bool done_enqueue = false;
volatile bool stop_all = false;
int ack_cnt = 0;

struct fifo *SPMC_fifo;
// =============================

std::string data_type = "uint8"; //default 0 ~255
float SCALE = 1; // only for developing use
char ZERO_POINT = 0; // only for develoing use

int TPU_ID  = 0;
int ITER    = 1; 
int VERBOSE = -1; //-1, or 0 - 10(max)
int ramdisk = 0;
int BREAKDOWN = 0; // for timming breakdown
int exact_mode = 0;
int start_chunk = 0;
int chunk_num = 8; // the input bit width for exact mode, default as 8 (bit), could be 16 bits
// the pwd when GPTPU was compiling
std::string PWD = CURRENT_DIR;

// for timing breakdown
long long int model_ns, list_ns, open_ns, itpr_init_ns = 0, itpr_ns = 0, input_ns, output_size, mem_ns = 0, run_ns = 0, pop_ns = 0, out_buf_ns = 0, check_ns = 0;
std::mutex mtx; // for multiple device run timing record
std::mutex mtx2;
// dev info
int dev_cnt = 1; // number of detected device(s), default as 1
bool DEV_OPENED = false;

// blocking algorithm params  
int MM_BLK_ROW = 16; // default value
int MM_BLK_COL = 16; // default value
int **partial_c; // for MM operation 

OP_node Q[queue_size];

// ===== condition for enqueue =====
// 1. Set "done_enqueue" to true at the last time of enqueuing before releasing the lock.
// Otherwise, the current runtime design will hang.
// 2. NOT QUIT SURE CONSTRAINT: task OP cnt >= dev_cnt, otherwise hang
// =================================
void wait_queue_all_finished(void){
    pthread_mutex_lock(&pmtx);
    while(ack_cnt < dev_cnt){
      pthread_cond_wait(&end_CV, &pmtx);
    }
    fifo_close(SPMC_fifo);
    pthread_mutex_unlock(&pmtx);
}

void set_start_chunk(int x){
  start_chunk = x;
}

void set_data_type(const std::string& a){
  if(a == "int8" || a == "uint8"){
    data_type = a; 
  }else{
    std::cout << __func__ <<  ": invalid data type: " << data_type << std::endl;
    exit(1);
  }
}
void set_chunk_num(int x){
  chunk_num = x;
}
void set_exact_mode(int the_mode){
 exact_mode = the_mode;
}

void set_scale(float s){
  SCALE = s;
}
void set_zp(int zp){
  ZERO_POINT = (char)zp;
}
void set_tpu_id(int tpuid){
  TPU_ID = tpuid;
}
void set_iteration(int iter){
  ITER = iter;
}
void set_verbose(int verbose){
  VERBOSE = verbose;
}
void set_breakdown(int b){
  BREAKDOWN = b;
}
void set_dev_cnt(int a){
  dev_cnt = a;
}
bool BLOCK_IS_SET = false;
void set_block(int row, int col){
  MM_BLK_ROW = row;
  MM_BLK_COL = col;
  BLOCK_IS_SET = true;
}
void set_ramdisk(int s){
  ramdisk = s;
}
// =============================
std::string data_dir;
std::string tempG_dir = "/usr/local/gptpu";
std::string local_temp_dir = PWD+"/template/";
std::string temp_dir  = tempG_dir+"/template/";
std::string template_path; //  = temp_dir + "dense_temp.tflite";
std::string mm_model       = "mm_model";
std::string mv_model       = "mv_model";
std::string imv_model      = "imv_model";
std::string bmv_model      = "bmv_model";
std::string conv_model     = "conv_model";
std::string vs_model       = "vs_model";
std::string tanh_model     = "tanh_model";
std::string relu_model     = "relu_model";
std::string add_model      = "add_model";
std::string sub_model      = "sub_model";
std::string mul_model      = "mul_model";
std::string max_model      = "max_model";
std::string maxpool_model  = "maxpool_model";
std::string mean_model     = "mean_model";
std::string log_model      = "log_model";
std::string crop_model     = "crop_model";
std::string ext_model      = "ext_model"; // reverse op of crop
std::string tf_matrix_path;  
std::string input_path     = data_dir + "input.txt";
std::string input2_path    = data_dir + "input2.txt";
std::string output_path    = data_dir + "output.txt";
std::string tf_output_path = data_dir + "tf_output.txt";

union scaling{
  float f;
  char c[sizeof(float)];
};

inline bool file_exist(std::string& file_path){
  std::ifstream file(file_path);
  return (!file)? false : true;
}

// ===== openctpu APIs start =====
openctpu_dimension::openctpu_dimension(){
  n = 2;
  dims[0] = dims[1] = dims[2] = 0;
}

openctpu_dimension::~openctpu_dimension(){ }

void openctpu_dimension::set_n_of_dims(int x){  n = x;  }

void openctpu_dimension::set_dims(int x, int y, int z){
  dims[0] = x;
  dims[1] = y;
  dims[2] = z;
}

int openctpu_dimension::get_n_of_dims(void){  return n;  }

void openctpu_dimension::get_dims(int& x, int& y, int& z){
  x = dims[0];
  y = dims[1];
  z = dims[2];
}

openctpu_buffer::openctpu_buffer(){
// default setting
  tensor_type = -1; // 0: model weight, 1: input data, 2: output tensor
  data_type = false;
  b_major   = true; // col_major
  is_out    = false;
  exact_mode = true;
  mm256_mode = true; //mm2conv otherwise
  chunk_num = 16;
  scale = 1;
  mean = 0;
  this->tile_info.A = this->tile_info.B = this->tile_info.C = 0;
  this->tile_info.blk_A = this->tile_info.blk_B = this->tile_info.blk_C = 256;
  this->tile_info.ROW_BLK_CNT = this->tile_info.INN_BLK_CNT = this->tile_info.COL_BLK_CNT = 1;
// for mm256conv
  if(chunk_num == 16){
    IN_W = 8192; IN_H = 8;   IN_C = 16; OUT_C = 4096; F_W = S_W = 8; F_H = S_H = 2;
  }else if(chunk_num == 8){
    IN_W = 512;  IN_H = 128; IN_C = 8;  OUT_C = 2048; F_W = S_W = 4; F_H = S_H = 8;
  }else if(chunk_num == 4){
    IN_W = 128;  IN_H = 256; IN_C = 8;  OUT_C = 1024; F_W = S_W = 8; F_H = S_H = 4;
  }else if(chunk_num == 2){
    IN_W = 256;  IN_H = 128; IN_C = 4;  OUT_C = 512;  F_W = S_W = 8; F_H = S_H = 8;
  }else if(chunk_num == 1){
    IN_W = 128;  IN_H = 128; IN_C = 4;  OUT_C = 256; F_W = S_W = 32; F_H = S_H = 2;
  }else{
    std::cout << __func__ << ": undefined exact mode chunk_num: " << chunk_num << std::endl; exit(0);
  }
}

openctpu_buffer::~openctpu_buffer(){
  for(int i = 0 ; i < this->tile_info.ROW_BLK_CNT*this->tile_info.INN_BLK_CNT*this->tile_info.COL_BLK_CNT; i++){
    delete [] a_blk[i];
    delete [] a_feed[i];
  }
  delete [] a_blk;
  delete [] a_feed;
  if(exact_mode == true && mm256_mode == true){
    for(int i = 0 ; i < this->tile_info.INN_BLK_CNT ; i++){
       for(int j = 0 ; j < this->tile_info.ROW_BLK_CNT*this->tile_info.COL_BLK_CNT ; j++){
         delete [] blk_exact_c[i][j];
       }
       delete [] blk_exact_c[i];
    }
    delete [] blk_exact_c;
  }else if(mm256_mode == false){
    for(int i = 0 ; i < this->tile_info.INN_BLK_CNT ; i++){
      delete [] partial_c[i];
    }
    delete [] partial_c;
  }
  if(config != NULL){ // at least one corresponding tensor
    delete [] config;
  }
}

void openctpu_buffer::set_config(openctpu_config* config){  
  if(config->get_data_type() == 1/*float*/){
    config->set_exact_mode(false);
    config->set_mm256_mode(false);
    config->set_chunk_num(8);
  }
  this->config = config;  
}

void openctpu_buffer::set_flags(bool B_major, int tensor_type){
/*
  definition of flags:
  flags[0]: bool as data_type, 0 for int array, 1 for model_path
  flags[1]: bool as b_major, 0 for row_major, 1 for col_major
  flags[2]: bool as output_data, 0 for not-output data array , 1 for output data array ,default should be 0
*/
// tensor - wise
    //data_type  = (bool)( 0x00000001 & flags);       // set model data or model weight otherwise
    //b_major    = (bool)((0x00000002 & flags) >> 1);  //  set col major or row-major otherwise
    //is_out     = (bool)((0x00000004 & flags) >> 2);  // is output tensor

    b_major   = B_major;
    data_type = tensor_type%2;
    is_out    = tensor_type/2;

}

void openctpu_buffer::set_blk_sizes(int a, int b, int c){
  this->tile_info.blk_A = a;
  this->tile_info.blk_B = b;
  this->tile_info.blk_C = c;
}

void openctpu_buffer::set_BLK_CNTs(int a, int b, int c){
  this->tile_info.A = a;
  this->tile_info.B = b;
  this->tile_info.C = c;
  this->tile_info.ROW_BLK_CNT = sub_cnt(a, this->tile_info.blk_A);
  this->tile_info.INN_BLK_CNT = sub_cnt(b, this->tile_info.blk_B);
  this->tile_info.COL_BLK_CNT = sub_cnt(c, this->tile_info.blk_C);
  this->tile_info.ROW_BLK_REM = a % this->tile_info.blk_A;
  this->tile_info.INN_BLK_REM = b % this->tile_info.blk_B;
  this->tile_info.COL_BLK_REM = c % this->tile_info.blk_C;
}

void openctpu_buffer::set_tile_info(int A, int B, int C, int blk_A, int blk_B, int blk_C){
  this->tile_info.A = A;
  this->tile_info.B = B;
  this->tile_info.C = C;
  this->tile_info.blk_A = blk_A;
  this->tile_info.blk_B = blk_B;
  this->tile_info.blk_C = blk_C;
  this->tile_info.ROW_BLK_CNT = sub_cnt(A, this->tile_info.blk_A);
  this->tile_info.INN_BLK_CNT = sub_cnt(B, this->tile_info.blk_B);
  this->tile_info.COL_BLK_CNT = sub_cnt(C, this->tile_info.blk_C);
  this->tile_info.ROW_BLK_REM = A % this->tile_info.blk_A;
  this->tile_info.INN_BLK_REM = B % this->tile_info.blk_B;
  this->tile_info.COL_BLK_REM = C % this->tile_info.blk_C;
}

void openctpu_buffer::set_conv_shape(int in_w, int in_h, int in_c, int out_c, int f_w, int f_h, int s_w, int s_h){
  IN_W  = in_w;
  IN_H  = in_h;
  IN_C  = in_c;
  OUT_C = out_c;
  F_W   = f_w;
  F_H   = f_h;
  S_W   = s_w;
  S_H   = s_h;
}

void  openctpu_buffer::set_mm256_conv_shape(int chunk_num){
// for mm256conv
  if(chunk_num == 16){
    IN_W = 8192; IN_H = 8;   IN_C = 16; OUT_C = 4096; F_W = S_W = 8; F_H = S_H = 2;
  }else if(chunk_num == 8){
    IN_W = 512;  IN_H = 128; IN_C = 8;  OUT_C = 2048; F_W = S_W = 4; F_H = S_H = 8;
  }else if(chunk_num == 4){
    IN_W = 128;  IN_H = 256; IN_C = 8;  OUT_C = 1024; F_W = S_W = 8; F_H = S_H = 4;
  }else if(chunk_num == 2){
    IN_W = 256;  IN_H = 128; IN_C = 4;  OUT_C = 512;  F_W = S_W = 8; F_H = S_H = 8;
  }else if(chunk_num == 1){
    IN_W = 128;  IN_H = 128; IN_C = 4;  OUT_C = 256; F_W = S_W = 32; F_H = S_H = 2;
  }else{
    std::cout << __func__ << ": undefined exact mode chunk_num: " << chunk_num << std::endl; exit(0);
  }
}
void openctpu_buffer::set_params(double scale, int mean){
  this->scale = scale;
  this->mean  = mean;
}
void openctpu_buffer::set_maxmin(float max, float min){
  this->data_max = max;
  this->data_min = min;
  std::cout << __func__ << ": max: " << max << ", min: " << min << std::endl;
}
void openctpu_buffer::set_int_or_float(bool flag){
  this->int_or_float = flag;
}

// get
openctpu_config* openctpu_buffer::get_config(void){  return config;  }

bool openctpu_buffer::get_data_type(void){  return data_type;  }

bool openctpu_buffer::get_b_major(void){  return b_major;  }

bool openctpu_buffer::get_is_out(void){  return is_out;  }

bool openctpu_buffer::get_exact_mode(void){  return config->get_exact_mode();  }

bool openctpu_buffer::get_mm256_mode(void){  return config->get_mm256_mode();  }

int  openctpu_buffer::get_chunk_num(void){  return config->get_chunk_num();  }

void openctpu_buffer::get_blk_sizes(int& a, int& b, int& c){
  a = this->tile_info.blk_A;
  b = this->tile_info.blk_B;
  c = this->tile_info.blk_C;
}

void openctpu_buffer::get_sizes(int& x, int& y, int& z){
  x = this->tile_info.A;
  y = this->tile_info.B;
  z = this->tile_info.C;
}

void openctpu_buffer::get_BLK_CNTs(int& x, int& y, int& z, int& a, int& b, int& c){
  x = this->tile_info.ROW_BLK_CNT;
  y = this->tile_info.INN_BLK_CNT;
  z = this->tile_info.COL_BLK_CNT;
  a = this->tile_info.ROW_BLK_REM;
  b = this->tile_info.INN_BLK_REM;
  c = this->tile_info.COL_BLK_REM;
}

struct TILE_INFO openctpu_buffer::get_tile_info(void){
  struct TILE_INFO tile_info;
  tile_info.A = this->tile_info.A;
  tile_info.B = this->tile_info.B;
  tile_info.C = this->tile_info.C;
  tile_info.blk_A = this->tile_info.blk_A;
  tile_info.blk_B = this->tile_info.blk_B;
  tile_info.blk_C = this->tile_info.blk_C;
  tile_info.ROW_BLK_CNT = this->tile_info.ROW_BLK_CNT;
  tile_info.INN_BLK_CNT = this->tile_info.INN_BLK_CNT;
  tile_info.COL_BLK_CNT = this->tile_info.COL_BLK_CNT;
  tile_info.ROW_BLK_REM = this->tile_info.ROW_BLK_REM;
  tile_info.INN_BLK_REM = this->tile_info.INN_BLK_REM;
  tile_info.COL_BLK_REM = this->tile_info.COL_BLK_REM;
  return tile_info;
}

void openctpu_buffer::get_conv_shape(int& in_w, int& in_h, int& in_c, int& out_c, int& f_w, int& f_h, int& s_w, int& s_h){
  in_w  = IN_W;
  in_h  = IN_H;
  in_c  = IN_C;
  out_c = OUT_C;
  f_w   = F_W;
  f_h   = F_H;
  s_w   = S_W;
  s_h   = S_H;
}
void openctpu_buffer::get_params(double& scale, int& mean){
  scale = this->scale;
  mean  = this->mean;
}
void openctpu_buffer::get_maxmin(float& max, float& min){
  max = this->data_max;
  min = this->data_min;
  std::cout << __func__ << ": max: " << max << ", min: " << min << std::endl;
}
void openctpu_buffer::get_int_or_float(bool& flag){
  flag = this->int_or_float;
}

void ChooseQuantizationParams(float max, float min, double& scale, int& mean/*nudged_zero_point*/){
  const float qmin = 0;
  const float qmax = 255;
  scale = (max - min)/(qmax - qmin);
  const double initial_zero_point = qmin - min / scale;
  std::uint8_t nudged_zero_point = 0;
  if(initial_zero_point < qmin){
    nudged_zero_point = qmin;
  }else if(initial_zero_point > qmax){
    nudged_zero_point = qmax;
  }else{
    nudged_zero_point = static_cast<std::uint8_t>(std::round(initial_zero_point));
  }
  mean = (int)nudged_zero_point;
}

void openctpu_buffer::quantize(float* in, int* out, int length){
  float max = FLT_MIN;
  float min = FLT_MAX;
// find max/min
  for(int i = 0 ; i < length ; i++){
    if(in[i] > max){ max = in[i]; }
    if(in[i] < min){ min = in[i]; }
  }
  double _scale;
  int _mean;
  this->get_params(_scale, _mean);
  ChooseQuantizationParams(max, min, _scale, _mean);
  this->set_params(_scale, _mean);
  this->set_maxmin(max, min);
  std::cout << __func__ << ": scale: " << _scale << ", mean: " << _mean << ", max: " << max << ", min: " << min << std::endl;
  for(int i = 0 ; i < length ; i++){
    const float transformed_val = _mean + in[i] / _scale;
    const float clamped_val = std::max(0.f, std::min(255.f, transformed_val));
    out[i] = static_cast<int>(std::round(clamped_val));
  }
}
void openctpu_buffer::dequantize(int* in, float* out, int depth, int length){
  double _scale;
  int _mean;
  this->get_params(_scale, _mean);
  std::cout << __func__ << ": _scale: " << _scale << ", _mean: " << _mean << ", depth: " << depth << ", length: " << length << std::endl;
  for(int i = 0 ; i < length ; i++){
    out[i] = (in[i] - depth * _mean) * _scale;
  }
}
void openctpu_buffer::allocate_a(bool mm256_mode){
  int total_cnt = this->tile_info.ROW_BLK_CNT*this->tile_info.INN_BLK_CNT;//*this->tile_info.COL_BLK_CNT;
  a_blk  = new int*[total_cnt];
  a_feed = new int*[total_cnt];
  int s = (mm256_mode == true)?this->chunk_num:1;
  for(int i = 0 ; i < total_cnt ; i++){
    a_blk[i]  = new int[this->tile_info.blk_A*this->tile_info.blk_B](); // make sure zero
    a_feed[i] = new int[this->tile_info.blk_A*this->tile_info.blk_B*s];
  }
}

void openctpu_buffer::fill_a(int* data, bool mm256_mode){
  int A, B, C, blk_A, blk_B, blk_C, ROW_BLK_CNT, INN_BLK_CNT, COL_BLK_CNT, ROW_BLK_REM, INN_BLK_REM, COL_BLK_REM, chunk_num;
  chunk_num = this->chunk_num;
  auto tile_info = this->get_tile_info();
  A = tile_info.A;
  B = tile_info.B;
//  C = tile_info.C;
  blk_A = tile_info.blk_A;
  blk_B = tile_info.blk_B;
//  blk_C = tile_info.blk_C;
  ROW_BLK_CNT = tile_info.ROW_BLK_CNT;
  INN_BLK_CNT = tile_info.INN_BLK_CNT;
  COL_BLK_CNT = tile_info.COL_BLK_CNT;
  ROW_BLK_REM = tile_info.ROW_BLK_REM;
  INN_BLK_REM = tile_info.INN_BLK_REM;
  COL_BLK_REM = tile_info.COL_BLK_REM;
//std::cout << ": A: " << A << ", B: " << B << ", C: " << C << ", blk_A: " << blk_A << ", blk_B: " << blk_B << ", blk_C: " << blk_C << std::endl;
//std::cout << " r cnt: " << ROW_BLK_CNT << "in cnt: " << INN_BLK_CNT << ", c cnt: " << COL_BLK_CNT << ", r rem: " << ROW_BLK_REM << ", in rem: " << INN_BLK_REM << ", c rem: " << COL_BLK_REM << std::endl;
  for(int j = 0 ; j < INN_BLK_CNT ; j++){
//    for(int k = 0 ; k < COL_BLK_CNT ; k++){
      for(int i = 0 ; i < ROW_BLK_CNT ; i++){
        //int idx = i*INN_BLK_CNT*COL_BLK_CNT + j*COL_BLK_CNT + k;
        int idx = i*INN_BLK_CNT + j;
        for(int curr_i = 0 ; curr_i < ((i<ROW_BLK_CNT-1 || ROW_BLK_REM == 0)?blk_A:ROW_BLK_REM) ; curr_i++){
//std::cout << "j: " << j << ", k: " << k << ", i: " << i << ", curr_i: " << curr_i << std::endl;
          memcpy(a_blk[idx]+curr_i*blk_B, data+(i*blk_A+curr_i)*B+(j*blk_B), ((j<INN_BLK_CNT-1 || INN_BLK_REM == 0)?(blk_B):(INN_BLK_REM))*sizeof(int)); 
        }
        if(mm256_mode == true){
          data_mm256conv(a_blk[idx], a_feed[idx], blk_A, blk_B, IN_W, IN_H, F_W, F_H, IN_C, chunk_num); 
        }else{
//std::cout << "blk_A: " << blk_A << ", blk_C: " << blk_C << ", IN_W: " << IN_W << ", IN_H: " << IN_H << ", F_W: " << F_W << ", F_H: " << F_H << ", IN_C " << IN_C << std::endl;
          data_mm2conv(a_blk[idx], a_feed[idx], blk_A, blk_B, IN_W, IN_H, F_W, F_H, IN_C); 
        }
      }
//    }
  } 
}

void openctpu_buffer::allocate_c(void* data, int type/*0: int, 1: float*/, bool mm256_mode){
  if(exact_mode == 1 && mm256_mode == true){ // exact_mode == 1
    int chunk_num = this->chunk_num;
// TODO: remove those temp buffer
    blk_exact_c = new int**[this->tile_info.INN_BLK_CNT];
    for(int i = 0 ; i < this->tile_info.INN_BLK_CNT ; i++){
      blk_exact_c[i] = new int*[this->tile_info.A*this->tile_info.C*chunk_num*chunk_num];
      for(int j = 0 ; j < this->tile_info.ROW_BLK_CNT*this->tile_info.COL_BLK_CNT ; j++){
        blk_exact_c[i][j] = new int[this->tile_info.blk_A*this->tile_info.blk_C*chunk_num*chunk_num];
      }
    }
  }else if(mm256_mode == false){
    partial_c = new int*[this->tile_info.INN_BLK_CNT];
    for(int i = 0 ; i < this->tile_info.INN_BLK_CNT; i++){
      partial_c[i] = new int[this->tile_info.A*this->tile_info.C];
    }
  }else{
    std::cout << __func__ << ": exact_mode: " << exact_mode << std::endl;
    exit(0);
  }
  if(type == 0){
    c = (int*)data; // keep the pointer from caller
  }else{
    c = new int[this->tile_info.A*this->tile_info.C];
    this->float_c = (float*)data;
  }
}

void openctpu_init(int opening_order, int wanted_dev_cnt){
  open_devices(opening_order, wanted_dev_cnt);
}

openctpu_config::openctpu_config(){
  this->chunk_num = 8;
  this->exact_mode = false;
  this->mm256_mode = false;
  this->blk_A = this->blk_B = this->blk_C = 256; 
}

openctpu_config::~openctpu_config(){  }

void openctpu_config::set_data_type(int data_type){     this->data_type  = data_type;   }
void openctpu_config::set_chunk_num(int chunk_num){     this->chunk_num  = chunk_num;   }
void openctpu_config::set_exact_mode(bool exact_mode){  this->exact_mode = exact_mode;  }
void openctpu_config::set_mm256_mode(bool mm256_mode){  this->mm256_mode = mm256_mode;  }
void openctpu_config::set_blks(int blk_A, int blk_B, int blk_C){
  this->blk_A = blk_A;
  this->blk_B = blk_B;
  this->blk_C = blk_C;
}

int  openctpu_config::get_data_type(void){   return data_type;   }
int  openctpu_config::get_chunk_num(void){   return chunk_num;   }
bool openctpu_config::get_exact_mode(void){  return exact_mode;  }
bool openctpu_config::get_mm256_mode(void){  return mm256_mode;  }
void openctpu_config::get_blks(int& blk_A, int& blk_B, int& blk_C){
  blk_A = this->blk_A;
  blk_B = this->blk_B;
  blk_C = this->blk_C;
}

openctpu_config* openctpu_setConfig(int data_type, bool exact_mode, bool mm256_mode, int chunk_num){
  openctpu_config* config = new openctpu_config();
  config->set_data_type(data_type);
  config->set_chunk_num(chunk_num);
  config->set_exact_mode(exact_mode);
  config->set_mm256_mode(mm256_mode);
  return config;
}

openctpu_dimension* openctpu_alloc_dimension(int dim, ...){
  openctpu_dimension* ret = new openctpu_dimension();
  va_list lst;
  va_start(lst, dim);
  ret->set_n_of_dims(dim);
  int tmp[3];
  for(int i = 0 ; i < dim ; i++){
    if(i > 2){
      std::cout << __func__ << ": allocate " << dim << " dimensions exceed 3" << std::endl;
      exit(0);
    }
    tmp[i] = va_arg(lst, int);
  }
  va_end(lst);
  ret->set_dims(tmp[0], tmp[1], tmp[2]);
  return ret;
}

#define tf_creates_tflite_and_template() \
  std::cout << "template_path: " << template_path << " not exist, creating it..., mm256_mode: " << mm256_mode << std::endl; \
  std::string weight_file_name = (mm256_mode == true)?("./../mm256conv_weight_"+itoa(i)+"x"+itoa(j)+"_outof_"+itoa(INN_BLK_CNT)+"x"+itoa(COL_BLK_CNT)+".txt"):("./../mm2conv_weight_"+itoa(i)+"x"+itoa(j)+"_outof_"+itoa(INN_BLK_CNT)+"x"+itoa(COL_BLK_CNT)+"_chunk"+itoa(w_chunk_idx)+"of"+itoa(chunk_num)+".txt"); \
  if(mm256_mode == true){ \
    mm256blk_save_weight((int*)data, ret->get_b_major(), weight_file_name, B, C, i, j, INN_BLK_CNT, COL_BLK_CNT, blk_B, blk_C, INN_BLK_REM, COL_BLK_REM, chunk_num); \
  }else{ \
    mm2conv_save_weight((int*)data, ret->get_b_major(), weight_file_name, B, C, i, j, INN_BLK_CNT, COL_BLK_CNT, blk_B, blk_C, INN_BLK_REM, COL_BLK_REM, w_chunk_idx); \
  } \
  std::string command = "python3 "+PWD+"/src/create_model.py --model=conv_model"+" --in_w_name="+weight_file_name+" --data_type="+data_type+" --out_scale="+itoa(SCALE)+" --outfile_name="+matrix_path+" --IN_W="+itoa(IN_W)+" --IN_H="+itoa(IN_H)+" --IN_C="+itoa(IN_C)+" --F_W="+itoa(F_W)+" --F_H="+itoa(F_H)+" --S_W="+itoa(F_W)+" --S_H="+itoa(F_H)+" --OUT_C="+itoa(OUT_C)+" --mm256blk="+itoa(mm256_mode); \
  system(command.c_str()); \
  std::cout << __func__ << ": matrix_path: " << matrix_path << ", local_temp_dir: " << local_temp_dir << ", template_name: " << template_name << ", blk_A: " << blk_A << ", blk_B: " << blk_B << ", blk_C: " << blk_C << ", chunk_num: " << chunk_num << std::endl; \
  make_mm2conv_temp(matrix_path, local_temp_dir+template_name, blk_A*chunk_num, blk_B, blk_C*chunk_num); \
  command = "sudo cp "+local_temp_dir+template_name+" "+template_path; \
  system(command.c_str()); \

openctpu_buffer* openctpu_create_buffer(openctpu_dimension* dim, void* data, openctpu_config* config, bool b_major, int tensor_type){
// there are three types of buffer: model, input data, output buffer
  openctpu_buffer* ret = new openctpu_buffer();
  ret->set_config(config);
  ret->set_flags(b_major, tensor_type); // set tensor-wise flag only
  int x,y,z;
  dim->get_dims(x, y, z);
  ret->set_dims(x, y, z);
// ===== core impl., currently do GEMM with mm256conv design only  ===
  int A, B, C, dummy;
  int ROW_BLK_CNT, INN_BLK_CNT, COL_BLK_CNT, ROW_BLK_REM, INN_BLK_REM, COL_BLK_REM;
  int blk_A, blk_B, blk_C;
  
  //struct TILE_INFO tile_info;
  int IN_W, IN_H, IN_C, OUT_C, F_W, F_H, S_W, S_H;
  int type        = ret->config->get_data_type();
  if(type == 1){
    ret->config->set_blks(x,y,z); // for float, no need to do sub-blocking
  }
  ret->config->get_blks(blk_A, blk_B, blk_C);
  bool exact_mode = (type == 0)?(ret->config->get_exact_mode()):(false/*float must be approx. mode*/);
  int chunk_num   = ret->config->get_chunk_num();
  bool mm256_mode = (type == 0)?(ret->config->get_mm256_mode()):(false);
  if(ret->get_data_type() == 1 && ret->get_is_out() == 0){ // model type, create model_path
    set_exact_mode(exact_mode);
    ret->get_dims(B, C, dummy);
    A = B;
// TODO: data_array is temp, could be removed as optimization
    SET_BLK(ret, A, B, C, ROW_BLK_CNT, INN_BLK_CNT, COL_BLK_CNT, ROW_BLK_REM, INN_BLK_REM, COL_BLK_REM);
    //ret->set_tile_info(A, B, C, blk_A, blk_B, blk_C);
    //tile_info = ret->get_tile_info();
// auto mm2conv mapping
    int data_array_size;
    set_chunk_size((exact_mode == true)?1:8);
    //set_scale(255);
    if(mm256_mode == true){
      data_array_size = blk_B*blk_C*chunk_num;
      ret->set_mm256_conv_shape(chunk_num);
      set_scale(255);
    }else{ // mm2conv mode
      SCALE = (exact_mode == true)?1:1; //TODO need an updated function  //get_auto_scale_factor_mm(a, b, A, B, C);
      chunk_num = CHAR_BIT/get_chunk_size();
      mm2conv_shape_mapping(A, B, C, blk_A, blk_B, blk_C, exact_mode, IN_W, IN_H, IN_C, F_W, F_H, S_W, S_H, OUT_C);
      ret->set_conv_shape(IN_W, IN_H, IN_C, OUT_C, F_W, F_H, S_W, S_H);
      ret->set_tile_info(A, B, C, blk_A, blk_B, blk_C);
      data_array_size = blk_B*blk_C;
      auto tile_info = ret->get_tile_info();
      ROW_BLK_CNT = tile_info.ROW_BLK_CNT;
      INN_BLK_CNT = tile_info.INN_BLK_CNT;
      COL_BLK_CNT = tile_info.COL_BLK_CNT;
      ROW_BLK_REM = tile_info.ROW_BLK_REM;
      INN_BLK_REM = tile_info.INN_BLK_REM;
      COL_BLK_REM = tile_info.COL_BLK_REM;
    }
//    std::cout << "A: " << A << ", B: " << B << ", C: " << C << ", blk_A: " << blk_A << ", blk_B: " << blk_B << ", blk_C: " << blk_C << ", data_array_size: " << data_array_size << std::endl;
    ret->get_conv_shape(IN_W, IN_H, IN_C, OUT_C, F_W, F_H, S_W, S_H);
std::cout << __func__ << ": IN_W: " << IN_W << ",IN_H: " << IN_H << ",IN_C: " << IN_C << ",OUT_C: " << OUT_C << ",F_W: " << F_W << ",F_H: " << F_H << std::endl;
    char* data_array = (char*) malloc(data_array_size*sizeof(char));
    bool template_created = false;
    std::string template_name = "conv_temp_"+itoa(IN_W)+"x"+itoa(IN_H)+"x"+itoa(IN_C)+"x"+itoa(F_W)+"x"+itoa(F_H)+"x"+itoa(S_W)+"x"+itoa(S_H)+"_"+data_type+".tflite";
    std::string template_path = temp_dir+template_name;
    int model_id;
// for 1: float usage
    int length;
    int* data_int;
    if(type == 1){ // 1:float
      length = B*C;
      data_int = (int*) malloc(length*sizeof(int));
      ret->quantize((float*)data, data_int, length);
    }
    for(int i = 0 ; i < INN_BLK_CNT ; i++){
      for(int j = 0 ; j < COL_BLK_CNT ; j++){
        // fuse mm256_mode and mm2conv mode 
        for(int w_chunk_idx = (mm256_mode == true)?(0):(start_chunk) ; w_chunk_idx < ((mm256_mode == true)?1:(chunk_num)) ; w_chunk_idx++){
          model_id = (mm256_mode == true)? (i*COL_BLK_CNT+j) : (i*COL_BLK_CNT*chunk_num+j*chunk_num+w_chunk_idx);
          std::string matrix_path = (mm256_mode == true)?
                                    (data_dir+"conv_model_tflite/conv_model_quant_"+itoa(IN_W)+"x"+itoa(IN_H)+"x"+itoa(IN_C)+"x"+itoa(F_W)+"x"+itoa(F_H)+"x"+itoa(S_W)+"x"+itoa(S_H)+"_256exact_"+itoa(i)+"x"+itoa(j)+"_outof_"+itoa(INN_BLK_CNT)+"x"+itoa(COL_BLK_CNT)+"_edgetpu.tflite")
                                   :(data_dir+"conv_model_tflite/conv_model_quant_"+itoa(IN_W)+"x"+itoa(IN_H)+"x"+itoa(IN_C)+"x"+itoa(F_W)+"x"+itoa(F_H)+"x"+itoa(S_W)+"x"+itoa(S_H)+"_2048based_"+itoa(i)+"x"+itoa(j)+"_outof_"+itoa(INN_BLK_CNT)+"x"+itoa(COL_BLK_CNT)+"_chunk"+itoa(w_chunk_idx)+"_of"+itoa(chunk_num)+"_edgetpu.tflite");
std::cout << __func__ << ": matrix_path " << matrix_path << ", data_dir: " << data_dir << std::endl;
          if(file_exist(template_path) == false && template_created == false){
            template_created = true; // once for all same shape blocks
            tf_creates_tflite_and_template(); // shorten 
            // build as model and ready to go
            build_model(matrix_path, model_id);
          }else{  
            if(type == 0){ // 0: int
              if(mm256_mode == true){
                set_mm256conv_array(((int*)data), b_major, data_array, B, C, i, j, INN_BLK_CNT, COL_BLK_CNT, blk_B, blk_C, INN_BLK_REM, COL_BLK_REM, chunk_num, 1/*exact_mode*/); // relayout data array
                create_mm2conv_tflite(template_path, flatbufs, /*matrix_path*/model_id, data_array, blk_A*chunk_num, blk_B, blk_C*chunk_num, SCALE, chunk_num); // create .tflite binary
              }else{
                set_mm2conv_array(((int*)data), ret->get_b_major(), data_array, B, C, i, j, INN_BLK_CNT, COL_BLK_CNT, blk_B, blk_C, INN_BLK_REM, COL_BLK_REM, w_chunk_idx, exact_mode); // relayout data array
                create_mm2conv_tflite(template_path, flatbufs, /*matrix_path*/model_id, data_array, blk_A, blk_B, blk_C, SCALE, 1/*dummy*/); // create .tflite binary
              }
            }else{ // 1:float   
              set_mm2conv_array(data_int, ret->get_b_major(), data_array, B, C, i, j, INN_BLK_CNT, COL_BLK_CNT, blk_B, blk_C, INN_BLK_REM, COL_BLK_REM, w_chunk_idx, false/*exact_mode*/); // relayout data array
              //create_mm2conv_tflite(template_path, flatbufs, /*matrix_path*/model_id, data_array, blk_A, blk_B, blk_C, SCALE, 1/*dummy*/); // create .tflite binary
              create_mm2conv_tflite(template_path, matrix_path, data_array, blk_A, blk_B, blk_C, SCALE, 1/*dummy*/); // create .tflite binary
            }
            // build as model and ready to go
            //build_model_from_buffer(flatbufs[model_id].buf, flatbufs[model_id].size, /*matrix_path,*/ model_id);
            build_model(matrix_path, model_id);
          }
        }
      }
    }
    free(data_array);
  }else if(ret->get_data_type() == 0 && ret->get_is_out() == 0){ // input data type, create data array
//std::cout << "is_input data, mm256_mode: " << mm256_mode << std::endl;
    ret->get_dims(A, B, dummy);
    C = B;
    SET_BLK(ret, A, B, C, ROW_BLK_CNT, INN_BLK_CNT, COL_BLK_CNT, ROW_BLK_REM, INN_BLK_REM, COL_BLK_REM);
    if(mm256_mode == true){
      ret->set_mm256_conv_shape(chunk_num);
    }else{ // mm2conv mode
      mm2conv_shape_mapping(A, B, C, blk_A, blk_B, blk_C, exact_mode, IN_W, IN_H, IN_C, F_W, F_H, S_W, S_H, OUT_C);
      ret->set_conv_shape(IN_W, IN_H, IN_C, OUT_C, F_W, F_H, S_W, S_H);
      ret->set_tile_info(A, B, C, blk_A, blk_B, blk_C);
    }
//TODO: need fusing optimization
    ret->allocate_a(mm256_mode); // allocate a_blk and a_feed internally after SET_BLK
    if(type == 1){ // 1:float
      int length = A*B;
      int* data_int = (int*) malloc(length*sizeof(int));
      ret->quantize((float*)data, data_int, length);
//      for(int i = 0 ; i < length ; i++){
//        std::cout << "i: " << i << ", data_int: " << data_int[i] << std::endl;
//      }
      ret->fill_a((int*)data_int, mm256_mode); // input data transformation included
    }else{
      ret->fill_a((int*)data, mm256_mode); // input data transformation included
    }
  }else{ // ret.is_out == 1
    ret->get_dims(A, C, dummy);
    B = C;
    SET_BLK(ret, A, B, C, ROW_BLK_CNT, INN_BLK_CNT, COL_BLK_CNT, ROW_BLK_REM, INN_BLK_REM, COL_BLK_REM);
    if(mm256_mode == true && type == 0){
      ret->set_mm256_conv_shape(chunk_num);
    }else{
      mm2conv_shape_mapping(A, B, C, blk_A, blk_B, blk_C, exact_mode, IN_W, IN_H, IN_C, F_W, F_H, S_W, S_H, OUT_C);
      ret->set_conv_shape(IN_W, IN_H, IN_C, OUT_C, F_W, F_H, S_W, S_H);
      ret->set_tile_info(A, B, C, blk_A, blk_B, blk_C);
    }
    ret->set_int_or_float(type);
    ret->allocate_c(data, type, (type == 1/*float*/)?0:mm256_mode);  // keep the pointer to data for populating result later
  }
  return ret;
}

struct OP_node* curr_node;

void openctpu_invoke_operator(const std::string op, openctpu_buffer* tensor_a, openctpu_buffer* tensor_b, openctpu_buffer* tensor_c){
  std::cout << __func__ << ": op invoking here" << std::endl;
  int cnt = 0;
  auto tile_info = tensor_c->get_tile_info();
  bool mm256_mode = tensor_c->get_mm256_mode();
  int chunk_num = CHAR_BIT/get_chunk_size();

// ==============================
  float a_max, a_min, b_max, b_min;
  tensor_a->get_maxmin(a_max, a_min);
  tensor_b->get_maxmin(b_max, b_min);
  float IN_SCALE = 1.0;
  if(op == mm_model && tensor_c->config->get_data_type() == 1){
    IN_SCALE = float(UCHAR_MAX)/float(tile_info.B * ((float)(a_max + a_min)/2) * ((float)(b_max + b_min)/2));
    //std::cout << __func__ << ", IN_SCALE: " << IN_SCALE << "B: " << tile_info.B << ", a_max: " << a_max << ", a_min: " << a_min << ", b_max: " << b_max << ", b_min: " << b_min << std::endl;
   }//else{
//    IN_SCALE = 1.0;
//  }
// ==============================
  curr_node  = new OP_node[tile_info.INN_BLK_CNT*tile_info.COL_BLK_CNT*chunk_num];
  for(int j = 0 ; j < tile_info.INN_BLK_CNT ; j++){
    for(int k = 0 ; k < tile_info.COL_BLK_CNT ; k++){
      for(int w_chunk_idx = ((mm256_mode == true)?0:start_chunk) ; w_chunk_idx < ((mm256_mode == true)?1:chunk_num) ; w_chunk_idx++){
        curr_node[cnt].op             = mm_model;
        curr_node[cnt].model_id       = (mm256_mode == true)?(j*tile_info.COL_BLK_CNT+k):((j*tile_info.COL_BLK_CNT+k)*chunk_num+w_chunk_idx);
        curr_node[cnt].a_feed         = tensor_a->a_feed;
std::cout << "j: " << j << ", k: " << k << ", mm256_mode: " << mm256_mode << std::endl;
        curr_node[cnt].partial_c      = (mm256_mode == true)?tensor_c->blk_exact_c[j]:tensor_c->partial_c;      
        curr_node[cnt].A              = tile_info.A;      
        curr_node[cnt].B              = tile_info.B;      
        curr_node[cnt].C              = tile_info.C;      
        curr_node[cnt].j              = j;      
        curr_node[cnt].k              = k;     
        curr_node[cnt].w_chunk_idx    = w_chunk_idx; // mm2conv only
        curr_node[cnt].start_chunk    = start_chunk; // mm2conv only
        curr_node[cnt].blk_A          = tile_info.blk_A;     
        curr_node[cnt].blk_B          = tile_info.blk_B;     
        curr_node[cnt].blk_C          = tile_info.blk_C;      
        curr_node[cnt].ROW_BLK_CNT    = tile_info.ROW_BLK_CNT;     
        curr_node[cnt].INNER_BLK_CNT  = tile_info.INN_BLK_CNT;     
        curr_node[cnt].COL_BLK_CNT    = tile_info.COL_BLK_CNT;     
        curr_node[cnt].SCALE          = IN_SCALE; // 1.0 
        curr_node[cnt].mm256          = mm256_mode;     
        curr_node[cnt].chunk_num      = tensor_c->get_chunk_num(); // mm256 only     
        fifo_push(SPMC_fifo, &curr_node[cnt]);
        cnt++;
        if(j == (tile_info.INN_BLK_CNT-1) && k == (tile_info.COL_BLK_CNT-1) && ((mm256_mode == true)?1:(w_chunk_idx == (chunk_num-1)))){pthread_mutex_lock(&pmtx); done_enqueue = true; pthread_mutex_unlock(&pmtx); }
      }
    }
  }
  std::cout << "done invoking" << std::endl;
}

void openctpu_enqueue(void(*func)(openctpu_buffer* a,
                                  openctpu_buffer* b,
                                  openctpu_buffer* c), 
                      openctpu_buffer* tensor_a, 
                      openctpu_buffer* tensor_b, 
                      openctpu_buffer* tensor_c){
  // call the function
  (*func)(tensor_a, tensor_b, tensor_c);
}

void openctpu_sync(openctpu_buffer* tensor_c){
// =============================================
// TODO:
// dequantize for int2float requires result_qparams from reference calculation 
// =============================================
  wait_queue_all_finished();
  delete [] curr_node;
  edgetpu_cleanup(); // clean up vectors: Itpr and model
  std::cout << __func__ << " done. partial sum summation stage..." << std::endl;
// TODO: after sync, sum up partial sum to make sure at this moment tensor_c has results in buffer.
// TODO: pipeline the invoking and summation stages
  int ROW_BLK_CNT, INN_BLK_CNT, COL_BLK_CNT, ROW_BLK_REM, INN_BLK_REM, COL_BLK_REM;
  int A, B, C;
  int blk_A, blk_B, blk_C;
  int chunk_num = tensor_c->get_chunk_num();
  bool type;
  tensor_c->get_int_or_float(type);
  if(tensor_c->get_mm256_mode() == true){
    tensor_c->get_sizes(A, B, C);
    tensor_c->get_blk_sizes(blk_A, blk_B, blk_C);
    tensor_c->get_BLK_CNTs(ROW_BLK_CNT, INN_BLK_CNT, COL_BLK_CNT, ROW_BLK_REM, INN_BLK_REM, COL_BLK_REM);
    int** partical_c;
    partial_c = (int**)malloc(INN_BLK_CNT*sizeof(int*));
    for(int i = 0 ; i < INN_BLK_CNT; i++){
      partial_c[i] = (int*)calloc(A*C*chunk_num*chunk_num, sizeof(int));
    }
    for(int k = 0 ; k < blk_A*blk_C*chunk_num*chunk_num ; k++){
      int chunk_r = k/(blk_A*blk_C*chunk_num);
      int chunk_c = (k%(blk_C*chunk_num))/blk_C;
      int inblk_r = (k/(blk_C*chunk_num))%blk_A;
      int inblk_c = (k%(blk_C*chunk_num))%blk_C;
      for(int i = 0 ; i < ROW_BLK_CNT ; i++){
        for(int j = 0 ; j < COL_BLK_CNT ; j++){
          int offset  = chunk_r*(A*C*chunk_num) + chunk_c * C + i/*blk_idx_i*/*(blk_A*C*chunk_num) + j/*blk_idx_k*/*blk_C + inblk_r*(C*chunk_num) + inblk_c;
          for(int idx = 0 ; idx < INN_BLK_CNT ; idx++){
            partial_c[idx][offset] = tensor_c->blk_exact_c[idx][i*COL_BLK_CNT+j][k] << (chunk_r + chunk_c);
          }
        }
      }
    }
    int sum, offset;
    for(int i = 0 ; i < A ; i++){
      for(int j = 0 ; j < C ; j++){
        sum = 0;
        for(int idx_r = 0 ; idx_r < chunk_num ; idx_r++){
          for(int idx_c = 0 ; idx_c < chunk_num ; idx_c++){
            for(int in_idx = 0 ; in_idx < INN_BLK_CNT ; in_idx++){
              offset = idx_r*(A*C*chunk_num) + idx_c*(C) + i*(C*chunk_num)+j;
              sum += partial_c[in_idx][offset];
            }
          }
        }
        tensor_c->c[i*C+j] = sum;
      } 
    }
    if(type == 1/*float*/)tensor_c->dequantize(tensor_c->c, tensor_c->float_c, chunk_num*chunk_num*INN_BLK_CNT, A*C);
    for(int i = 0 ; i < INN_BLK_CNT ; i++){
      free(partial_c[i]);
    }
    free(partial_c);
  }else{ // mm2conv mode
    auto tile_info = tensor_c->get_tile_info();
    int sum = 0;
    for(int i = 0 ; i < tile_info.A*tile_info.C ; i++){
      sum = 0;
      for(int j = 0 ; j < tile_info.INN_BLK_CNT ; j++){ 
        sum += tensor_c->partial_c[j][i];
        if(i == 0){
           std::cout << "partial_c[" << j << "][" << i << "]: " << tensor_c->partial_c[j][i] << std::endl;
        }
      }
      tensor_c->c[i] = sum;
    }
    if(type == 1/*float*/)tensor_c->dequantize(tensor_c->c, tensor_c->float_c, tile_info.INN_BLK_CNT, tile_info.A*tile_info.C);
  }
//  if(1/*TODO: mm_model*/){
//    tensor_c->get_sizes(A, B, C);
//    tensor_c->dequantize(tensor_c->c, tensor_c->float_c, A*C);
//  }
//  void openctpu_buffer::dequantize(int* in, float* out, int length){
}

//TODO: verify if these two tensors are identical (element-wise), otherwise give RMSE and error rate report
void openctpu_verify(openctpu_buffer* buf, float* ref, int dim, ... ){
  va_list lst;
  va_start(lst, dim);
  int tmp[3];
  for(int i = 0 ; i < dim ; i++){
    if(i > 1){
      std::cout << __func__ << ": target dims has more than 2 " << std::endl;
      exit(0);
    }
    tmp[i] = va_arg(lst, int);
  }
  va_end(lst);
// TODO: check dim matches or not
  int error_cnt = 0;
  double square_sum_avg = 0;
  double rate = 0;
  double avg = 0;
  int t = 5; // threshold
  std::cout << "buf->float_c:" << std::endl;
  for(int i = 0 ; i < t ; i++){
    for(int j = 0 ; j < t ; j++){
      std::cout << buf->float_c[i*tmp[1]+j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "ref:" << std::endl;
  for(int i = 0 ; i < t ; i++){
    for(int j = 0 ; j < t ; j++){
      std::cout << ref[i*tmp[1]+j] << " ";
    }
    std::cout << std::endl;
  }

  for(int i = 0 ; i < tmp[0] ; i++){
    for(int j = 0 ; j < tmp[1] ; j++){
      if(buf->float_c[i*tmp[1]+j] != ref[i*tmp[1]+j]){
        if(error_cnt < t)std::cout << "[" << i << ", " << j << "]: " << buf->c[i*tmp[1]+j] << " != " << ref[i*tmp[1]+j] << std::endl;
        error_cnt++;
      }
      square_sum_avg = (square_sum_avg * i + pow(buf->float_c[i*tmp[1]+j] - ref[i*tmp[1]+j], 2)) / (i+1);
      rate = (rate * i + abs(buf->float_c[i*tmp[1]+j] - ref[i*tmp[1]+j])) / (i+1);
      avg = (avg * i + ref[i*tmp[1]+j]) / (i+1);
    }
  }
  double RMSE = sqrt(square_sum_avg);
  std::cout << __func__ << ": -- summary --" << std::endl;
  std::cout << ((error_cnt == 0)?"verify pass":"verify fail") << std::endl;
  if(error_cnt > 0){ std::cout << "error cnt: (" << error_cnt << "/" << tmp[0]*tmp[1] << ")" << std::endl; }
  printf("RMSE      : %f, RMSE%%: %f %%\n", RMSE, (RMSE/avg)*100);
  printf("error rate: %f, rate%%: %f %%\n", rate, (rate/avg)*100);
  printf("average   : %f, sizes : %dx%d\n",  avg, tmp[0], tmp[1]);
}
void openctpu_verify(openctpu_buffer* buf, int* ref, int dim, ... ){
  va_list lst;
  va_start(lst, dim);
  int tmp[3];
  for(int i = 0 ; i < dim ; i++){
    if(i > 1){
      std::cout << __func__ << ": target dims has more than 2 " << std::endl;
      exit(0);
    }
    tmp[i] = va_arg(lst, int);
  }
  va_end(lst);
// TODO: check dim matches or not
  int error_cnt = 0;
  double square_sum_avg = 0;
  double rate = 0;
  double avg = 0;
  int t = 5; // threshold
  std::cout << "buf->c:" << std::endl;
  for(int i = 0 ; i < t ; i++){
    for(int j = 0 ; j < t ; j++){
      std::cout << buf->c[i*tmp[1]+j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "ref:" << std::endl;
  for(int i = 0 ; i < t ; i++){
    for(int j = 0 ; j < t ; j++){
      std::cout << ref[i*tmp[1]+j] << " ";
    }
    std::cout << std::endl;
  }

  for(int i = 0 ; i < tmp[0] ; i++){
    for(int j = 0 ; j < tmp[1] ; j++){
      if(buf->c[i*tmp[1]+j] != ref[i*tmp[1]+j]){
        if(error_cnt < t)std::cout << "[" << i << ", " << j << "]: " << buf->c[i*tmp[1]+j] << " != " << ref[i*tmp[1]+j] << std::endl;
        error_cnt++;
      }
      square_sum_avg = (square_sum_avg * i + pow(buf->c[i*tmp[1]+j] - ref[i*tmp[1]+j], 2)) / (i+1);
      rate = (rate * i + abs(buf->c[i*tmp[1]+j] - ref[i*tmp[1]+j])) / (i+1);
      avg = (avg * i + ref[i*tmp[1]+j]) / (i+1);
    }
  }
  double RMSE = sqrt(square_sum_avg);
  std::cout << __func__ << ": -- summary --" << std::endl;
  std::cout << ((error_cnt == 0)?"verify pass":"verify fail") << std::endl;
  if(error_cnt > 0){ std::cout << "error cnt: (" << error_cnt << "/" << tmp[0]*tmp[1] << ")" << std::endl; }
  printf("RMSE      : %f, RMSE%%: %f %%\n", RMSE, (RMSE/avg)*100);
  printf("error rate: %f, rate%%: %f %%\n", rate, (rate/avg)*100);
  printf("average   : %f, sizes : %dx%d\n",  avg, tmp[0], tmp[1]);
}

// ===== openctpu APIs end =====

struct mul_args{
  int tpu_id;
  std::string model_name;
  int A;
  int B;
  int ROW_BLK_CNT;
  int COL_BLK_CNT;
  int* a;
  int* b;
  int* c;
};

//struct Task_OP_queue{
//  struct mul_args op_args;
//  int xi, yi;
//};
//
//struct Task_OP_queue *Q;


struct mean_args{
  int tpu_id;
  int A;
  int B;
  int* a;
  int* c;
};

void *add_pthread(void *arguments){
  struct mul_args *args = (struct mul_args *)arguments;
  int tpu_id             = args->tpu_id;
  std::string model_name = args->model_name;
  int A                  = args->A;
  int B                  = args->B;
  int* a                 = args->a;
  int* b                 = args->b;
  int* c                 = args->c;
  unsigned long long int out_size = A*B;
  long long int dev_mem_ns = 0, dev_run_ns = 0, dev_pop_ns = 0;
  long long int mem_ns, run_ns, pop_ns;
  int output_size = 0;

  int* per_a = (int*) calloc(A*B ,sizeof(int));
  int* per_b = (int*) calloc(A*B ,sizeof(int));
std::cout << "exact_mode = " << exact_mode << std::endl;
  if(exact_mode == 0){
    run_element_wise_modelV2(a, b, out_size, ITER, output_size, c, tpu_id, /*model_id*/tpu_id, VERBOSE, mem_ns, run_ns, pop_ns);
    mtx.lock();
    dev_mem_ns += mem_ns;
    dev_run_ns += run_ns;
    dev_pop_ns += pop_ns;
    mtx.unlock();
  }else{
    for(int chunk_idx = 0 ; chunk_idx < sub_cnt(32, add_chunk_size) ; chunk_idx++){ // 5 chunks: 4+7+7+7+7=32
      if(chunk_idx%dev_cnt == tpu_id){
        int chunk_mask = (~(0xffffffff << add_chunk_size)) << (chunk_idx * add_chunk_size);
        for(int i = 0 ; i < out_size; i++){
          per_a[i] = (a[i] & chunk_mask) >> (chunk_idx * add_chunk_size);
          per_b[i] = (b[i] & chunk_mask) >> (chunk_idx * add_chunk_size);
          //if(i == 0){ std::cout << "per_a: " << per_a[i] << ", per_b: " << per_b[i] << std::endl;}
        }
        mtx.lock();
        run_element_wise_modelV3(per_a, per_b, out_size, ITER, output_size, c, tpu_id, /*model_id*/tpu_id, 0, chunk_idx, VERBOSE, mem_ns, run_ns, pop_ns);
        dev_mem_ns += mem_ns;
        dev_run_ns += run_ns;
        dev_pop_ns += pop_ns;
        mtx.unlock();
      }
    }
  }
  mtx.lock();
  std::cout << "dev " << std::to_string(tpu_id) << ": ";
  std::cout << "mem_ns: " << dev_mem_ns << ", run_ns: " << dev_run_ns << ", pop_ns: " << dev_pop_ns << std::endl;
  mtx.unlock();
  free(per_a);
  free(per_b);
}

void *mean_pthread(void *arguments){
  struct mean_args *args = (struct mean_args *)arguments;
  int tpu_id             = args->tpu_id;
  int A                  = args->A;
  int B                  = args->B;
  int* a                 = args->a;
  int* c                 = args->c;
  unsigned long long int out_size = A*B;
  long long int dev_mem_ns = 0, dev_run_ns = 0, dev_pop_ns = 0;
  long long int mem_ns, run_ns, pop_ns;
  int output_size = 0;
  int* per_a = (int*) calloc(A*B ,sizeof(int));
  if(exact_mode == 0){
    mtx.lock();
printf("before: c: %d\n", c[0]);
    run_modelV2(a, out_size, ITER, output_size, c, tpu_id, tpu_id/*model_id*/, data_type, 0, 0/*chunk_idx*/, VERBOSE, mem_ns, run_ns, pop_ns);
printf("after : c: %d\n", c[0]);
    dev_mem_ns += mem_ns;
    dev_run_ns += run_ns;
    dev_pop_ns += pop_ns;
    mtx.unlock();
  }else{
    for(int chunk_idx = 0 ; chunk_idx < sub_cnt(32, mean_chunk_size) ; chunk_idx++){ // 5 chunks: 4+7+7+7+7=32
      if(chunk_idx%dev_cnt == tpu_id){
        int chunk_mask = (~(0xffffffff << mean_chunk_size)) << (chunk_idx * mean_chunk_size);
        for(int i = 0 ; i < out_size; i++){
//          per_a[i] = (a[i] & chunk_mask) >> (chunk_idx * mean_chunk_size);
//          if(i < 10){ std::cout << "per_a: " << per_a[i] << std::endl;}
        }
        mtx.lock();
        run_modelV2(a, out_size, ITER, output_size, c, tpu_id, tpu_id/*model_id*/, data_type, 0, chunk_idx, VERBOSE, mem_ns, run_ns, pop_ns);
        dev_mem_ns += mem_ns;
        dev_run_ns += run_ns;
        dev_pop_ns += pop_ns;
        mtx.unlock();
      }
    }
  }
  mtx.lock();
  std::cout << "dev " << std::to_string(tpu_id) << ": ";
  std::cout << "mem_ns: " << dev_mem_ns << ", run_ns: " << dev_run_ns << ", pop_ns: " << dev_pop_ns << std::endl;
  mtx.unlock();
  free(per_a);
}

void *mul_pthread(void *arguments){
  struct mul_args *args = (struct mul_args *)arguments;
  int tpu_id             = args->tpu_id;
  std::string model_name = args->model_name;
  int A                  = args->A;
  int B                  = args->B;
  int ROW_BLK_CNT        = args->ROW_BLK_CNT;
  int COL_BLK_CNT        = args->COL_BLK_CNT;
  int* a                 = args->a;
  int* b                 = args->b;
  int* c                 = args->c;
  unsigned long long int out_size = A*B;
  long long int dev_mem_ns = 0, dev_run_ns = 0, dev_pop_ns = 0;
  long long int mem_ns, run_ns, pop_ns;
  int output_size = 0;
  if(exact_mode == 0){
    run_element_wise_modelV2(a, b, out_size, ITER, output_size, c, tpu_id, /*model_id*/tpu_id, VERBOSE, mem_ns, run_ns, pop_ns);
    dev_mem_ns += mem_ns;
    dev_run_ns += run_ns;
    dev_pop_ns += pop_ns;
  }else{ // exact_mode == 1
    int mask1;// = (~(0xffffffff << 4)) << (3 * 4);
    int mask2;// = (~(0xffffffff << 4)) << (2 * 4);
    int* per_a = (int*) calloc(A*B ,sizeof(int));
    int* per_b = (int*) calloc(A*B ,sizeof(int));
//TODO: currently support 16-bit input / 32-bit output
    int offset = 0; 
    for(int i = 0 ; i < ROW_BLK_CNT ; i++){
      for(int j = 0 ; j < COL_BLK_CNT ; j++){
        offset = (i*COL_BLK_CNT + j)*out_size;
        for(int xi = 0 ; xi < sub_cnt(16, mul_chunk_size) ; xi++){
          for(int yi = 0 ; yi < sub_cnt(16, mul_chunk_size) ; yi++){
            if((xi * sub_cnt(16, mul_chunk_size) + yi)%dev_cnt == tpu_id){
              mask1 = (~(0xffffffff << mul_chunk_size)) << (xi * mul_chunk_size);
              mask2 = (~(0xffffffff << mul_chunk_size)) << (yi * mul_chunk_size);
              for(int i = 0 ; i < out_size ; i++){
                per_a[i] = (a[i+offset] & mask1) >> (xi * mul_chunk_size);
                per_b[i] = (b[i+offset] & mask2) >> (yi * mul_chunk_size);
              }
              mtx.lock();
              run_element_wise_modelV3(per_a, per_b, out_size, ITER, output_size, &c[offset], tpu_id, /*model_id*/tpu_id, xi, yi, VERBOSE, mem_ns, run_ns, pop_ns);
              dev_mem_ns += mem_ns;
              dev_run_ns += run_ns;
              dev_pop_ns += pop_ns;
              mtx.unlock();
            }
          }
        }
      }
    }
    free(per_a);
    free(per_b);
  } // end exact_mode == 1
  mtx.lock();
  std::cout << "dev " << std::to_string(tpu_id) << ": ";
  std::cout << "mem_ns: " << dev_mem_ns << ", run_ns: " << dev_run_ns << ", pop_ns: " << dev_pop_ns << std::endl;
  mtx.unlock();
}
struct arg_mm2conv_struct{
  int tid;
  int** a;
  int** partial_c;
  int A, B, C;
  int blk_A, blk_B, blk_C;
  int ROW_BLK_CNT, INN_BLK_CNT, COL_BLK_CNT;
  float SCALE;
};
struct conv_args{
  int tpu_id;
  int A;
  int B;
  int blk_A;
  int blk_B;
  int A_pad;
  int B_pad;
  int ROW_BLK_CNT;
  int COL_BLK_CNT;
  int padding;
  int* a;
  int* c;
};

void *conv_pthread(void *arguments){
  struct conv_args *args = (struct conv_args *)arguments;
  int tpu_id        = args->tpu_id;
  int A             = args->A;
  int B             = args->B;
  int blk_A         = args->blk_A;
  int blk_B         = args->blk_B;
  int A_pad         = args->A_pad;
  int B_pad         = args->B_pad;
  int ROW_BLK_CNT   = args->ROW_BLK_CNT;
  int COL_BLK_CNT   = args->COL_BLK_CNT;
  int padding       = args->padding;
  int* a            = args->a;
  int* c            = args->c;
  unsigned long long int size     = (unsigned long long int)(blk_A*blk_B);
  int output_size = 0; //(unsigned long long int)(blk_A*blk_B);
  int* a_blk  = (int*) malloc(blk_A*blk_B*sizeof(int));
  int* a_pad  = (int*) malloc(A_pad*B_pad*sizeof(int));
  int* c_blk  = (int*) malloc(blk_A*blk_B*sizeof(int));
  long long int dev_mem_ns = 0, dev_run_ns = 0, dev_pop_ns = 0;
  long long int mem_ns, run_ns, pop_ns;
  int model_id, i, j, ii, jj;
  for(i = 0 ; i < ROW_BLK_CNT ; i++){
    for(j = 0 ;  j < COL_BLK_CNT ; j++){
      model_id = i*COL_BLK_CNT + j;
      if((model_id%dev_cnt) == tpu_id){
        partition_conv_input(a, a_blk, A, B, blk_A, blk_B, i , j);
        if(padding != 0){
          pad_input(a, a_blk, a_pad, A, B, blk_A, blk_B, A_pad, B_pad, padding, i, j, ROW_BLK_CNT, COL_BLK_CNT);
          size = A_pad*B_pad;
          mtx.lock();
          run_modelV2(a_pad, size, ITER, output_size, c_blk, tpu_id, tpu_id, data_type, 0, 0/*chunk_idx*/, VERBOSE, mem_ns, run_ns, pop_ns);
          dev_mem_ns += mem_ns;
          dev_run_ns += run_ns;
          dev_pop_ns += pop_ns;
          mtx.unlock();
        }else{
          mtx.lock();
          run_modelV2(a_blk, size, ITER, output_size, c_blk, tpu_id, model_id, data_type, 0, 0/*chunk_idx*/, VERBOSE, mem_ns, run_ns, pop_ns);
          dev_mem_ns += mem_ns;
          dev_run_ns += run_ns;
          dev_pop_ns += pop_ns;
          mtx.unlock();
        }
// put c_blk back to correct position in c
        for(ii = 0 ;  ii < blk_A ; ii++){
          for(jj = 0 ; jj < blk_B ; jj++){
            c[(ii+i*blk_A)*B+(jj+j*blk_B)] = c_blk[ii*blk_B+jj];
          }
        }
      } // end if
    }
  }
  mtx.lock();
  std::cout << "dev " << std::to_string(tpu_id) << ": ";
  std::cout << "mem_ns: " << dev_mem_ns << ", run_ns: " << dev_run_ns << ", pop_ns: " << dev_pop_ns << std::endl;
  mtx.unlock();
  free(a_blk);
  free(c_blk);
  free(a_pad);
}

void *mpmm_run(void *arguments){
  struct arg_mm2conv_struct *args = (struct arg_mm2conv_struct *)arguments;
  int tpu_id        = args->tid;
//  int A             = args->A;
//  int B             = args->B;
  long long int run_ns = 0;
  int A = 1024;
  for(int i = 0 ; i < A/dev_cnt ; i++){
    run_ns += invoke_model(tpu_id, ITER);
  }
  printf("mpmm: dev[%d] invoke time: %f (us)\n", tpu_id, run_ns/1000.0);

}

float gptpu_main(int* a, int* b, int* c, int A, int B, int blk_row, int blk_col, int start_i, int start_j, float scale, const std::string& model_name, int tf, bool transpose){
  //===== init params =====
  timing total_start = clk::now();
  
  //std::cout << model_name << ": A: " << std::to_string(A) << ", B: " << std::to_string(B) << ", blk_row: " << std::to_string(blk_row) << ", blk_col: " << std::to_string(blk_col) << ", start_i: " << std::to_string(start_i) << ", start_j: " << std::to_string(start_j) << std::endl;
  std::string matrix_path    = tempG_dir+"/"+model_name+"_tflite/"+model_name+"_quant.tflite";  
  timing set_start, set_end, new_start, new_end, save_start, save_end, py_start,py_end;
  double make_tflite_model_us, py_us, total_us, set_us, new_us, model_us, itpr_init_us, itpr_us, mem_us, run_us, pop_us, out_buf_us, exe_us, sum_us, trans_us;
  int ret, input_mode, model_id = 0, padding = 0/*for conv2D*/, A_pad = A, B_pad = B;
  union scaling matrix_s, bias_s;
  unsigned long long int size, out_size, M/*conv's filter width*/, N/*conv's filter height*/;
  int* c_result;
  std::string mm_input_path_name;
  int chunk_num = CHAR_BIT/get_chunk_size();
// For conv2D, the special cass use
// ================================
  int max_blk = 1024;  
  int blk_A = (A >= MM_BLK_ROW)?max_blk/*MM_BLK_ROW*/:A;
  int blk_B = (B >= MM_BLK_COL)?max_blk/*MM_BLK_COL*/:B;
  int ROW_BLK_CNT = sub_cnt(A, blk_A); //  (A / blk_A) + ((A % blk_A != 0)?1:0);  
  int COL_BLK_CNT = sub_cnt(B, blk_B); //  (B / blk_B) + ((B % blk_B != 0)?1:0);  
  int ROW_BLK_REM = A % blk_A;
  int COL_BLK_REM = B % blk_B;
  std::cout << __func__ << ": A: " << A << ", B: " << B << ", blk_A: " << blk_A << ", blk_B: " << blk_B << std::endl;
  if(model_name == mv_model || model_name == bmv_model){
    size     = (unsigned long long int)(A*B);
    out_size = (unsigned long long int)(A);
  }else if(model_name == add_model     || model_name == sub_model || model_name == mul_model || 
           model_name == maxpool_model || model_name == log_model || model_name == tanh_model || model_name == relu_model){
    size     = (unsigned long long int)(blk_A*blk_B);
    out_size = (unsigned long long int)(blk_A*blk_B); 
    matrix_path    = data_dir+model_name+"_tflite/"+model_name+"_"+std::to_string(blk_A)+"x"+std::to_string(blk_B)+"_quant.tflite";  
  }else if(model_name == mean_model || model_name == max_model){
    size     = (unsigned long long int)(A*B);
    out_size = 1; // reduction operation(s)
  }else if(model_name == conv_model){
    size     = (unsigned long long int)(A*B);
    out_size = (unsigned long long int)(A*B);
    M        = (unsigned long long int)(blk_row); // just reuse the parameter 
    N        = (unsigned long long int)(blk_col); // same
    padding  = (unsigned long long int)(start_i); // same
  }else if(model_name == vs_model){
    size     = (unsigned long long int)(B);
    out_size = 1;
  }else if(model_name == crop_model){
    size     = (unsigned long long int)(A*B);
    out_size = (unsigned long long int)(blk_row*blk_col); 
  }else if(model_name == ext_model){
    size     = (unsigned long long int)(blk_row*blk_col);
    out_size = (unsigned long long int)(A*B); 
  }else if(model_name == "black_model"){
    size     = (unsigned long long int)(A);
    out_size = (unsigned long long int)(B);
    matrix_path    = "/usr/local/gptpu/"+model_name+"_tflite/"+model_name+"_"+std::to_string(A)+"x"+std::to_string(B)+"_quant.tflite";  
  }else{
    std::cout << "undefined model name: " << model_name << std::endl;
    return 1;
  }
  if(tf == 0){
    assert(("offset system doesn't support small data matrix size", size >= 8));
  }
  char* data_array = (char*)malloc(size*sizeof(char));

//  itpr_init_ns = interpreter_initialization(/*model_cnt*/dev_cnt);

// ===== create ramdisk using tmpfs =====
// TODO: good policy to maintain tmpfs allocation and deletion when calling
  struct sysinfo myinfo;
  unsigned long mem;
  sysinfo(&myinfo);
  mem = myinfo.mem_unit * myinfo.freeram;
  //std::cout << "mem size: " << mem << std::endl;
  std::string command;
  command = "sudo mount -t tmpfs -o size=8G tmpfs /mnt/ramdisk";
  //system(command.c_str());

// ===== generate tflite =====
  timing make_start = clk::now();
  std::string sub_path =  model_name+"_temp.tflite";
  template_path  = temp_dir + sub_path;
  //std::cout << template_path << std::endl;
  if(file_exist(template_path) == false && model_name != conv_model){
    std::cout << "required template file: " << template_path << " doesn't exist." << std::endl;
    std::cout << "making template, out_path: " << matrix_path << ", template path:" << template_path << std::endl;
    command = "python3 "+PWD+"/src/create_model.py --platform=m2 --model="+model_name+" --outfile_name="+matrix_path+" --in_size="+std::to_string(A)+" --out_size="+std::to_string(B)+" --blk_row="+std::to_string(blk_row)+" --blk_col="+std::to_string(blk_col)+" --start_i="+std::to_string(start_i)+" --start_j="+std::to_string(start_j)+" --ramdisk="+std::to_string(ramdisk);
    std::cout << command << std::endl;
    py_start = clk::now();
    int ret = system(command.c_str());
    py_end   = clk::now();
    make_temp(model_name, matrix_path/*in_path*/, local_temp_dir+sub_path/*out_path*/); 
    command = "sudo cp "+local_temp_dir+sub_path+" "+template_path;
    ret = system(command.c_str());
    std::cout << command << std::endl;
  }
  new_start = clk::now();
  if(model_name == crop_model){
    matrix_path = data_dir+model_name+"_tflite/"+model_name+"_["+std::to_string(A)+"x"+std::to_string(B)+"]_["+std::to_string(blk_row)+"x"+std::to_string(blk_col)+"]_["+std::to_string(start_i)+"_"+std::to_string(start_j)+"]_quant.tflite";  
    ret = create_crop_tflite(matrix_path, A, B, blk_row, blk_col, start_i, start_j, VERBOSE);
  }else if(model_name == sub_model){
    matrix_path = data_dir+model_name+"_tflite/"+model_name+"_["+std::to_string(A)+"x"+std::to_string(B)+"]_quant.tflite";  
    ret = create_sub_tflite(matrix_path, A, B, VERBOSE);
    //command = "python3 ./../src/create_model.py --platform=m2 --model="+model_name+" --outfile_name="+matrix_path+" --in_size="+std::to_string(A)+" --out_size="+std::to_string(B)+" --blk_row="+std::to_string(blk_row)+" --blk_col="+std::to_string(blk_col)+" --start_i="+std::to_string(start_i)+" --start_j="+std::to_string(start_j)+" --ramdisk="+std::to_string(ramdisk);
    //py_start = clk::now();
    //int ret = system(command.c_str());
    //py_end   = clk::now();
  }else if(model_name == conv_model){
      std::string weight_file_name = "./../conv_filter_"+std::to_string(M)+"x"+std::to_string(N); // reuse is fine, temp one
      conv_save_weight(b, weight_file_name, M, N);
      if(padding != 0){ 
        A_pad = blk_A + (M-1); //TODO: support even number size of filter later
        B_pad = blk_B + (N-1);
      }
      matrix_path = data_dir + model_name + "_tflite/" + model_name + "_"+std::to_string(M)+"x"+std::to_string(N)+"_quant_edgetpu.tflite";
      command = "python3 "+PWD+"/src/create_model.py --platform=m2 --model="+model_name+" --in_w_name="+weight_file_name+" --outfile_name="+matrix_path+" --IN_W="+std::to_string(A_pad)+" --IN_H="+std::to_string(B_pad)+" --IN_C="+std::to_string(1)+" --OUT_C="+std::to_string(1)+" --F_W="+std::to_string(M)+" --F_H="+std::to_string(N)+" --S_W=1 --S_H=1"+" --PADDING=none"+" --out_scale="+itoa(SCALE)+" --ramdisk="+std::to_string(ramdisk);
      std::cout << "command: " << command.c_str() << std::endl;
      py_start = clk::now();
      int ret = system(command.c_str());
      py_end   = clk::now();
  }else{ // all other operations
    command = "python3 "+PWD+"/src/create_model.py --platform=m2 --model="+model_name+" --outfile_name="+matrix_path+" --in_size="+std::to_string(blk_A)+" --out_size="+std::to_string(blk_B)+" --out_scale="+itoa(SCALE)+" --blk_row="+std::to_string(blk_row)+" --blk_col="+std::to_string(blk_col)+" --start_i="+std::to_string(start_i)+" --start_j="+std::to_string(start_j)+" --ramdisk="+std::to_string(ramdisk);
    py_start = clk::now();
    int ret = system(command.c_str());
    py_end   = clk::now();
    
  }
  new_end   = clk::now();
  //new_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(new_end - new_start).count(); 
  timing make_end = clk::now();

// prepare edgetpu run
  for(int i = 0 ; i < dev_cnt ; i++){
    model_ns += build_model(matrix_path, /*model_id*/i);
    itpr_ns  += build_interpreter(/*tpu_id*/i, /*model_id*/i);
  }
  timing exe_start = clk::now();
  int output_size = 0;
  if(model_name == add_model){ /*element-wise model*/
    // TODO: dispatch  N chunks to dev_cnt devices
    // make sure that c is zero before summation
    for(int i = 0 ; i < A*B ; i++){
      c[i] = 0;
    }
    if(exact_mode == 0){
      struct OP_node *curr_node = (struct OP_node*) calloc(ROW_BLK_CNT*COL_BLK_CNT, sizeof(struct OP_node)); 
      int cnt = 0;
      for(int i = 0 ; i < ROW_BLK_CNT ; i++){
        for(int j = 0 ; j < COL_BLK_CNT ; j++){
          curr_node[cnt].op = model_name;
          curr_node[cnt].A  = blk_A;
          curr_node[cnt].B  = blk_B;
          curr_node[cnt].i  = i;
          curr_node[cnt].j  = j;
          curr_node[cnt].ROW_BLK_CNT = ROW_BLK_CNT;
          curr_node[cnt].COL_BLK_CNT = COL_BLK_CNT;
          curr_node[cnt].a = a;
          curr_node[cnt].b = b;
          curr_node[cnt].c = c;
          curr_node[cnt].w_chunk_idx = 0;
          curr_node[cnt].offset = (i*COL_BLK_CNT+j)*out_size;
          fifo_push(SPMC_fifo, &curr_node[cnt]);
          cnt++;
          if(i == (ROW_BLK_CNT-1) && j == (COL_BLK_CNT-1)){ pthread_mutex_lock(&pmtx); done_enqueue = true; pthread_mutex_unlock(&pmtx); }
        }
      }
      std::cout << "producer ends enqueue, start to wait" << std::endl;
      wait_queue_all_finished();
    }else{
      struct OP_node *curr_node = (struct OP_node*) calloc(ROW_BLK_CNT*COL_BLK_CNT*sub_cnt(32, add_chunk_size), sizeof(struct OP_node)); 
      int cnt = 0;
      for(int i = 0 ; i < ROW_BLK_CNT ; i++){
        for(int j = 0 ; j < COL_BLK_CNT ; j++){
          for(int chunk_idx = 0 ; chunk_idx < sub_cnt(32, add_chunk_size) ; chunk_idx++){ // 5 chunks: 4+7+7+7+7=32
            curr_node[cnt].op = model_name;
            curr_node[cnt].A  = blk_A;
            curr_node[cnt].B  = blk_B;
            curr_node[cnt].i  = i;
            curr_node[cnt].j  = j;
            curr_node[cnt].ROW_BLK_CNT = ROW_BLK_CNT;
            curr_node[cnt].COL_BLK_CNT = COL_BLK_CNT;
            curr_node[cnt].a = a;
            curr_node[cnt].b = b;
            curr_node[cnt].c = c;
            curr_node[cnt].w_chunk_idx = chunk_idx;
            curr_node[cnt].offset = (i*COL_BLK_CNT+j)*out_size;
            fifo_push(SPMC_fifo, &curr_node[cnt]);
            cnt++;
            if(i == (ROW_BLK_CNT-1) && j == (COL_BLK_CNT-1) && chunk_idx == sub_cnt(32, add_chunk_size) -1){ pthread_mutex_lock(&pmtx); done_enqueue = true; pthread_mutex_unlock(&pmtx); }
          }
        }
      }
      std::cout << "producer ends enqueue, start to wait" << std::endl;
      wait_queue_all_finished();
    }
  }else if(model_name == sub_model || model_name == mul_model){
    for(int i = 0 ; i < A*B ; i++){
      c[i] = 0;
    }
    if(exact_mode == 0){
      struct OP_node *curr_node = (struct OP_node*) calloc(ROW_BLK_CNT*COL_BLK_CNT, sizeof(struct OP_node)); 
      int cnt = 0;
      for(int i = 0 ; i < ROW_BLK_CNT ; i++){
        for(int j = 0 ; j < COL_BLK_CNT ; j++){
          curr_node[cnt].op = model_name;
          curr_node[cnt].A  = blk_A;
          curr_node[cnt].B  = blk_B;
          curr_node[cnt].i  = i;
          curr_node[cnt].j  = j;
          curr_node[cnt].ROW_BLK_CNT = ROW_BLK_CNT;
          curr_node[cnt].COL_BLK_CNT = COL_BLK_CNT;
          curr_node[cnt].a = a;
          curr_node[cnt].b = b;
          curr_node[cnt].c = c;
          curr_node[cnt].xi = 0;
          curr_node[cnt].yi = 0;
          curr_node[cnt].offset = (i*COL_BLK_CNT+j)*out_size;
          fifo_push(SPMC_fifo, &curr_node[cnt]);
          cnt++;
          if(i == (ROW_BLK_CNT-1) && j == (COL_BLK_CNT-1)){ pthread_mutex_lock(&pmtx); done_enqueue = true; pthread_mutex_unlock(&pmtx); }
        }
      }
      wait_queue_all_finished();
      free(curr_node);
    }else{ // exact_mode == 1
      struct OP_node *curr_node = (struct OP_node*) calloc(ROW_BLK_CNT*COL_BLK_CNT*sub_cnt(16, mul_chunk_size)*sub_cnt(16, mul_chunk_size), sizeof(struct OP_node)); 
      int cnt = 0;
      for(int i = 0 ; i < ROW_BLK_CNT ; i++){
        for(int j = 0 ; j < COL_BLK_CNT ; j++){
          for(int xi = 0 ; xi < sub_cnt(16, mul_chunk_size) ; xi++){
            for(int yi = 0 ; yi < sub_cnt(16, mul_chunk_size) ; yi++){
                curr_node[cnt].op = model_name;
                curr_node[cnt].A  = blk_A;
                curr_node[cnt].B  = blk_B;
                curr_node[cnt].i  = i;
                curr_node[cnt].j  = j;
                curr_node[cnt].ROW_BLK_CNT = ROW_BLK_CNT;
                curr_node[cnt].COL_BLK_CNT = COL_BLK_CNT;
                curr_node[cnt].a = a;
                curr_node[cnt].b = b;
                curr_node[cnt].c = c;
                curr_node[cnt].xi = xi;
                curr_node[cnt].yi = yi;
                curr_node[cnt].offset = (i*COL_BLK_CNT+j)*out_size;
                fifo_push(SPMC_fifo, &curr_node[cnt]);
                cnt++;
              if(i == (ROW_BLK_CNT-1) && j == (COL_BLK_CNT-1) && xi == (sub_cnt(16, mul_chunk_size)-1) && yi == (sub_cnt(16, mul_chunk_size)-1)){ pthread_mutex_lock(&pmtx); done_enqueue = true; pthread_mutex_unlock(&pmtx); }
            }
          }
        }
      }
      wait_queue_all_finished();
      free(curr_node);
    }
  }else{
    if(model_name == crop_model && transpose == true){
      c_result = (int*) malloc(blk_row*blk_col*sizeof(int));
      run_modelV2(a, size, ITER, output_size, c_result, /*tpu_id*/0, /*model_id*/0, data_type, 0, 0/*chunk_idx*/, VERBOSE, mem_ns, run_ns, pop_ns);
    }else if(model_name == crop_model && transpose == true){
      c_result = (int*) malloc(A*B*sizeof(int));
      run_modelV2(a, size, ITER, output_size, c_result, /*tpu_id*/0, /*model_id*/0, data_type, 0, 0/*chunk_idx*/,VERBOSE, mem_ns, run_ns, pop_ns);
    }else if(model_name == conv_model){
      // need padding input if needed
//TODO: partition conv2D input 2D array to 2Kx2K blocks since edgetpu_compiler can't generate sizes exceeding 2K
      pthread_t tid[dev_cnt];
      struct conv_args args[dev_cnt];
      for(int i = 0 ; i < dev_cnt ; i++){
        args[i].tpu_id      = i;
        args[i].A           = A;
        args[i].B           = B;
        args[i].blk_A       = blk_A;
        args[i].blk_B       = blk_B;
        args[i].A_pad       = A_pad;
        args[i].B_pad       = B_pad;
        args[i].ROW_BLK_CNT = ROW_BLK_CNT;
        args[i].COL_BLK_CNT = COL_BLK_CNT;
        args[i].padding     = padding;
        args[i].a           = a;
        args[i].c           = c;
        pthread_create(&tid[i], NULL, conv_pthread, (void *)&args[i]);
      }
      for(int i = 0 ; i < dev_cnt ; i++){
        pthread_join(tid[i], NULL);
      }
    }else if(model_name == mean_model){
//      std::cout << "size: " << size << ", output_size: " << output_size << std::endl;
//      run_modelV2(a, size, ITER, output_size, c, /*tpu_id*/0, /*model_id*/0, data_type, 0, 0/*chunk_idx*/, VERBOSE, mem_ns, run_ns, pop_ns);
      pthread_t tid[dev_cnt];
      struct mean_args args[dev_cnt];
      for(int i = 0 ; i < dev_cnt ; i++){
        args[i].tpu_id      = i;
        args[i].A           = A;
        args[i].B           = B;
        args[i].a           = a;
        args[i].c           = c;
        pthread_create(&tid[i], NULL, mean_pthread, (void *)&args[i]);
      }
      for(int i = 0 ; i < dev_cnt ; i++){
        pthread_join(tid[i], NULL);
      }
    }else{
      std::cout << "run other op" << std::endl;
      run_modelV2(a, size, ITER, output_size, c, /*tpu_id*/0, /*model_id*/0, data_type, 0, 0/*chunk_idx*/, VERBOSE, mem_ns, run_ns, pop_ns);
    }
  }
  timing exe_end   = clk::now();
  timing trans_start = clk::now();
  if(transpose == true){
    if(model_name == crop_model || model_name == ext_model){
      // transpose c
      for(int i = 0 ; i < blk_row ; i++){
        for(int j = 0 ; j < blk_col ; j++){
          c[i*blk_col+j] = c_result[j*blk_row+i];
        }
      }
    }else{
      std::cout << "operation " << model_name << " (other than crop) tries to transpose the result, which is not implemented yet." << std::endl;
      exit(0);
    }
  }
  timing trans_end   = clk::now();
  timing total_end = clk::now();

  if(BREAKDOWN == 1){
//    std::cout << std::fixed;
    std::cout << std::setw(12);
    make_tflite_model_us = std::chrono::duration_cast<std::chrono::nanoseconds>(make_end  - make_start ).count()/1000.0;
    py_us                = std::chrono::duration_cast<std::chrono::nanoseconds>(py_end    - py_start   ).count()/1000.0;
    total_us             = std::chrono::duration_cast<std::chrono::nanoseconds>(total_end - total_start).count()/1000.0;
    set_us               = std::chrono::duration_cast<std::chrono::nanoseconds>(set_end   - set_start  ).count()/1000.0;
    new_us               = std::chrono::duration_cast<std::chrono::nanoseconds>(new_end   - new_start  ).count()/1000.0;
    model_us             = model_ns/1000.0;
    itpr_init_us         = itpr_init_ns /1000.0;
    itpr_us              = itpr_ns /1000.0;
    mem_us               = mem_ns  /1000.0;
    run_us               = run_ns  /1000.0;

    FILE* fp;    
    fp = fopen("./record.txt", "a");
    fprintf(fp, "%s, A:%d, B:%d, blk_row:%d, blk_col:%d, run_us:%f, iter:%d\n", model_name.c_str(), A, B, blk_row, blk_col, run_us, ITER);
    fclose(fp);

    pop_us               = pop_ns  /1000.0;
    exe_us               = std::chrono::duration_cast<std::chrono::nanoseconds>(exe_end   - exe_start  ).count()/1000.0;
//    std::cout << std::setprecision(12);
    printf("+----- gptpu_tf%d_%s timing breakdown ---------------------+\n", tf, model_name.c_str());
//    printf("|make tflite model     : [%7.3f%]\t%12.3f (us).|\n", (make_tflite_model_us/total_us)*100, make_tflite_model_us);
//    printf("|  itpr initialization : [%7.3f%%]\t%12.3f (us).\t|\n", (itpr_us/total_us)*100, itpr_us);  
    printf("+----- make model breakdown below ------------------------------+ average, each time may differ \n");
    if(tf == 1){
      printf("| python make model    : [%7.3f%%]\t%12.3f (us).\t|\n", (py_us/total_us)*100, py_us);  
    }else{
      printf("| input data converting: [%7.3f%%]\t%12.3f (us).\t|\n", (set_us/total_us)*100, set_us);  
      printf("| create tflite lite   : [%7.3f%%]\t%12.3f (us).\t|\n", (new_us/total_us)*100, new_us);  
    }
    printf("+----- run detail below ----------------------------------------+\n");
    printf("| build model          : [%7.3f%%]\t%12.3f (us).\t|\n", (model_us/total_us)*100, model_us);  
    printf("| build itpr           : [%7.3f%%]\t%12.3f (us).\t|\n", (itpr_us/total_us)*100, itpr_us);  
    printf("| transfer input       : [%7.3f%%]\t%12.3f (us).\t|\n", (mem_us/total_us)*100, mem_us);  
    printf("| run (invoke)         : [%7.3f%%]\t%12.3f (us).\t|\n", (run_us/total_us)*100, run_us);  
    printf("| populate out buffer  : [%7.3f%%]\t%12.3f (us).\t|\n", (pop_us/total_us)*100, pop_us);  
    printf("+---------------------------------------------------------------+\n");
    printf("| total                : [%7.3f%%]\t%12.3f (us).\t| (some impl. related overhead ignored for now)\n", (total_us/total_us)*100, total_us);  
    printf("+---------------------------------------------------------------+\n");
    long long int ns     = run_ns;
    long long int term1 = ITER * B;
    long long int term2 = (2 * A) - 1;
    long long int op_cnt = term1 * term2; 
    printf("| GOPS: %10.4f, op_cnt: %13lld, us: %12.3f\t| (valid for mm only now)\n", (float)op_cnt/ns, op_cnt, run_us);   
    printf("+---------------------------------------------------------------+\n");
    //std::cout << "|malloc partial       : " << std::chrono::duration_cast<std::chrono::nanoseconds>(malloc_e - malloc_s).count()/1000.0 << "\t(us).\t\t|" << std::endl; 
  }
  if(model_name == crop_model && transpose){
    free(c_result);
  }
  free(data_array);
  return run_us;
}

struct arg_struct{
  int tpu_id;
  int* b; // pointer to matrix b
  int A;
  int B;
  int in;
  int ROW_BLK_CNT;
  int INNER_BLK_CNT;
  int COL_BLK_CNT;
  int** partial_c; // pointer to matrix partial_c
  bool b_major;
};

#ifdef __aarch64__
  void run_func(int* b, int A, int B, int in, int ROW_BLK_CNT, int INNER_BLK_CNT, int COL_BLK_CNT, int** partial_c, bool b_major){
  int tpu_id = 0;
#else
void *run_func(void *arguments){
  struct arg_struct *args = (struct arg_struct *)arguments;
  int tpu_id        = args->tpu_id;
  int* b            = args->b;
  int A             = args->A;
  int B             = args->B;
  int in            = args->in;
  int ROW_BLK_CNT   = args->ROW_BLK_CNT;
  int INNER_BLK_CNT = args->INNER_BLK_CNT;
  int COL_BLK_CNT   = args->COL_BLK_CNT;
  int** partial_c   = args->partial_c;
  bool b_major      = args->b_major;
#endif
  int output_size = 0;
  int model_id = 0;
  long long int dev_mem_ns = 0, dev_run_ns = 0, dev_pop_ns = 0, dev_out_buf_ns = 0;
  long long int mem1_ns, run1_ns, pop1_ns; 
  int chunk_num = CHAR_BIT/get_chunk_size();
  int i, j, k, l, tmp, b_offset, c_offset;
  if(b_major == true){
    for(i = 0 ; i < ROW_BLK_CNT ; i++){
      for(j = 0 ; j < INNER_BLK_CNT; j++){ // thickness of partial sum map
        for(int w_chunk_idx = 0 ; w_chunk_idx < chunk_num ; w_chunk_idx++){
          model_id = (i*INNER_BLK_CNT + j)*chunk_num + w_chunk_idx; // iterate per block model of A
          if((model_id%dev_cnt) == tpu_id){
            itpr_ns  += build_interpreter((i*INNER_BLK_CNT+j)%dev_cnt, model_id); // per task
            for(k = 0 ; k < B ; k++){ // per slice vector      
              for(int in_chunk_idx = 0 ; in_chunk_idx < chunk_num ; in_chunk_idx++){  
                b_offset = k*(in)+j*(MM_BLK_COL);
                c_offset = k*in+i*MM_BLK_ROW;
mtx.lock();  
                run_modelV2(b+b_offset, MM_BLK_COL, ITER, output_size, partial_c[j]+c_offset, tpu_id, model_id, data_type, w_chunk_idx, in_chunk_idx, VERBOSE, mem1_ns, run1_ns, pop1_ns);
//                std::cout << "mem_ns: " << mem1_ns << ", run_ns: " << run1_ns << ", pop_ns: " << pop1_ns << std::endl;
mtx.unlock();  
                dev_mem_ns += mem1_ns;
                dev_run_ns += run1_ns;
                dev_pop_ns += pop1_ns;
              }
            }
          }
        } // end if
      }
    }  
  }else{
    for(i = 0 ; i < ROW_BLK_CNT ; i++){
      for(j = 0 ; j < INNER_BLK_CNT; j++){ // thickness of partial sum map
        for(int w_chunk_idx = 0 ; w_chunk_idx < chunk_num ; w_chunk_idx++){
          //model_id = i*INNER_BLK_CNT + j;
          model_id = (i*INNER_BLK_CNT + j)*chunk_num + w_chunk_idx; // iterate per block model of A
          if((model_id%dev_cnt) == tpu_id){
            itpr_ns  += build_interpreter((i*INNER_BLK_CNT+j)%dev_cnt, model_id); // per task
            for(k = 0 ; k < COL_BLK_CNT; k++){ // row direction block cnt for output matrix
              for(l = 0 ; l < MM_BLK_ROW; l++){  // block column width for output matrix
                for(int in_chunk_idx = 0 ; in_chunk_idx < chunk_num ; in_chunk_idx++){
                  b_offset = (k*MM_BLK_ROW+l)*in+j*MM_BLK_COL;
                  c_offset = k*(MM_BLK_ROW*MM_BLK_COL*COL_BLK_CNT)+i*(MM_BLK_ROW)+l*A;
mtx.lock();  
                  run_modelV2(b+b_offset, MM_BLK_COL, ITER, output_size, partial_c[j]+c_offset, tpu_id, model_id, data_type, w_chunk_idx, in_chunk_idx, VERBOSE, mem1_ns, run1_ns, pop1_ns);
                  std::cout << "mem_ns: " << mem1_ns << ", run_ns: " << run1_ns << ", pop_ns: " << pop1_ns << std::endl;
mtx.unlock();  
                  dev_mem_ns += mem1_ns;
                  dev_run_ns += run1_ns;
                  dev_pop_ns += pop1_ns;
                }
              }
            }
          }
        } // end if
      }
    }  
  }// end else on b_major
  // max out timing out of all device(s)
  mtx.lock();
  std::cout << "dev " << std::to_string(tpu_id) << ": ";
  std::cout << "mem_ns: " << dev_mem_ns << ", run_ns: " << dev_run_ns << ", pop_ns: " << dev_pop_ns << std::endl;
  mem_ns     = MAX(mem_ns, dev_mem_ns);
  run_ns     = MAX(run_ns, dev_run_ns);
  pop_ns     = MAX(pop_ns, dev_pop_ns);
  mtx.unlock();

}

// blocking algorithm on MM operation
int gptpu_block(int* a, int* b, bool b_major, int* c, int A ,int in, int B, float scale, const std::string& model_name, int tf){
  timing total_start = clk::now();
  //===== init params =====
  std::string matrix_path    = data_dir+model_name+"_tflite/"+model_name+"_quant.tflite";  
  timing set_start, set_end, new_start, new_end, exe_start, exe_end, save_start, save_end, py_start, py_end;
  int ret, input_mode, model_id = 0;
  int ROW_BLK_CNT, INNER_BLK_CNT, COL_BLK_CNT = 0, matrix_first_dim, ROW_BLK_REM, COL_BLK_REM;
  int chunk_num = CHAR_BIT/get_chunk_size();
  matrix_first_dim = (b_major == true)?(A):(B);

  // automatically select proper block size for given input matrix size based on perfomance consideration
  if(BLOCK_IS_SET == false){select_blk_shape(A, in, B, MM_BLK_ROW, MM_BLK_COL, b_major);}

  ROW_BLK_CNT   = sub_cnt(matrix_first_dim, MM_BLK_ROW);//  (matrix_first_dim  / MM_BLK_ROW) + ((matrix_first_dim  % MM_BLK_ROW != 0)?1:0);  
  INNER_BLK_CNT = sub_cnt(in, MM_BLK_COL);              //  (in                / MM_BLK_COL) + ((in                % MM_BLK_COL != 0)?1:0);
  ROW_BLK_REM   = matrix_first_dim % MM_BLK_ROW; // TODO
  COL_BLK_REM   = in               % MM_BLK_COL; 
  if(b_major == false){ COL_BLK_CNT = sub_cnt(B, MM_BLK_ROW); /*COL_BLK_CNT = B / MM_BLK_ROW + ((B               % MM_BLK_ROW != 0)?1:0);*/ }  
  std::cout << "A:" << A << ", in : " << in << ", B: " << B << ", MM_BLK_ROW: " << MM_BLK_ROW << ", MM_BLK_COL: " << MM_BLK_COL << std::endl;
  ASSERT(matrix_first_dim >= MM_BLK_ROW, ((b_major == true)?("A: "):("B: ") + std::to_string(matrix_first_dim)  + " is smaller than MM_BLK_ROW: " + std::to_string(MM_BLK_ROW) + " when b is in "+((b_major==true)?("col-major."):("row-major."))).c_str());
//  ASSERT(in >= MM_BLK_COL,  (" in: " + std::to_string(in) + " is smaller than MM_BLK_COL: " + std::to_string(MM_BLK_COL)).c_str());

//  std::cout << "ROW_BLK_CNT  : " << ROW_BLK_CNT << ", A: " << A << ", MM_BLK_ROW: " << MM_BLK_ROW << std::endl;
//  itpr_init_ns = interpreter_initialization(ROW_BLK_CNT*INNER_BLK_CNT*chunk_num);

  double make_tflite_model_us, py_us, total_us, set_us, new_us, model_us, itpr_init_us, itpr_us, mem_us, run_us, pop_us, out_buf_us, exe_us, sum_us, trans_us;
  int blk_cnt;

  // allocate partial c
  timing malloc_s = clk::now(); 
  partial_c = (int**)malloc(INNER_BLK_CNT*sizeof(int*));
  for(int i = 0 ; i < INNER_BLK_CNT; i++){
    partial_c[i] = (int*)calloc(A*B,sizeof(int));
  }
  timing malloc_e = clk::now(); 

  union scaling matrix_s, bias_s;
  unsigned long long int size;
  std::string mm_input_path_name, command;
  if(model_name == mm_model || model_name == mv_model || model_name == imv_model){
    // a[A][in] weight, but in tflite model [in] as input size, [A] as output size
    // b[in][B] input vector: b[:][idx]
    // c[A][B]  result
    // c = a * b
    size = (unsigned long long int)(MM_BLK_ROW * MM_BLK_COL);  //per block size
    if(tf == 0){
      assert(("offset system doesn't support small data matrix size", size >= 8));
    }
  }else{
    std::cout << "MM_model only, undefined model name: " << model_name << std::endl;
    exit(0);
  }
  char* data_array = (char*)malloc(MM_BLK_ROW*MM_BLK_COL*sizeof(char));
  long long int set_ns = 0, new_ns = 0; 
// ===== generate tflite =====
  timing make_start = clk::now();
  if(tf == 0){ // gptpu-generated tflite
    // =====  create template tflite from tf-generated tflite =====
    std::string sub_path =  "dense_temp_"+std::to_string(MM_BLK_ROW)+"x"+std::to_string(MM_BLK_COL)+".tflite";
    template_path  = temp_dir + sub_path;
    //std::cout << template_path << std::endl;
    if(file_exist(template_path) == false){
      std::cout << "required template file: " << template_path << " doesn't exist." << std::endl;
      std::cout << "making template, out_path: " << matrix_path << ", template path:" << template_path << std::endl;
      command = "python3 "+PWD+"/src/create_model.py --platform=m2 --model="+model_name+" --outfile_name="+matrix_path+" --in_size="+std::to_string(MM_BLK_COL)+" --out_size="+std::to_string(MM_BLK_ROW)+" --ramdisk="+std::to_string(ramdisk);
      int ret = system(command.c_str());
      std::cout << "during creating template, ori matrix_path for template: " << matrix_path << std::endl;
      make_dense_temp(matrix_path/*in_path*/, local_temp_dir+sub_path/*out_path*/); 
      command = "sudo cp "+local_temp_dir+sub_path+" "+template_path;
      ret = system(command.c_str());
    }
    // ===== blocking algorithm =====
    for(int i = 0 ; i < ROW_BLK_CNT ; i++){
      for(int j = 0 ; j < INNER_BLK_CNT ; j++){
        for(int w = 0 ; w < chunk_num ; w++){
          set_start = clk::now(); // data quantization
          set_block_array(a, b_major, data_array, (b_major==true)?i:j, (b_major==true)?j:i, A, in, ROW_BLK_REM, COL_BLK_REM, w);
          set_end   = clk::now();
          matrix_path = data_dir+model_name+"_tflite/"+model_name+"_["+std::to_string(i)+"]["+std::to_string(j)+"]_["+std::to_string(MM_BLK_ROW)+"x"+std::to_string(MM_BLK_COL)+"]["+std::to_string(A)+"x"+std::to_string(in)+"]_chunk"+std::to_string(w)+"of"+std::to_string(chunk_num)+"_quant.tflite";  
          new_start = clk::now();
          bias_s.f = matrix_s.f = scale; // the local scale
printf("the local scale: %f\n", scale);
          ret = create_dense_tflite(matrix_path, MM_BLK_ROW, MM_BLK_COL, data_array, matrix_s.c, bias_s.c, &ZERO_POINT, data_type, VERBOSE);
          new_end   = clk::now();
          set_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(set_end - set_start).count(); 
          new_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(new_end - new_start).count(); 
        }
      }
    }
  }else{ // use tf-generated tflite
    py_start = clk::now();
    for(int i = 0 ; i < ROW_BLK_CNT ; i++){
      for(int j = 0 ; j < INNER_BLK_CNT ; j++){
        for(int w = 0 ; w < chunk_num ; w++){
          matrix_path = data_dir+model_name+"_tflite/"+model_name+"_["+std::to_string(i)+"]["+std::to_string(j)+"]_["+std::to_string(MM_BLK_ROW)+"x"+std::to_string(MM_BLK_COL)+"]["+std::to_string(A)+"x"+std::to_string(in)+"]_chunk"+std::to_string(w)+"of"+std::to_string(chunk_num)+"_quant.tflite";  
        //matrix_path = data_dir+model_name+"_tflite/"+model_name+"_quant_edgetpu.tflite";  
          command = "python3 "+PWD+"/src/create_model.py --platform=m2 --model="+model_name+" --data_type="+data_type+"  --outfile_name="+matrix_path+" --in_size="+std::to_string(MM_BLK_COL)+" --out_size="+std::to_string(MM_BLK_ROW)+" --ramdisk="+std::to_string(ramdisk);
//        std::cout << "command: " << command << std::endl;
          int ret = system(command.c_str());
        }
      }
    }
    py_end   = clk::now();
  } //end using tf-generated tflite
  timing make_end = clk::now();
//std::cout << "start building models" << std::endl;
  // dispatcher: build model
  timing com_s, com_e;
  long long int model_ns = 0;
  float com_us = 0;
  for(int i = 0 ; i < ROW_BLK_CNT ; i++){
    for(int j = 0 ; j < INNER_BLK_CNT ; j++){
      for(int w = 0 ;w < chunk_num ; w++){
      //matrix_path = data_dir+model_name+"_tflite/"+model_name+"_["+std::to_string(i)+"]["+std::to_string(j)+"]_["+std::to_string(MM_BLK_ROW)+"x"+std::to_string(MM_BLK_COL)+"]["+std::to_string(A)+"x"+std::to_string(in)+"]_quant.tflite";  
      //command = "edgetpu_compiler "+matrix_path+" -o "+data_dir+model_name+"_tflite/"+" -s";
      //com_s = clk::now();
      //system(command.c_str());
      //com_e = clk::now();
      //com_us += std::chrono::duration_cast<std::chrono::nanoseconds>(com_e - com_s).count()/1000.0;
      //matrix_path = data_dir+model_name+"_tflite/"+model_name+"_["+std::to_string(i)+"]["+std::to_string(j)+"]_["+std::to_string(MM_BLK_ROW)+"x"+std::to_string(MM_BLK_COL)+"]["+std::to_string(A)+"x"+std::to_string(in)+"]_quant_edgetpu.tflite";  
        matrix_path = data_dir+model_name+"_tflite/"+model_name+"_["+std::to_string(i)+"]["+std::to_string(j)+"]_["+std::to_string(MM_BLK_ROW)+"x"+std::to_string(MM_BLK_COL)+"]["+std::to_string(A)+"x"+std::to_string(in)+"]_chunk"+std::to_string(w)+"of"+std::to_string(chunk_num)+"_quant.tflite";  
        model_id = (i*INNER_BLK_CNT + j)*chunk_num + w; //TODO: need re-design
        //std::cout << __func__ << ": matrix_path: " << matrix_path << std::endl;
        model_ns += build_model(matrix_path, model_id);// per task
        //itpr_ns  += build_interpreter(model_id%dev_cnt, model_id); // per task
      }
    }
  }
// start running 
  exe_start = clk::now();
#ifdef __aarch64__
  run_func(b, A, B, in, ROW_BLK_CNT, INNER_BLK_CNT, COL_BLK_CNT, partial_c, b_major);
#else
// TODO: loop through all tasks
  pthread_t tid[dev_cnt];
  struct arg_struct args[dev_cnt];
  for(int i = 0 ; i < dev_cnt ; i++){
    args[i].tpu_id = i;
    args[i].b      = b;
    args[i].A      = A;
    args[i].B      = B;
    args[i].in     = in;
    args[i].ROW_BLK_CNT = ROW_BLK_CNT;
    args[i].INNER_BLK_CNT = INNER_BLK_CNT;
    args[i].COL_BLK_CNT = COL_BLK_CNT;
    args[i].partial_c = partial_c;
    args[i].b_major = b_major;
    pthread_create(&tid[i], NULL, run_func, (void *)&args[i]);
  }
  for(int i = 0 ; i < dev_cnt ; i++){
    pthread_join(tid[i], NULL);
  }

//
//  struct OP_node *curr_node = (struct OP_node*) calloc(ROW_BLK_CNT*INNER_BLK_CNT*chunk_num, sizeof(struct OP_node)); 
//  int cnt = 0;
//  for(int i = 0 ; i < ROW_BLK_CNT ; i++){
//    for(int j = 0 ; j < INNER_BLK_CNT ; j++){
//      for(int w_chunk_idx = 0 ; w_chunk_idx < chunk_num ; w_chunk_idx++){
//        model_id = (i*INNER_BLK_CNT + j) * chunk_num + w_chunk_idx;
//        curr_node[cnt].op = model_name;
// std::cout << "enqueue model_name: " << model_name << ",i: " << i << ", j: " << j << ", w_chunk_idx: " << w_chunk_idx << std::endl;
//        curr_node[cnt].model_id = model_id;
//        curr_node[cnt].b_major = b_major;
//        curr_node[cnt].A  = A;
//        curr_node[cnt].B  = B;
//        curr_node[cnt].ROW_BLK_CNT   = ROW_BLK_CNT;
//        curr_node[cnt].INNER_BLK_CNT = INNER_BLK_CNT;
//        curr_node[cnt].COL_BLK_CNT   = COL_BLK_CNT;
//        curr_node[cnt].b = b;
//        curr_node[cnt].partial_c = partial_c;
//        curr_node[cnt].i = i;
//        curr_node[cnt].j = j;
//        curr_node[cnt].w_chunk_idx = w_chunk_idx;
//        fifo_push(SPMC_fifo, &curr_node[cnt]);
//        cnt++;
//        if(i == (ROW_BLK_CNT-1) && j == (INNER_BLK_CNT-1) && w_chunk_idx == (chunk_num-1)){ pthread_mutex_lock(&pmtx); done_enqueue = true; pthread_mutex_unlock(&pmtx); }
//      }
//    }
//  }
//  wait_queue_all_finished(); // wait until all items in queue are finished
//  free(curr_node);
// ============================
#endif
  exe_end = clk::now();
  
// save partial for debugging
//  save_partial(A, B, INNER_BLK_CNT);
  
// element-wise partial sum up 
  timing out_start = clk::now();
  int * trans_c;
  if(b_major == true){
    trans_c = (int*) malloc(A*B*sizeof(int));
  }else{
    trans_c = c;
  }
  int sum;
  for(int i = 0 ; i < A*B ; i++){
    sum = 0;
    for(int j = 0 ; j < INNER_BLK_CNT ; j++){
      sum += partial_c[j][i];
    }  
    trans_c[i] = sum;
  }
  timing out_end   = clk::now();
  printf("result C size: %d*%d = %d\n", A, B, A*B);
// transpose the result for matched checking 
  timing trans_start = clk::now();
  if(b_major == true){
    for(int i = 0 ; i < A ; i++){
      for(int j = 0 ; j < B ; j++){
        c[i*B+j] = trans_c[j*A+i];  
      }
    }
  }
  timing trans_end = clk::now();

// clean up
  for(int i = 0 ; i < INNER_BLK_CNT ; i++){
    free(partial_c[i]);
  }
  free(partial_c);
  free(data_array);
  if(b_major == true){
    free(trans_c);
  }
  timing total_end = clk::now();

  if(BREAKDOWN == 1){
//    std::cout << std::fixed;
    std::cout << std::setw(12);
    make_tflite_model_us = std::chrono::duration_cast<std::chrono::nanoseconds>(make_end  - make_start ).count()/1000.0;
    py_us                = std::chrono::duration_cast<std::chrono::nanoseconds>(py_end    - py_start   ).count()/1000.0;
    total_us             = std::chrono::duration_cast<std::chrono::nanoseconds>(total_end - total_start).count()/1000.0;
    set_us               = set_ns  /1000.0;
    new_us               = new_ns  /1000.0;
    model_us             = model_ns/1000.0;
    itpr_init_us         = itpr_init_ns /1000.0;
    itpr_us              = itpr_ns /1000.0;
    mem_us               = mem_ns  /1000.0;
    run_us               = run_ns  /1000.0;

    FILE* fp;    
    fp = fopen("./record.txt", "a");
    fprintf(fp, "%s, A:%d, in:%d, B:%d, run_us:%f, iter:%d\n", model_name.c_str(), A, in, B, run_us, ITER);
    fclose(fp);

    pop_us               = pop_ns  /1000.0;
    out_buf_us           = out_buf_ns/1000.0;
    exe_us               = std::chrono::duration_cast<std::chrono::nanoseconds>(exe_end   - exe_start  ).count()/1000.0;
    sum_us               = std::chrono::duration_cast<std::chrono::nanoseconds>(out_end   - out_start  ).count()/1000.0;
    trans_us             = std::chrono::duration_cast<std::chrono::nanoseconds>(trans_end - trans_start).count()/1000.0;
    blk_cnt              = ROW_BLK_CNT*INNER_BLK_CNT;
    printf("+----- gptpu_tf%d_%s timing breakdown ---------------------+\n", tf, model_name.c_str());
//    printf("|  itpr initialization : [%7.3f%%]\t%12.3f (us).\t| per itpr : (%7.3f%%)\t%12.3f (us).\n", (itpr_init_us/total_us)*100, itpr_init_us, ((itpr_init_us/total_us)/blk_cnt)*100, itpr_init_us/blk_cnt);  
    printf("+----- make model breakdown below ------------------------------+ average, each time may differ \n");
    if(tf == 1){
      printf("| python make model    : [%7.3f%%]\t%12.3f (us).\t| per py   : (%7.3f%%)\t%12.3f (us).\n", (py_us/total_us)*100, py_us, ((py_us/total_us)/blk_cnt)*100, py_us/blk_cnt);  
    }else{
      printf("| input data converting: [%7.3f%%]\t%12.3f (us).\t| per conv : (%7.3f%%)\t%12.3f (us).\n", (set_us/total_us)*100, set_us, ((set_us/total_us)/blk_cnt)*100, set_us/blk_cnt);  
      printf("| create tflite lite   : [%7.3f%%]\t%12.3f (us).\t| per creat: (%7.3f%%)\t%12.3f (us).\n", (new_us/total_us)*100, new_us, ((new_us/total_us)/blk_cnt)*100, new_us/blk_cnt);  
      printf("| compiler             : [%7.3f%%]\t%12.3f (us).\t| per creat: (%7.3f%%)\t%12.3f (us).\n", (com_us/total_us)*100, com_us, ((com_us/total_us)/blk_cnt)*100, com_us/blk_cnt);  
    }
    printf("+----- run detail below ----------------------------------------+\n");
    printf("| build model          : [%7.3f%%]\t%12.3f (us).\t| per build: (%7.3f%%)\t%12.3f (us).\n", (model_us/total_us)*100, model_us, ((model_us/total_us)/blk_cnt)*100, model_us/blk_cnt);  
    printf("| build itpr           : [%7.3f%%]\t%12.3f (us).\t| per itpr : (%7.3f%%)\t%12.3f (us).\n", (itpr_us/total_us)*100, itpr_us, ((itpr_us/total_us)/blk_cnt)*100, itpr_us/blk_cnt);  
    printf("| transfer input       : [%7.3f%%]\t%12.3f (us).\t| per trans: (%7.3f%%)\t%12.3f (us).\n", (mem_us/total_us)*100, mem_us, ((mem_us/total_us)/blk_cnt)*100, mem_us/blk_cnt);  
    printf("| run (invoke)         : [%7.3f%%]\t%12.3f (us).\t| per run  : (%7.3f%%)\t%12.3f (us).\n", (run_us/total_us)*100, run_us, ((run_us/total_us)/blk_cnt)*100, run_us/blk_cnt);  
    printf("| populate out buffer  : [%7.3f%%]\t%12.3f (us).\t| per pop  : (%7.3f%%)\t%12.3f (us).\n", (pop_us/total_us)*100, pop_us, ((pop_us/total_us)/blk_cnt)*100, pop_us/blk_cnt);  
    printf("| run func total       : (need redesign later)\t\t\t|          : (%7.3f%%)\t%12.3f (us).\n", (exe_us/total_us)*100, exe_us);  
    printf("+---------------------------------------------------------------+\n");
    printf("| sum up partial map   : [%7.3f%%]\t%12.3f (us).\t|\n", (sum_us/total_us)*100, sum_us);  
    printf("| matrix transpose     : [%7.3f%%]\t%12.3f (us).\t|\n", (trans_us/total_us)*100, trans_us);  
    printf("+---------------------------------------------------------------+\n");
    printf("| total                : [%7.3f%%]\t%12.3f (us).\t| (some impl. related overhead ignored for now)\n", (total_us/total_us)*100, total_us);  
    printf("+---------------------------------------------------------------+\n");
    long long int ns     = run_ns;
    long long int term1=0, term2=0;
    if(model_name == mm_model){
      term1 = MM_BLK_ROW * MM_BLK_COL * (2 * MM_BLK_COL - 1);
      term2 = ROW_BLK_CNT * INNER_BLK_CNT * (B / MM_BLK_COL);
    }else if(model_name == mv_model){
      term1 = MM_BLK_ROW * 1 * (2 * MM_BLK_COL - 1);
      term2 = ROW_BLK_CNT * INNER_BLK_CNT * 1;
    }
    long long int op_cnt = term1 * term2;  
    printf("| GOPS: %10.4f, op_cnt: %13lld, us: %12.3f\t|\n", (float)op_cnt/ns, op_cnt, run_us);   
    printf("+---------------------------------------------------------------+\n");
    std::cout << "|malloc partial       : " << std::chrono::duration_cast<std::chrono::nanoseconds>(malloc_e - malloc_s).count()/1000.0 << "\t(us).\t\t|" << std::endl; 
  }

  return 0;
}


void *mm2conv_run(void *arguments){
  struct arg_mm2conv_struct *args = (struct arg_mm2conv_struct *)arguments;
  int tid         = args->tid;
  int** a_feed    = args->a;
  int** partial_c = args->partial_c;
  int A           = args->A;
  int B           = args->B;
  int C           = args->C;
  int blk_A       = args->blk_A; 
  int blk_B       = args->blk_B; 
  int blk_C       = args->blk_C; 
  int ROW_BLK_CNT = args->ROW_BLK_CNT;
  int INN_BLK_CNT = args->INN_BLK_CNT;
  int COL_BLK_CNT = args->COL_BLK_CNT;
  float SCALE     = args->SCALE;
  long long int mem1_ns = 0, run1_ns = 0, pop1_ns = 0;
  int model_id = 0;
  int chunk_num = CHAR_BIT/get_chunk_size();
  int cnt = 0;
  for(int j = 0 ; j < INN_BLK_CNT ; j++){ 
    for(int k = 0 ; k < COL_BLK_CNT ; k++){ // outer loops can be in parallel for multiple devices to run
      for(int w_chunk_idx = 0 ; w_chunk_idx < chunk_num ; w_chunk_idx++){ 
        model_id = j*COL_BLK_CNT*chunk_num + k*chunk_num + w_chunk_idx;
        build_interpreter(/*tpu_id*/(model_id)%dev_cnt, model_id);
        if(model_id%dev_cnt == tid){
          for(int i = 0 ; i < ROW_BLK_CNT ; i++){ // running block execution
            for(int in_chunk_idx = 0 ; in_chunk_idx < chunk_num ; in_chunk_idx++){ //stride quantization overhead
              mtx.lock();
              mem1_ns += populate_input_chunking(a_feed[i*INN_BLK_CNT*COL_BLK_CNT + j*COL_BLK_CNT + k], blk_A*blk_C/*size*/, model_id, in_chunk_idx, data_type);
              run1_ns += invoke_model(model_id, ITER);
              pop1_ns += populate_output_chunking(partial_c[j], A, C, blk_A, blk_C, i, k, model_id, in_chunk_idx, w_chunk_idx, SCALE);  
//              printf("mem_ns: %12lld, run_ns: %12lld, pop_ns: %12lld, (i,j,k=(%d,%d,%d)), model_id: %d, tid: %d, in_chunk_idx: %d, w_chunk_idx = %d\n", mem1_ns, run1_ns, pop1_ns, i, j, k, model_id, tid, in_chunk_idx, w_chunk_idx);
              cnt++;
              mtx.unlock();
            } 
          }
        }// end if
      }
    }
  }
  double mem_us, run_us, pop_us;
  mem_us = mem1_ns/1000.0;
  run_us = run1_ns/1000.0;
  pop_us = pop1_ns/1000.0;
  printf("tid:%d,  (%12.3f, %12.3f, %12.3f) (us).\n", tid, mem_us, run_us, pop_us);  
  printf("cnt = %d\n", cnt);
    FILE* fp;    
    fp = fopen("./record.txt", "a");
    fprintf(fp, "A:%d, B:%d, C:%d, run_us:%f, iter:%d\n", A, B, C, run_us, ITER);
    fclose(fp);
}
// min-plus matrix multiplication
int gptpu_mpmm(const std::string& model_path, int* a, int* b, int* c, int A, int B, bool b_major){
  // for current design, only square matrix are allowed
  timing total_s, total_e, init_s, init_e, save_weight_s, set_data_s, set_data_e, save_weight_e, py_s, py_e, convert_s, convert_e, create_s, create_e, build_m_s, build_m_e, build_itpr_s, build_itpr_e, malloc_s, malloc_e, copy_s, copy_e, mm2conv_s, mm2conv_e, run_s, run_e, sum_s, sum_e, free_s, free_e;
  double set_data_ns = 0, create_ns = 0, model_ns = 0, itpr_ns = 0;
  pthread_t tid[dev_cnt];
  struct arg_mm2conv_struct args[dev_cnt];
  for(int i = 0 ; i < dev_cnt ; i++){
    build_model(model_path, i);
    build_interpreter(i, i);
  }  

  for(int tid_idx = 0 ; tid_idx < dev_cnt ; tid_idx++){
    args[tid_idx].tid         = tid_idx;
//    args[tid_idx].a           = a_feed;
//    args[tid_idx].partial_c   = partial_c;
    args[tid_idx].A           = A;
    args[tid_idx].B           = B;
//    args[tid_idx].C           = C;
//    args[tid_idx].blk_A       = blk_A;
//    args[tid_idx].blk_B       = blk_B;
//    args[tid_idx].blk_C       = blk_C;
//    args[tid_idx].ROW_BLK_CNT = ROW_BLK_CNT;
//    args[tid_idx].INN_BLK_CNT = INN_BLK_CNT;
//    args[tid_idx].COL_BLK_CNT = COL_BLK_CNT;
//    args[tid_idx].SCALE       = IN_SCALE;
    pthread_create(&tid[tid_idx], NULL, mpmm_run, (void *)&args[tid_idx]);
  }
  for(int i = 0 ; i < dev_cnt ; i++){
    pthread_join(tid[i], NULL);
  }
}
//  total_s = clk::now();
//  std::string matrix_path;
//  std::string command;
//  int IN_W, IN_H, P_W, P_H;
//  std::string prefix_model_name, model_name, weight_file_name;
//  // determine convolution shape for mm or mv based on input A, B, C
//  int blk_A;
//  int blk_B;
//  int blk_C;
//// end of shape =====================================================================
//  std::string mpmm_template_prefix_name = "maxpool_temp_"+itoa(IN_W)+"x"+itoa(IN_H)+"x"+itoa(P_W)+"x"+itoa(P_H);
//  int ROW_BLK_CNT = sub_cnt(A, blk_A); //  (A / blk_A) + ((A % blk_A != 0)?1:0);  
//  int INN_BLK_CNT = sub_cnt(B, blk_B); //  (B / blk_B) + ((B % blk_B != 0)?1:0);  
//  int COL_BLK_CNT = sub_cnt(C, blk_C); //  (C / blk_C) + ((C % blk_C != 0)?1:0);
//  int ROW_BLK_REM = A % blk_A;
//  int INN_BLK_REM = B % blk_B;
//  int COL_BLK_REM = C % blk_C;
//  //std::cout << "blk_A: " << blk_A << ", blk_B: " << blk_B << ",blk_C: " << blk_C << ", ROW_BLK_CNT: " << ROW_BLK_CNT << ", INN_BLK_CNT: " << INN_BLK_CNT << ", COL_BLK_CNT: " << COL_BLK_CNT << ", ROW_BLK_REM: " << ROW_BLK_REM << ", INN_BLK_REM: " << INN_BLK_REM << ", COL_BLK_REM: " << COL_BLK_REM << std::endl;
//  int model_id = 0;
//  init_s = clk::now();
//  init_e = clk::now();
//  //create matrix weight from array 'b' (input from array 'a' later)
//  char* data_array = (char*)malloc(blk_B*blk_C*sizeof(char));
//  bool template_created = false; //avoid re-creating same shape of mm during blocking algorithm
//
//// ===== get default scaling factor =====
//  float IN_SCALE = 1;//get_auto_scale_factor_mm(a, b, A, B, C);
//  set_scale(255); //for binary dechypher
////  set_scale(1);
//  //std::cout << "SCALE: " << SCALE << ", IN_SCALE: " << IN_SCALE << std::endl;
//// ===== end scaling ========
//  for(int i = 0 ; i < INN_BLK_CNT ; i++){
//    for(int j = 0 ; j< COL_BLK_CNT ; j++){
//        model_id = i*COL_BLK_CNT + j;
//        // TODO: template is constrianted with shape size 
//        prefix_model_name = "maxpool_model_quant_"+itoa(IN_W)+"x"+itoa(IN_H)+"x"+itoa(P_W)+"x"+itoa(P_H)+"_outof_"+itoa(INN_BLK_CNT)+"x"+itoa(COL_BLK_CNT);
//        std::string template_name = mpmm_template_prefix_name + "_" + data_type + ".tflite";
//        template_path = temp_dir + template_name;
//        matrix_path   = data_dir+"maxpool_model_tflite/" + prefix_model_name + "_edgetpu.tflite";  
//        std::cout << "template_path: " << template_path << std::endl;
//        std::cout << "matrix_path  : " << matrix_path << std::endl;
//        if(0 || (file_exist(template_path) == false /*|| !(blk_B == blk_C && blk_C == 1024)*/) && template_created == false){
//          template_created = true; // if really need to create template, only once for all same shape blocks
//          //save weight in file(s) for python to create model
//          std::cout << "template_path: " << template_path << " not exist, creating it..." << std::endl;
//          //create xxx_quant.tflite
//          //model_name     = prefix_model_name + "_edgetpu.tflite";
//          //matrix_path    = data_dir+"conv_model_tflite/conv_model_quant_edgetpu.tflite";  
////TODO; the scale is decided by max value
//          std::cout << "scale is: " << SCALE << std::endl;
//          command = "python3 "+PWD+"/src/create_model.py"+" --model=maxpool_model"+" --in_w_name="+weight_file_name+" --data_type="+data_type+" --out_scale="+itoa(SCALE)+" --outfile_name="+matrix_path+" --IN_W="+itoa(IN_W)+" --IN_H="+itoa(IN_H)+" --F_W="+itoa(P_W)+" --F_H="+itoa(P_H);
//          py_s = clk::now();
//          std::cout << "command: " << command.c_str() << std::endl;
//          int ret = system(command.c_str());
//          py_e = clk::now();
//          make_maxpool_temp(matrix_path, local_temp_dir+template_name, blk_A, blk_B, blk_C);
//          command = "sudo cp "+local_temp_dir+template_name+" "+template_path;
//          ret = system(command.c_str());
//        }
//        // set data array
//        set_data_s = clk::now();
//        set_mm256conv_array(b, b_major, data_array, B, C, i, j, INN_BLK_CNT, COL_BLK_CNT, blk_B, blk_C, INN_BLK_REM, COL_BLK_REM, chunk_num, exact_mode);
//        set_data_e = clk::now();
//        set_data_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(set_data_e - set_data_s).count();
//        // create model
//        create_s = clk::now();
//        create_mm2conv_tflite(template_path, matrix_path, data_array, blk_A*chunk_num, blk_B, blk_C*chunk_num, SCALE, chunk_num);
//        create_e = clk::now();
//        create_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(create_e - create_s).count();
//        //build model
//        //template_path  = data_dir+"conv_model_tflite/" + prefix_model_name + "_edgetpu.tflite";  
//        //matrix_path    = data_dir+"conv_model_tflite/conv_model_quant_edgetpu.tflite";  
//        build_m_s = clk::now();
//std::cout << "build model, the matrix_path: " << matrix_path << std::endl;
//        build_model(matrix_path, model_id);
//        build_m_e = clk::now();
//        model_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(build_m_e - build_m_s).count();
//        //build interpreter
//        build_itpr_s = clk::now();
//        //build_interpreter(/*tpu_id*/(model_id)%dev_cnt, model_id);
//        build_itpr_e = clk::now();
//        itpr_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(build_itpr_e - build_itpr_s).count();
//    }
//  }
//  long long int mem1_ns, run1_ns, pop1_ns;
//  int output_size = 0;
//  malloc_s = clk::now(); 
//  partial_c = (int**)malloc(INN_BLK_CNT*sizeof(int*));
//  for(int i = 0 ; i < INN_BLK_CNT; i++){
//    partial_c[i] = (int*)calloc(A*C*chunk_num*chunk_num, sizeof(int));
//  }
//  int*** blk_exact_c = (int***) malloc(INN_BLK_CNT*sizeof(int**));
//  for(int i = 0 ; i < INN_BLK_CNT ; i++){
//     blk_exact_c[i] = (int**) malloc(ROW_BLK_CNT*COL_BLK_CNT*sizeof(int*));
//     for(int j = 0 ; j < ROW_BLK_CNT*COL_BLK_CNT ; j++){
//       blk_exact_c[i][j] = (int*) calloc(blk_A*blk_C*chunk_num*chunk_num, sizeof(int));
//     }
//  }
//  malloc_e = clk::now(); 
////TODO: need relayout optimization
//  int** a_blk  = (int**)malloc(ROW_BLK_CNT*INN_BLK_CNT*COL_BLK_CNT*sizeof(int*));
//  int** a_feed = (int**)malloc(ROW_BLK_CNT*INN_BLK_CNT*COL_BLK_CNT*sizeof(int*));
//  for(int i = 0 ; i < ROW_BLK_CNT*INN_BLK_CNT*COL_BLK_CNT; i++){
//    a_blk[i]  = (int*)calloc(blk_A*blk_C, sizeof(int)); // make sure non-mapped elements are zero by default
//    a_feed[i] = (int*)malloc(blk_A*blk_C*chunk_num*sizeof(int));
//  }  
//  // prepare input matrix data a, including mm2conv
//  double copy_us    = 0;
//  double mm2conv_us = 0;
//  for(int j = 0 ; j < INN_BLK_CNT ; j++){
//    for(int k = 0 ; k < COL_BLK_CNT ; k++){
//      for(int i = 0 ; i < ROW_BLK_CNT ; i++){
//        int idx = i*INN_BLK_CNT*COL_BLK_CNT + j*COL_BLK_CNT + k;
//        copy_s = clk::now();
//        // TODO: modify for supporting random shape copying
//        for(int curr_i = 0 ; curr_i < ((i < ROW_BLK_CNT-1 || ROW_BLK_REM == 0)?blk_A:ROW_BLK_REM) ; curr_i++){
//          memcpy(a_blk[idx]+curr_i*blk_B, a+(i*blk_A+curr_i)*B+(j*blk_B), ((j < INN_BLK_CNT-1 || INN_BLK_REM == 0)?(blk_B):(INN_BLK_REM))*sizeof(int));
//        }
//        copy_e = clk::now();
//        mm2conv_s = clk::now();
//        data_mm256conv(a_blk[idx], a_feed[idx], blk_A, blk_C, IN_W, IN_H, F_W, F_H, IN_C, chunk_num);
//        mm2conv_e = clk::now();
//        copy_us    += std::chrono::duration_cast<std::chrono::nanoseconds>(copy_e - copy_s).count()/1000.0;
//        mm2conv_us += std::chrono::duration_cast<std::chrono::nanoseconds>(mm2conv_e - mm2conv_s).count()/1000.0;
//      }
//    }
//  }
//  // start pthread creation and actual exeution
//  timing exe_s = clk::now();
//  double mem_us, run_us, pop_us;
//#ifdef __aarch64__ //  the run section for coral dev board
//  mem1_ns = 0, run1_ns = 0, pop1_ns = 0;
//  int tid = 0;
//  model_id = 0;
//  chunk_num = 16/*CHAR_BIT*//get_chunk_size();
//  for(int j = 0 ; j < INN_BLK_CNT ; j++){ 
//    for(int k = 0 ; k < COL_BLK_CNT ; k++){ // outer loops can be in parallel for multiple devices to run
//      for(int w_chunk_idx = 0 ; w_chunk_idx < chunk_num ; w_chunk_idx++){ 
//        model_id = j*COL_BLK_CNT*chunk_num + k*chunk_num + w_chunk_idx;
//        if(model_id%dev_cnt == tid){
//          for(int i = 0 ; i < ROW_BLK_CNT ; i++){ // running block execution
//            for(int in_chunk_idx = 0 ; in_chunk_idx < chunk_num ; in_chunk_idx++){ //stride quantization overhead
//              mem1_ns += populate_input_chunking(a_feed[i*INN_BLK_CNT*COL_BLK_CNT + j*COL_BLK_CNT + k], blk_A*blk_C/*size*/, model_id, in_chunk_idx, data_type);
//              run1_ns += invoke_model(model_id);
//              pop1_ns += populate_output_chunking(partial_c[j], A, C, blk_A, blk_C, i, k, model_id, in_chunk_idx, w_chunk_idx);  
//              printf("mem_ns: %12lld, run_ns: %12lld, pop_ns: %12lld, (i,j,k=(%d,%d,%d)), model_id: %d, tid: %d, in_chunk_idx: %d\n", mem1_ns, run1_ns, pop1_ns, i, j, k, model_id, tid, in_chunk_idx);
//            } 
//          }
//        }// end if
//      }
//    }
//  }
//  mem_us = mem1_ns/1000.0;
//  run_us = run1_ns/1000.0;
//  pop_us = pop1_ns/1000.0;
//  printf("(%12.3f, %12.3f, %12.3f) (us).\n", mem_us, run_us, pop_us);  
//#else // the run section for host, see pthread function for detail
//  struct OP_node *curr_node = (struct OP_node*) calloc(INN_BLK_CNT*COL_BLK_CNT*chunk_num, sizeof(struct OP_node)); 
//  int cnt = 0;
//  for(int j = 0 ; j < INN_BLK_CNT ; j++){
//    for(int k = 0 ; k < COL_BLK_CNT ; k++){
//        //model_id = j*COL_BLK_CNT*chunk_num + k*chunk_num + w_chunk_idx;
//        model_id = j*COL_BLK_CNT + k;
//        curr_node[cnt].op            = mm_model;
//        curr_node[cnt].model_id      = model_id;
//        curr_node[cnt].a_feed        = a_feed;
//        curr_node[cnt].partial_c     = partial_c;
//        curr_node[cnt].A             = A;
//        curr_node[cnt].B             = B;
//        curr_node[cnt].C             = C;
//        curr_node[cnt].j             = j;
//        curr_node[cnt].k             = k;
//        curr_node[cnt].blk_A         = blk_A;
//        curr_node[cnt].blk_B         = blk_B;
//        curr_node[cnt].blk_C         = blk_C;
//        curr_node[cnt].ROW_BLK_CNT   = ROW_BLK_CNT;
//        curr_node[cnt].INNER_BLK_CNT = INN_BLK_CNT;
//        curr_node[cnt].COL_BLK_CNT   = COL_BLK_CNT;
//        curr_node[cnt].partial_c     = blk_exact_c[j];//partial_c;
//std::cout << __func__ << ": enqueue, SCALE: " << SCALE << std::endl; 
//        curr_node[cnt].SCALE         = IN_SCALE;
//        curr_node[cnt].mm256         = true;
//        curr_node[cnt].chunk_num     = chunk_num;
//        fifo_push(SPMC_fifo, &curr_node[cnt]);
//        //std::cout << __func__ << ": model_id: " << model_id << ", A: " << A << "    , B: " << B << ", C: " << C << ", j: " << j << ", k: " << ", w_chunk_idx: " << w_chunk_idx << ", blk_A: " << blk_A << ", blk_B: " << blk_B << ", blk_C: " << blk_C << ", ROW_BLK_CNT: " << ROW_BLK_CNT << ", INN_BLK_CNT: " << INN_BLK_CNT << ", COL_BLK_CNT: " << COL_BLK_CNT << ", SCALE: " << SCALE << std::endl;
//        cnt++;
//        if(j == (INN_BLK_CNT-1) && k == (COL_BLK_CNT-1)){ pthread_mutex_lock(&pmtx); done_enqueue = true; pthread_mutex_unlock(&pmtx); }
//    }
//  }
//  wait_queue_all_finished();
//  free(curr_node);
//
//#endif
//  timing exe_e = clk::now();
//  // summation stage step1 : shrink the (A*chunk_num)*(C*chunk_num) block into A*C
//  
////  for(int i  = 0 ; i < 10/*A*chunk_num*/ ; i++){
////    for(int j = 0 ; j < 10/*C*chunk_num*/ ; j++){
////      printf("%d ", partial_c[0][i*(C*chunk_num)+j]);
////  //     if(partial_c[0][i*(C*chunk_num)+j] != 0 && partial_c[0][i*(C*chunk_num)+j] != 1)
////  //       printf("(%4d, %4d) %4d\n", i, j, partial_c[0][i*(C*chunk_num)+j]);
////  //     if(partial_c[0][i*(C*chunk_num)+j] == 1)
////  //       partial_c[0][i*(C*chunk_num)+j] = 0;
////     }
////     printf("\n");
////  }
//  for(int idx = 0 ; idx < INN_BLK_CNT ; idx++){
//    for(int i = 0 ; i < ROW_BLK_CNT ; i++){
//      for(int j = 0 ; j < COL_BLK_CNT ; j++){
//        for(int k = 0 ; k < blk_A*blk_C*chunk_num*chunk_num ; k++){
//          int chunk_r = k/(blk_A*blk_C*chunk_num);
//          int chunk_c = (k%(blk_C*chunk_num))/blk_C;
//          int inblk_r = (k/(blk_C*chunk_num))%blk_A;
//          int inblk_c = (k%(blk_C*chunk_num))%blk_C;
//          int offset  = chunk_r*(A*C*chunk_num) + chunk_c * C + i/*blk_idx_i*/*(blk_A*C*chunk_num) + j/*blk_idx_k*/*blk_C + inblk_r*(C*chunk_num) + inblk_c;
//          partial_c[idx][offset] = blk_exact_c[idx][i*COL_BLK_CNT+j][k] << (chunk_r + chunk_c);
//        }
//      }
//    }
//  }
//
//  int sum = 0, offset;
//  int pi=0,pj=0;
//  std::cout << "print all result chunks for c[" << pi << "][" << pj <<  "]" << std::endl;
//  for(int i = 0 ; i < A ; i++){
//    for(int j = 0 ; j < C ; j++){
//      sum = 0;
//      for(int in_idx = 0 ; in_idx < INN_BLK_CNT ; in_idx++){
//        if(i == pi && j == pj)std::cout << "INN_BLK_idx: " << in_idx << std::endl;
//        for(int idx_r = 0 ; idx_r < chunk_num ; idx_r++){
//          for(int idx_c = 0 ; idx_c < chunk_num ; idx_c++){
//            offset = idx_r*(A*C*chunk_num) + idx_c*(C) + i*(C*chunk_num)+j;
//            if(i == pi && j == pj){
//              printf("%d ", partial_c[in_idx][offset]);
//            }
//            sum += partial_c[in_idx][offset];
//          }
//         if(i == pi && j == pj)printf("\n");
//        }
//      }
//      if(i == pi && j == pj)printf("sum: %d\n", sum);
//      c[i*C+j] = sum;
//    } 
//  }
////  for(int i = 0 ; i < A ; i++){
////    for(int j = 0 ; j < C ; j++){
////      for(int in_idx = 0 ; in_idx < INN_BLK_CNT ; in_idx++){
////        for(int idx_r = 0 ; idx_r < chunk_num ; idx_r++){
////          for(int idx_c = 0 ; idx_c < chunk_num ; idx_c++){
////            offset = idx_r*(A*C*chunk_num) + idx_c*(C) + i*(C*chunk_num)+j;
////            if(partial_c[in_idx][offset] != 0){
////              std::cout << "partial_c[" << in_idx << "][" << offset << "] is " << partial_c[in_idx][offset] << ", i: " << i << ", j: " << j << ", in_idx: " << in_idx << ", idx_r: " << idx_r << ", idx_c: " << idx_c << std::endl;
////            }
////          }
////        }
////      }
////    }
////  }  
//// partial result summation
////  sum_s = clk::now();
////  sum = 0;
////  for(int i = 0; i < A*C ; i++){
////    sum = 0;
////    for(int j =  0; j< INN_BLK_CNT ; j++){  sum += partial_c[j][i]; }
////    c[i] = sum;
////  }
////  sum_e = clk::now();
//  // free up buffer
//  free_s = clk::now();
//  for(int i = 0 ; i < INN_BLK_CNT; i++){ free(partial_c[i]); }
//  free(partial_c);
//  for(int i = 0 ; i < ROW_BLK_CNT*INN_BLK_CNT*COL_BLK_CNT; i++){
//    free(a_blk[i]);
//    free(a_feed[i]);
//  }
//  free(a_blk);
//  free(a_feed);
//  free(data_array);
//  free_e = clk::now();
//  total_e = clk::now();
//  double save_weight_us = std::chrono::duration_cast<std::chrono::nanoseconds>(save_weight_e - save_weight_s).count()/1000.0;
//  double init_us        = std::chrono::duration_cast<std::chrono::nanoseconds>(init_e - init_s).count()/1000.0;
//  double py_us          = std::chrono::duration_cast<std::chrono::nanoseconds>(py_e - py_s).count()/1000.0;
//  double convert_us     = std::chrono::duration_cast<std::chrono::nanoseconds>(convert_e - convert_s).count()/1000.0;
//  double set_data_us    = set_data_ns/1000.0;
//  double create_us      = create_ns/1000.0;
//  double build_m_us     = model_ns/1000.0;
//  double build_itpr_us  = itpr_ns/1000.0;
////  double copy_us        = std::chrono::duration_cast<std::chrono::nanoseconds>(copy_e - copy_s).count()/1000.0;
////  double mm2conv_us     = std::chrono::duration_cast<std::chrono::nanoseconds>(mm2conv_e - mm2conv_s).count()/1000.0;
//  double exe_us         = std::chrono::duration_cast<std::chrono::nanoseconds>(exe_e - exe_s).count()/1000.0;
//         mem_us         = mem1_ns/(1000.0);
//         run_us         = run1_ns/(1000.0*ITER);
//         pop_us         = pop1_ns/(1000.0);
//  double sum_us         = std::chrono::duration_cast<std::chrono::nanoseconds>(sum_e - sum_s).count()/1000.0;
//  double free_us        = std::chrono::duration_cast<std::chrono::nanoseconds>(free_e - free_s).count()/1000.0;
//  double total_us       = std::chrono::duration_cast<std::chrono::nanoseconds>(total_e - total_s).count()/1000.0;
//  printf("| save weights : [%7.3f%%]\t%12.3f (us).|\n", (save_weight_us/total_us)*100, save_weight_us);  
//  printf("| itpr init    : [%7.3f%%]\t%12.3f (us).|\n", (init_us/total_us)*100, init_us);  
//
//  printf("| py create m  : [%7.3f%%]\t%12.3f (us).|\n", (py_us/total_us)*100, py_us);  
//  printf("| convert model: [%7.3f%%]\t%12.3f (us).|\n", (convert_us/total_us)*100, convert_us);  
//
//  
//  printf("| set data     : [%7.3f%%]\t%12.3f (us).|\n", (set_data_us/total_us)*100, set_data_us);  
//  printf("| create binary: [%7.3f%%]\t%12.3f (us).|\n", (create_us/total_us)*100, create_us);  
//
//  printf("| build model  : [%7.3f%%]\t%12.3f (us).|\n", (build_m_us/total_us)*100, build_m_us);  
//  printf("| build itpr   : [%7.3f%%]\t%12.3f (us).|\n", (build_itpr_us/total_us)*100, build_itpr_us);  
//  printf("| copy input   : [%7.3f%%]\t%12.3f (us).|\n", (copy_us/total_us)*100, copy_us);  
//  printf("| mm2conv      : [%7.3f%%]\t%12.3f (us).|\n", (mm2conv_us/total_us)*100, mm2conv_us);  
//  printf("| mem          : [%7.3f%%]\t%12.3f (us).|\n", (mem_us/total_us)*100, mem_us);  
//  printf("| avg run      : [%7.3f%%]\t%12.3f (us).|\n", (run_us/total_us)*100, run_us);  
//  printf("| exe in total : [%7.3f%%]\t%12.3f (us).|\n", (exe_us/total_us)*100, exe_us);  
//  printf("| pop          : [%7.3f%%]\t%12.3f (us).|\n", (pop_us/total_us)*100, pop_us);  
//  printf("| sum          : [%7.3f%%]\t%12.3f (us).|\n", (sum_us/total_us)*100, sum_us);  
//  printf("| free         : [%7.3f%%]\t%12.3f (us).|\n", (free_us/total_us)*100, free_us);  
//  long long int term1=0, term2=0;
//  term1 = 2*B-1;
//  term2 = A*C;
//  long long int op_cnt = term1 * term2;  
//  printf("| invoke  GOPS: %10.4f, op_cnt: %13lld, us: %12.3f\t|\n", (float)op_cnt/(exe_us*1000), op_cnt, exe_us);   
//  printf("| end2end GOPS: %10.4f, op_cnt: %13lld, us: %12.3f\t|\n", (float)op_cnt/(total_us*1000), op_cnt, total_us);   
//  return 0;
//}
int gptpu_256mm(int* a, int* b, int* c, int A, int B, int C, bool b_major){
  if (exact_mode != 1) {std::cout << "warning: " << __func__ << " is not in exact mode." << std::endl; exit(0);}
  // for current design, only square matrix are allowed
  timing total_s, total_e, init_s, init_e, save_weight_s, set_data_s, set_data_e, save_weight_e, py_s, py_e, convert_s, convert_e, create_s, create_e, build_m_s, build_m_e, build_itpr_s, build_itpr_e, malloc_s, malloc_e, copy_s, copy_e, mm2conv_s, mm2conv_e, run_s, run_e, sum_s, sum_e, free_s, free_e;
  double set_data_ns = 0, create_ns = 0, model_ns = 0, itpr_ns = 0;
  total_s = clk::now();
  std::string matrix_path;
  std::string command;
  int IN_W, IN_H, F_W, F_H, IN_C, S_W, S_H, OUT_C;
  std::string prefix_model_name, model_name, weight_file_name;
  // determine convolution shape for mm or mv based on input A, B, C
  int blk_A;
  int blk_B;
  int blk_C;
// the dedicated shape for exact mode (16b->32b) mm operation =======================
  if(chunk_num == 16){
    blk_A = 256; blk_B = 256; blk_C = 256; IN_W = 8192; IN_H = 8; 
    IN_C  = 16; OUT_C = 4096; F_W = S_W = 8; F_H = S_H = 2;
  }else if(chunk_num == 8){
    blk_A = 256; blk_B = 256; blk_C = 256; IN_W = 512; IN_H = 128;
    IN_C = 8; OUT_C = 2048; F_W = S_W = 4; F_H = S_H = 8;
  }else if(chunk_num == 4){
    blk_A = 256; blk_B = 256; blk_C = 256; IN_W = 128; IN_H = 256;
    IN_C = 8; OUT_C = 1024; F_W = S_W = 8; F_H = S_H = 4;
  }else if(chunk_num == 2){
    blk_A = 256; blk_B = 256; blk_C = 256; IN_W = 256; IN_H = 128;
    IN_C = 4; OUT_C = 512; F_W = S_W = 8; F_H = S_H = 8;
  }else if(chunk_num == 1){
    blk_A = 256; blk_B = 256; blk_C = 256; IN_W = 128; IN_H = 128;
    IN_C = 4; OUT_C = 256; F_W = S_W = 32; F_H = S_H = 2;
  }else{
    std::cout << __func__ << ": undefined exact mode chunk_num: " << chunk_num << std::endl; exit(0);
  }
// end of shape =====================================================================
  std::string mm2conv_template_prefix_name = "conv_temp_"+itoa(IN_W)+"x"+itoa(IN_H)+"x"+itoa(IN_C)+"x"+itoa(F_W)+"x"+itoa(F_H)+"x"+itoa(S_W)+"x"+itoa(S_H);
  int ROW_BLK_CNT = sub_cnt(A, blk_A); //  (A / blk_A) + ((A % blk_A != 0)?1:0);  
  int INN_BLK_CNT = sub_cnt(B, blk_B); //  (B / blk_B) + ((B % blk_B != 0)?1:0);  
  int COL_BLK_CNT = sub_cnt(C, blk_C); //  (C / blk_C) + ((C % blk_C != 0)?1:0);
  int ROW_BLK_REM = A % blk_A;
  int INN_BLK_REM = B % blk_B;
  int COL_BLK_REM = C % blk_C;
  //std::cout << "blk_A: " << blk_A << ", blk_B: " << blk_B << ",blk_C: " << blk_C << ", ROW_BLK_CNT: " << ROW_BLK_CNT << ", INN_BLK_CNT: " << INN_BLK_CNT << ", COL_BLK_CNT: " << COL_BLK_CNT << ", ROW_BLK_REM: " << ROW_BLK_REM << ", INN_BLK_REM: " << INN_BLK_REM << ", COL_BLK_REM: " << COL_BLK_REM << std::endl;
  int model_id = 0;
  init_s = clk::now();
  init_e = clk::now();
  //create matrix weight from array 'b' (input from array 'a' later)
  char* data_array = (char*)malloc(blk_B*blk_C*chunk_num*sizeof(char));
  bool template_created = false; //avoid re-creating same shape of mm during blocking algorithm

// ===== get default scaling factor =====
  float IN_SCALE = 1;//get_auto_scale_factor_mm(a, b, A, B, C);
  set_scale(255); //for binary dechypher
  //set_scale(1); // for python
  //std::cout << "SCALE: " << SCALE << ", IN_SCALE: " << IN_SCALE << std::endl;
// ===== end scaling ========
  for(int i = 0 ; i < INN_BLK_CNT ; i++){
    for(int j = 0 ; j< COL_BLK_CNT ; j++){
        model_id = i*COL_BLK_CNT + j;
        // TODO: template is constrianted with shape size 
        prefix_model_name = "conv_model_quant_"+itoa(IN_W)+"x"+itoa(IN_H)+"x"+itoa(IN_C)+"x"+itoa(F_W)+"x"+itoa(F_H)+"x"+itoa(S_W)+"x"+itoa(S_H)+"_256exact_"+itoa(i)+"x"+itoa(j)+"_outof_"+itoa(INN_BLK_CNT)+"x"+itoa(COL_BLK_CNT);
        std::string template_name = mm2conv_template_prefix_name + "_" + data_type + ".tflite";
        template_path = temp_dir + template_name;
        matrix_path   = data_dir+"conv_model_tflite/" + prefix_model_name + "_edgetpu.tflite";  
        //std::cout << "template_path: " << template_path << std::endl;
        //std::cout << "matrix_path  : " << matrix_path << std::endl;
        if( (file_exist(template_path) == false /*|| !(blk_B == blk_C && blk_C == 1024)*/) && template_created == false){
          template_created = true; // if really need to create template, only once for all same shape blocks
          //save weight in file(s) for python to create model
          std::cout << "template_path: " << template_path << " not exist, creating it..." << std::endl;
          weight_file_name = "./../mm256conv_weight_"+itoa(i)+"x"+itoa(j)+"_outof_"+itoa(INN_BLK_CNT)+"x"+itoa(COL_BLK_CNT)+".txt";
          save_weight_s = clk::now();
          mm256blk_save_weight(b, b_major, weight_file_name, B, C, i, j, INN_BLK_CNT, COL_BLK_CNT, blk_B, blk_C, INN_BLK_REM, COL_BLK_REM, chunk_num);
          save_weight_e = clk::now();
          //create xxx_quant.tflite
          //model_name     = prefix_model_name + "_edgetpu.tflite";
          //matrix_path    = data_dir+"conv_model_tflite/conv_model_quant_edgetpu.tflite";  
//TODO; the scale is decided by max value
          std::cout << "scale is: " << SCALE << std::endl;
          command = "python3 "+PWD+"/src/create_model.py"+" --model=conv_model"+" --in_w_name="+weight_file_name+" --data_type="+data_type+" --out_scale="+itoa(SCALE)+" --outfile_name="+matrix_path+" --IN_W="+itoa(IN_W)+" --IN_H="+itoa(IN_H)+" --IN_C="+itoa(IN_C)+" --F_W="+itoa(F_W)+" --F_H="+itoa(F_H)+" --S_W="+std::to_string(F_W)+" --S_H="+std::to_string(F_H)+" --OUT_C="+itoa(OUT_C)+" --mm256blk=1";
          py_s = clk::now();
          std::cout << "command: " << command.c_str() << std::endl;
          int ret = system(command.c_str());
          py_e = clk::now();
          //make_mm2conv_temp(matrix_path, local_temp_dir+template_name, blk_A*chunk_num, blk_B, blk_C*chunk_num);
          command = "sudo cp "+local_temp_dir+template_name+" "+template_path;
          ret = system(command.c_str());
        }
        // set data array
        set_data_s = clk::now();
        set_mm256conv_array(b, b_major, data_array, B, C, i, j, INN_BLK_CNT, COL_BLK_CNT, blk_B, blk_C, INN_BLK_REM, COL_BLK_REM, chunk_num, exact_mode);
        set_data_e = clk::now();
        set_data_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(set_data_e - set_data_s).count();
        // create model
        create_s = clk::now();
        create_mm2conv_tflite(template_path, flatbufs, /*matrix_path*/model_id, data_array, blk_A*chunk_num, blk_B, blk_C*chunk_num, SCALE, chunk_num);
        create_e = clk::now();
        create_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(create_e - create_s).count();
        //build model
        //template_path  = data_dir+"conv_model_tflite/" + prefix_model_name + "_edgetpu.tflite";  
        //matrix_path    = data_dir+"conv_model_tflite/conv_model_quant_edgetpu.tflite";  
        build_m_s = clk::now();
//std::cout << "build model, the matrix_path: " << matrix_path << std::endl;
        build_model(matrix_path, model_id);
        build_m_e = clk::now();
        model_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(build_m_e - build_m_s).count();
        //build interpreter
        build_itpr_s = clk::now();
        //build_interpreter(/*tpu_id*/(model_id)%dev_cnt, model_id);
        build_itpr_e = clk::now();
        itpr_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(build_itpr_e - build_itpr_s).count();
    }
  }
  long long int mem1_ns, run1_ns, pop1_ns;
  int output_size = 0;
  malloc_s = clk::now(); 
  partial_c = (int**)malloc(INN_BLK_CNT*sizeof(int*));
  for(int i = 0 ; i < INN_BLK_CNT; i++){
    partial_c[i] = (int*)calloc(A*C*chunk_num*chunk_num, sizeof(int));
  }
  int*** blk_exact_c = (int***) malloc(INN_BLK_CNT*sizeof(int**));
  for(int i = 0 ; i < INN_BLK_CNT ; i++){
     blk_exact_c[i] = (int**) malloc(ROW_BLK_CNT*COL_BLK_CNT*sizeof(int*));
     for(int j = 0 ; j < ROW_BLK_CNT*COL_BLK_CNT ; j++){
       blk_exact_c[i][j] = (int*) calloc(blk_A*blk_C*chunk_num*chunk_num, sizeof(int));
     }
  }
  malloc_e = clk::now(); 
//TODO: need relayout optimization
  int** a_blk  = (int**)malloc(ROW_BLK_CNT*INN_BLK_CNT*COL_BLK_CNT*sizeof(int*));
  int** a_feed = (int**)malloc(ROW_BLK_CNT*INN_BLK_CNT*COL_BLK_CNT*sizeof(int*));
  for(int i = 0 ; i < ROW_BLK_CNT*INN_BLK_CNT*COL_BLK_CNT; i++){
    a_blk[i]  = (int*)calloc(blk_A*blk_C, sizeof(int)); // make sure non-mapped elements are zero by default
    a_feed[i] = (int*)malloc(blk_A*blk_C*chunk_num*sizeof(int));
  }  
  // prepare input matrix data a, including mm2conv
  double copy_us    = 0;
  double mm2conv_us = 0;
  for(int j = 0 ; j < INN_BLK_CNT ; j++){
    for(int k = 0 ; k < COL_BLK_CNT ; k++){
      for(int i = 0 ; i < ROW_BLK_CNT ; i++){
        int idx = i*INN_BLK_CNT*COL_BLK_CNT + j*COL_BLK_CNT + k;
        copy_s = clk::now();
        // TODO: modify for supporting random shape copying
        for(int curr_i = 0 ; curr_i < ((i < ROW_BLK_CNT-1 || ROW_BLK_REM == 0)?blk_A:ROW_BLK_REM) ; curr_i++){
          memcpy(a_blk[idx]+curr_i*blk_B, a+(i*blk_A+curr_i)*B+(j*blk_B), ((j < INN_BLK_CNT-1 || INN_BLK_REM == 0)?(blk_B):(INN_BLK_REM))*sizeof(int));
        }
        copy_e = clk::now();
        mm2conv_s = clk::now();
        data_mm256conv(a_blk[idx], a_feed[idx], blk_A, blk_C, IN_W, IN_H, F_W, F_H, IN_C, chunk_num);
        mm2conv_e = clk::now();
        copy_us    += std::chrono::duration_cast<std::chrono::nanoseconds>(copy_e - copy_s).count()/1000.0;
        mm2conv_us += std::chrono::duration_cast<std::chrono::nanoseconds>(mm2conv_e - mm2conv_s).count()/1000.0;
      }
    }
  }
  // start pthread creation and actual exeution
  timing exe_s = clk::now();
  double mem_us, run_us, pop_us;
#ifdef __aarch64__ //  the run section for coral dev board
  mem1_ns = 0, run1_ns = 0, pop1_ns = 0;
  int tid = 0;
  model_id = 0;
  chunk_num = 16/*CHAR_BIT*//get_chunk_size();
  for(int j = 0 ; j < INN_BLK_CNT ; j++){ 
    for(int k = 0 ; k < COL_BLK_CNT ; k++){ // outer loops can be in parallel for multiple devices to run
      for(int w_chunk_idx = 0 ; w_chunk_idx < chunk_num ; w_chunk_idx++){ 
        model_id = j*COL_BLK_CNT*chunk_num + k*chunk_num + w_chunk_idx;
        if(model_id%dev_cnt == tid){
          for(int i = 0 ; i < ROW_BLK_CNT ; i++){ // running block execution
            for(int in_chunk_idx = 0 ; in_chunk_idx < chunk_num ; in_chunk_idx++){ //stride quantization overhead
              mem1_ns += populate_input_chunking(a_feed[i*INN_BLK_CNT*COL_BLK_CNT + j*COL_BLK_CNT + k], blk_A*blk_C/*size*/, model_id, in_chunk_idx, data_type);
              run1_ns += invoke_model(model_id);
              pop1_ns += populate_output_chunking(partial_c[j], A, C, blk_A, blk_C, i, k, model_id, in_chunk_idx, w_chunk_idx);  
              printf("mem_ns: %12lld, run_ns: %12lld, pop_ns: %12lld, (i,j,k=(%d,%d,%d)), model_id: %d, tid: %d, in_chunk_idx: %d\n", mem1_ns, run1_ns, pop1_ns, i, j, k, model_id, tid, in_chunk_idx);
            } 
          }
        }// end if
      }
    }
  }
  mem_us = mem1_ns/1000.0;
  run_us = run1_ns/1000.0;
  pop_us = pop1_ns/1000.0;
  printf("(%12.3f, %12.3f, %12.3f) (us).\n", mem_us, run_us, pop_us);  
#else // the run section for host, see pthread function for detail
  struct OP_node *curr_node = (struct OP_node*) calloc(INN_BLK_CNT*COL_BLK_CNT*chunk_num, sizeof(struct OP_node)); 
  int cnt = 0;
  for(int j = 0 ; j < INN_BLK_CNT ; j++){
    for(int k = 0 ; k < COL_BLK_CNT ; k++){
        //model_id = j*COL_BLK_CNT*chunk_num + k*chunk_num + w_chunk_idx;
        model_id = j*COL_BLK_CNT + k;
        curr_node[cnt].op            = mm_model;
        curr_node[cnt].model_id      = model_id;
        curr_node[cnt].a_feed        = a_feed;
        curr_node[cnt].partial_c     = partial_c;
        curr_node[cnt].A             = A;
        curr_node[cnt].B             = B;
        curr_node[cnt].C             = C;
        curr_node[cnt].j             = j;
        curr_node[cnt].k             = k;
        curr_node[cnt].blk_A         = blk_A;
        curr_node[cnt].blk_B         = blk_B;
        curr_node[cnt].blk_C         = blk_C;
        curr_node[cnt].ROW_BLK_CNT   = ROW_BLK_CNT;
        curr_node[cnt].INNER_BLK_CNT = INN_BLK_CNT;
        curr_node[cnt].COL_BLK_CNT   = COL_BLK_CNT;
        curr_node[cnt].partial_c     = blk_exact_c[j];//partial_c;
std::cout << __func__ << ": enqueue, SCALE: " << SCALE << std::endl; 
        curr_node[cnt].SCALE         = IN_SCALE;
        curr_node[cnt].mm256         = true;
        curr_node[cnt].chunk_num     = chunk_num;
        fifo_push(SPMC_fifo, &curr_node[cnt]);
        //std::cout << __func__ << ": model_id: " << model_id << ", A: " << A << "    , B: " << B << ", C: " << C << ", j: " << j << ", k: " << ", w_chunk_idx: " << w_chunk_idx << ", blk_A: " << blk_A << ", blk_B: " << blk_B << ", blk_C: " << blk_C << ", ROW_BLK_CNT: " << ROW_BLK_CNT << ", INN_BLK_CNT: " << INN_BLK_CNT << ", COL_BLK_CNT: " << COL_BLK_CNT << ", SCALE: " << SCALE << std::endl;
        cnt++;
        if(j == (INN_BLK_CNT-1) && k == (COL_BLK_CNT-1)){ pthread_mutex_lock(&pmtx); done_enqueue = true; pthread_mutex_unlock(&pmtx); }
    }
  }
  wait_queue_all_finished();
  free(curr_node);

#endif
  timing exe_e = clk::now();
  // summation stage step1 : shrink the (A*chunk_num)*(C*chunk_num) block into A*C
  
//  for(int i  = 0 ; i < 10/*A*chunk_num*/ ; i++){
//    for(int j = 0 ; j < 10/*C*chunk_num*/ ; j++){
//      printf("%d ", partial_c[0][i*(C*chunk_num)+j]);
//  //     if(partial_c[0][i*(C*chunk_num)+j] != 0 && partial_c[0][i*(C*chunk_num)+j] != 1)
//  //       printf("(%4d, %4d) %4d\n", i, j, partial_c[0][i*(C*chunk_num)+j]);
//  //     if(partial_c[0][i*(C*chunk_num)+j] == 1)
//  //       partial_c[0][i*(C*chunk_num)+j] = 0;
//     }
//     printf("\n");
//  }
  for(int idx = 0 ; idx < INN_BLK_CNT ; idx++){
    for(int i = 0 ; i < ROW_BLK_CNT ; i++){
      for(int j = 0 ; j < COL_BLK_CNT ; j++){
        for(int k = 0 ; k < blk_A*blk_C*chunk_num*chunk_num ; k++){
          int chunk_r = k/(blk_A*blk_C*chunk_num);
          int chunk_c = (k%(blk_C*chunk_num))/blk_C;
          int inblk_r = (k/(blk_C*chunk_num))%blk_A;
          int inblk_c = (k%(blk_C*chunk_num))%blk_C;
          int offset  = chunk_r*(A*C*chunk_num) + chunk_c * C + i/*blk_idx_i*/*(blk_A*C*chunk_num) + j/*blk_idx_k*/*blk_C + inblk_r*(C*chunk_num) + inblk_c;
          partial_c[idx][offset] = blk_exact_c[idx][i*COL_BLK_CNT+j][k] << (chunk_r + chunk_c);
        }
      }
    }
  }

  int sum = 0, offset;
  int pi=0,pj=0;
  //std::cout << "print all result chunks for c[" << pi << "][" << pj <<  "]" << std::endl;
  for(int i = 0 ; i < A ; i++){
    for(int j = 0 ; j < C ; j++){
      sum = 0;
      for(int in_idx = 0 ; in_idx < INN_BLK_CNT ; in_idx++){
        //if(i == pi && j == pj)std::cout << "INN_BLK_idx: " << in_idx << std::endl;
        for(int idx_r = 0 ; idx_r < chunk_num ; idx_r++){
          for(int idx_c = 0 ; idx_c < chunk_num ; idx_c++){
            offset = idx_r*(A*C*chunk_num) + idx_c*(C) + i*(C*chunk_num)+j;
            //if(i == pi && j == pj){
            //  printf("%d ", partial_c[in_idx][offset]);
            //}
            sum += partial_c[in_idx][offset];
          }
         //if(i == pi && j == pj)printf("\n");
        }
      }
      //if(i == pi && j == pj)printf("sum: %d\n", sum);
      c[i*C+j] = sum;
    } 
  }
  // free up buffer
  free_s = clk::now();
  for(int i = 0 ; i < INN_BLK_CNT; i++){ free(partial_c[i]); }
  free(partial_c);
  for(int i = 0 ; i < ROW_BLK_CNT*INN_BLK_CNT*COL_BLK_CNT; i++){
    free(a_blk[i]);
    free(a_feed[i]);
  }
  free(a_blk);
  free(a_feed);
  free(data_array);
  free_e = clk::now();
  total_e = clk::now();
  double save_weight_us = std::chrono::duration_cast<std::chrono::nanoseconds>(save_weight_e - save_weight_s).count()/1000.0;
  double init_us        = std::chrono::duration_cast<std::chrono::nanoseconds>(init_e - init_s).count()/1000.0;
  double py_us          = std::chrono::duration_cast<std::chrono::nanoseconds>(py_e - py_s).count()/1000.0;
  double convert_us     = std::chrono::duration_cast<std::chrono::nanoseconds>(convert_e - convert_s).count()/1000.0;
  double set_data_us    = set_data_ns/1000.0;
  double create_us      = create_ns/1000.0;
  double build_m_us     = model_ns/1000.0;
  double build_itpr_us  = itpr_ns/1000.0;
//  double copy_us        = std::chrono::duration_cast<std::chrono::nanoseconds>(copy_e - copy_s).count()/1000.0;
//  double mm2conv_us     = std::chrono::duration_cast<std::chrono::nanoseconds>(mm2conv_e - mm2conv_s).count()/1000.0;
  double exe_us         = std::chrono::duration_cast<std::chrono::nanoseconds>(exe_e - exe_s).count()/1000.0;
         mem_us         = mem1_ns/(1000.0);
         run_us         = run1_ns/(1000.0*ITER);
         pop_us         = pop1_ns/(1000.0);
  double sum_us         = std::chrono::duration_cast<std::chrono::nanoseconds>(sum_e - sum_s).count()/1000.0;
  double free_us        = std::chrono::duration_cast<std::chrono::nanoseconds>(free_e - free_s).count()/1000.0;
  double total_us       = std::chrono::duration_cast<std::chrono::nanoseconds>(total_e - total_s).count()/1000.0;
  printf("| save weights : [%7.3f%%]\t%12.3f (us).|\n", (save_weight_us/total_us)*100, save_weight_us);  
  printf("| itpr init    : [%7.3f%%]\t%12.3f (us).|\n", (init_us/total_us)*100, init_us);  

  printf("| py create m  : [%7.3f%%]\t%12.3f (us).|\n", (py_us/total_us)*100, py_us);  
  printf("| convert model: [%7.3f%%]\t%12.3f (us).|\n", (convert_us/total_us)*100, convert_us);  

  
  printf("| set data     : [%7.3f%%]\t%12.3f (us).|\n", (set_data_us/total_us)*100, set_data_us);  
  printf("| create binary: [%7.3f%%]\t%12.3f (us).|\n", (create_us/total_us)*100, create_us);  

  printf("| build model  : [%7.3f%%]\t%12.3f (us).|\n", (build_m_us/total_us)*100, build_m_us);  
  printf("| build itpr   : [%7.3f%%]\t%12.3f (us).|\n", (build_itpr_us/total_us)*100, build_itpr_us);  
  printf("| copy input   : [%7.3f%%]\t%12.3f (us).|\n", (copy_us/total_us)*100, copy_us);  
  printf("| mm2conv      : [%7.3f%%]\t%12.3f (us).|\n", (mm2conv_us/total_us)*100, mm2conv_us);  
  printf("| mem          : [%7.3f%%]\t%12.3f (us).|\n", (mem_us/total_us)*100, mem_us);  
  printf("| avg run      : [%7.3f%%]\t%12.3f (us).|\n", (run_us/total_us)*100, run_us);  
  printf("| exe in total : [%7.3f%%]\t%12.3f (us).|\n", (exe_us/total_us)*100, exe_us);  
  printf("| pop          : [%7.3f%%]\t%12.3f (us).|\n", (pop_us/total_us)*100, pop_us);  
  printf("| sum          : [%7.3f%%]\t%12.3f (us).|\n", (sum_us/total_us)*100, sum_us);  
  printf("| free         : [%7.3f%%]\t%12.3f (us).|\n", (free_us/total_us)*100, free_us);  
  long long int term1=0, term2=0;
  term1 = 2*B-1;
  term2 = A*C;
  long long int op_cnt = term1 * term2;  
  printf("| invoke  GOPS: %10.4f, op_cnt: %13lld, us: %12.3f\t|\n", (float)op_cnt/(exe_us*1000), op_cnt, exe_us);   
  printf("| end2end GOPS: %10.4f, op_cnt: %13lld, us: %12.3f\t|\n", (float)op_cnt/(total_us*1000), op_cnt, total_us);   
  return 0;
}

int gptpu_mm2conv(int* a, int* b, int* c, int A, int B, int C, bool b_major){
  // for current design, only square matrix are allowed
  timing total_s, total_e, init_s, init_e, save_weight_s, set_data_s, set_data_e, save_weight_e, py_s, py_e, convert_s, convert_e, create_s, create_e, build_m_s, build_m_e, build_itpr_s, build_itpr_e, malloc_s, malloc_e, copy_s, copy_e, mm2conv_s, mm2conv_e, run_s, run_e, sum_s, sum_e, free_s, free_e;
  double set_data_ns = 0, create_ns = 0, model_ns = 0, itpr_ns = 0;
  total_s = clk::now();
  std::string matrix_path;
  std::string command;
  int IN_W, IN_H, F_W, F_H, IN_C, S_W, S_H, OUT_C;
  std::string prefix_model_name, model_name, weight_file_name;
  // determine convolution shape for mm or mv based on input A, B, C
  int blk_A;
  int blk_B;
  int blk_C;
  mm2conv_shape_mapping(A, B, C, blk_A, blk_B, blk_C, exact_mode, IN_W, IN_H, IN_C, F_W, F_H, S_W, S_H, OUT_C);

  std::string mm2conv_template_prefix_name = "conv_temp_"+itoa(IN_W)+"x"+itoa(IN_H)+"x"+itoa(IN_C)+"x"+itoa(F_W)+"x"+itoa(F_H)+"x"+itoa(S_W)+"x"+itoa(S_H);
  int ROW_BLK_CNT = sub_cnt(A, blk_A); //  (A / blk_A) + ((A % blk_A != 0)?1:0);  
  int INN_BLK_CNT = sub_cnt(B, blk_B); //  (B / blk_B) + ((B % blk_B != 0)?1:0);  
  int COL_BLK_CNT = sub_cnt(C, blk_C); //  (C / blk_C) + ((C % blk_C != 0)?1:0);
  int ROW_BLK_REM = A % blk_A;
  int INN_BLK_REM = B % blk_B;
  int COL_BLK_REM = C % blk_C;
  std::cout << "A: " << A << ", B: " << B << ", C: " << C << ", cblk_A: " << blk_A << ", blk_B: " << blk_B << ",blk_C: " << blk_C << ", ROW_BLK_CNT: " << ROW_BLK_CNT << ", INN_BLK_CNT: " << INN_BLK_CNT << ", COL_BLK_CNT: " << COL_BLK_CNT << ", ROW_BLK_REM: " << ROW_BLK_REM << ", INN_BLK_REM: " << INN_BLK_REM << ", COL_BLK_REM: " << COL_BLK_REM << std::endl;
  int chunk_num = CHAR_BIT/get_chunk_size();
  int model_id = 0;
  //create matrix weight from array 'b' (input from array 'a' later)
  char* data_array = (char*)malloc(blk_B*blk_C*sizeof(char));
  bool template_created = false; //avoid re-creating same shape of mm during blocking algorithm

// ===== get default scaling factor =====
  float IN_SCALE = 1;
  SCALE = (exact_mode== 1)?1:get_auto_scale_factor_mm(a, b, A, B, C);
  //set_scale(1.0/65536.0);
  std::cout << "SCALE: " << SCALE << ", IN_SCALE: " << IN_SCALE << std::endl;
// ===== end scaling ========
  for(int i = 0 ; i < INN_BLK_CNT ; i++){
    for(int j = 0 ; j< COL_BLK_CNT ; j++){
      for(int w_chunk_idx = start_chunk ; w_chunk_idx < chunk_num ; w_chunk_idx++){
        model_id = i*COL_BLK_CNT*chunk_num + j*chunk_num + w_chunk_idx;
        // TODO: template is constrianted with shape size 
        prefix_model_name = "conv_model_quant_"+itoa(IN_W)+"x"+itoa(IN_H)+"x"+itoa(IN_C)+"x"+itoa(F_W)+"x"+itoa(F_H)+"x"+itoa(S_W)+"x"+itoa(S_H)+"_2048based_"+itoa(i)+"x"+itoa(j)+"_outof_"+itoa(INN_BLK_CNT)+"x"+itoa(COL_BLK_CNT)+"_chunk"+itoa(w_chunk_idx)+"of"+itoa(chunk_num);
        std::string template_name = mm2conv_template_prefix_name + "_" + data_type + ".tflite";
        template_path = temp_dir + template_name;
        matrix_path   = data_dir+"conv_model_tflite/" + prefix_model_name + "_edgetpu.tflite";  
        //std::cout << "template_path: " << template_path << std::endl;
        //std::cout << "matrix_path  : " << matrix_path << std::endl;
        if( 1 || (file_exist(template_path) == false) && template_created == false){
          template_created = true; // if really need to create template, only once for all same shape blocks
          //save weight in file(s) for python to create model
          std::cout << "template_path: " << template_path << " not exist, creating it..." << std::endl;
          weight_file_name = "./../mm2conv_weight_"+itoa(i)+"x"+itoa(j)+"_outof_"+itoa(INN_BLK_CNT)+"x"+itoa(COL_BLK_CNT)+"_chunk"+itoa(w_chunk_idx)+"of"+itoa(chunk_num)+".txt";
          save_weight_s = clk::now();
          mm2conv_save_weight(b, b_major, weight_file_name, B, C, i, j, INN_BLK_CNT, COL_BLK_CNT, blk_B, blk_C, INN_BLK_REM, COL_BLK_REM, w_chunk_idx);
          save_weight_e = clk::now();
          //create xxx_quant.tflite
          //model_name     = prefix_model_name + "_edgetpu.tflite";
          //matrix_path    = data_dir+"conv_model_tflite/conv_model_quant_edgetpu.tflite";  
//TODO; the scale is decided by max value
          std::cout << "scale is: " << SCALE << std::endl;
          command = "python3 "+PWD+"/src/create_model.py --model=conv_model --in_w_name="+weight_file_name+" --data_type="+data_type+" --out_scale="+itoa(SCALE)+" --outfile_name="+matrix_path+" --IN_W="+itoa(IN_W)+" --IN_H="+itoa(IN_H)+" --IN_C="+itoa(IN_C)+" --F_W="+itoa(F_W)+" --F_H="+itoa(F_H)+" --S_W="+std::to_string(F_W)+" --S_H="+std::to_string(F_H)+" --OUT_C="+itoa(OUT_C);
          py_s = clk::now();
          std::cout << "command: " << command.c_str() << std::endl;
          int ret = system(command.c_str());
          py_e = clk::now();
          //make_mm2conv_temp(matrix_path, local_temp_dir+template_name, blk_A, blk_B, blk_C);
          command = "sudo cp "+local_temp_dir+template_name+" "+template_path;
          ret = system(command.c_str());
        }
        // set data array
        set_data_s = clk::now();
std::cout << "B: " << B << ", C: " << C << ", i: " << i << ", j: " << j << ", INN_BLK_CNT: " << INN_BLK_CNT << ", COL_BLK_CNT: " << COL_BLK_CNT << ", blk_B: " << blk_B << ", blk_C: " << blk_C << ", INN_BLK_REM: " << INN_BLK_REM << ", COL_BLK_REM: " << COL_BLK_REM << ", w_chunk_idx: " << w_chunk_idx << ", exact_mode: " << exact_mode << std::endl;
        set_mm2conv_array(b, b_major, data_array, B, C, i, j, INN_BLK_CNT, COL_BLK_CNT, blk_B, blk_C, INN_BLK_REM, COL_BLK_REM, w_chunk_idx, exact_mode);
        set_data_e = clk::now();
        set_data_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(set_data_e - set_data_s).count();
        // create model
        create_s = clk::now();
//std::cout << "blk_A: " << blk_A << ", SCALE: " << SCALE << std::endl;
        create_mm2conv_tflite(template_path, flatbufs, /*matrix_path*/model_id, data_array, blk_A, blk_B, blk_C, SCALE, 1/*dummy*/);
        create_e = clk::now();
        create_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(create_e - create_s).count();
        //build model
        //template_path  = data_dir+"conv_model_tflite/" + prefix_model_name + "_edgetpu.tflite";  
        //matrix_path    = data_dir+"conv_model_tflite/conv_model_quant_edgetpu.tflite";  
        build_m_s = clk::now();
//std::cout << "build model, the matrix_path: " << matrix_path << std::endl;
        build_model(matrix_path, model_id);
        build_m_e = clk::now();
        model_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(build_m_e - build_m_s).count();
        //build interpreter
        build_itpr_s = clk::now();
        //build_interpreter(/*tpu_id*/(model_id)%dev_cnt, model_id);
        build_itpr_e = clk::now();
        itpr_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(build_itpr_e - build_itpr_s).count();
      }
    }
  }
  long long int mem1_ns, run1_ns, pop1_ns;
  int output_size = 0;
  malloc_s = clk::now(); 
  partial_c = (int**)malloc(INN_BLK_CNT*sizeof(int*));
  for(int i = 0 ; i < INN_BLK_CNT; i++){
    partial_c[i] = (int*)calloc(A*C, sizeof(int));
  }
  malloc_e = clk::now(); 
  // how does conv2D view it's input shape?
  // input[IN_W,IN_H,IN_C]
  //prepare input A
/*
  input a     weight B            result c
  +---+---+   +---+---+---+---+   +-------+-------+-------+-------+
  | A | B |   | 1 | 2 | 3 | 4 |   | 1A+5B | 2A+6B | 3A+7B | 4A+8B |
  +---+---+ X +---+---+---+---+ = +-------+-------+-------+-------+
  | C | D |   | 5 | 6 | 7 | 8 |   | 1C+5D | 2C+6D | 3C+7D | 4C+8D |
  +---+---+   +---+---+---+---+   +-------+-------+-------+-------+
  | E | F |                       | 1E+5F | 2E+6F | 3E+7F | 4E+8F |
  +---+---+                       +-------+-------+-------+-------+

  ROW_BLK_CNT X INN_BLK_CNT = ROW_BLK_CNT
  INN_BLK_CNT   COL_BLK_CNT   COL_BLK_CNT
*/
//TODO: need relayout optimization
  int** a_blk  = (int**)malloc(ROW_BLK_CNT*INN_BLK_CNT*COL_BLK_CNT*sizeof(int*));
  int** a_feed = (int**)malloc(ROW_BLK_CNT*INN_BLK_CNT*COL_BLK_CNT*sizeof(int*));
  for(int i = 0 ; i < ROW_BLK_CNT*INN_BLK_CNT*COL_BLK_CNT; i++){
    a_blk[i]  = (int*)calloc(blk_A*blk_C, sizeof(int)); // make sure non-mapped elements are zero by default
    a_feed[i] = (int*)malloc(blk_A*blk_C*sizeof(int));
  }  
  // prepare input matrix data a, including mm2conv
  double copy_us    = 0;
  double mm2conv_us = 0;
  for(int j = 0 ; j < INN_BLK_CNT ; j++){
    for(int k = 0 ; k < COL_BLK_CNT ; k++){
      for(int i = 0 ; i < ROW_BLK_CNT ; i++){
        int idx = i*INN_BLK_CNT*COL_BLK_CNT + j*COL_BLK_CNT + k;
        copy_s = clk::now();
        // TODO: modify for supporting random shape copying
        for(int curr_i = 0 ; curr_i < ((i < ROW_BLK_CNT-1 || ROW_BLK_REM == 0)?blk_A:ROW_BLK_REM) ; curr_i++){
          memcpy(a_blk[idx]+curr_i*blk_B, a+(i*blk_A+curr_i)*B+(j*blk_B), ((j < INN_BLK_CNT-1 || INN_BLK_REM == 0)?(blk_B):(INN_BLK_REM))*sizeof(int));
        }
// old design
//        for(int curr_i = 0 ; curr_i < blk_A ; curr_i++){
//          memcpy(a_blk[idx]+curr_i*blk_B, a+(i*blk_A+curr_i)*B+(j*blk_B), blk_B*sizeof(int));
//        }
        copy_e = clk::now();
        mm2conv_s = clk::now();
        data_mm2conv(a_blk[idx], a_feed[idx], blk_A, blk_C, IN_W, IN_H, F_W, F_H, IN_C);
        mm2conv_e = clk::now();
        copy_us    += std::chrono::duration_cast<std::chrono::nanoseconds>(copy_e - copy_s).count()/1000.0;
        mm2conv_us += std::chrono::duration_cast<std::chrono::nanoseconds>(mm2conv_e - mm2conv_s).count()/1000.0;
      }
    }
  }
  // start pthread creation and actual exeution
  timing exe_s = clk::now();
  double mem_us, run_us, pop_us;
#ifdef __aarch64__ //  the run section for coral dev board
  mem1_ns = 0, run1_ns = 0, pop1_ns = 0;
  int tid = 0;
  model_id = 0;
  //chunk_num = 16/*CHAR_BIT*//get_chunk_size();
  for(int j = 0 ; j < INN_BLK_CNT ; j++){ 
    for(int k = 0 ; k < COL_BLK_CNT ; k++){ // outer loops can be in parallel for multiple devices to run
      for(int w_chunk_idx = 0 ; w_chunk_idx < chunk_num ; w_chunk_idx++){ 
        model_id = j*COL_BLK_CNT*chunk_num + k*chunk_num + w_chunk_idx;
        if(model_id%dev_cnt == tid){
          for(int i = 0 ; i < ROW_BLK_CNT ; i++){ // running block execution
            for(int in_chunk_idx = 0 ; in_chunk_idx < chunk_num ; in_chunk_idx++){ //stride quantization overhead
              mem1_ns += populate_input_chunking(a_feed[i*INN_BLK_CNT*COL_BLK_CNT + j*COL_BLK_CNT + k], blk_A*blk_C/*size*/, model_id, in_chunk_idx, data_type);
              run1_ns += invoke_model(model_id);
              pop1_ns += populate_output_chunking(partial_c[j], A, C, blk_A, blk_C, i, k, model_id, in_chunk_idx, w_chunk_idx);  
              printf("mem_ns: %12lld, run_ns: %12lld, pop_ns: %12lld, (i,j,k=(%d,%d,%d)), model_id: %d, tid: %d, in_chunk_idx: %d\n", mem1_ns, run1_ns, pop1_ns, i, j, k, model_id, tid, in_chunk_idx);
            } 
          }
        }// end if
      }
    }
  }
  mem_us = mem1_ns/1000.0;
  run_us = run1_ns/1000.0;
  pop_us = pop1_ns/1000.0;
  printf("(%12.3f, %12.3f, %12.3f) (us).\n", mem_us, run_us, pop_us);  
#else // the run section for host, see pthread function for detail
  struct OP_node *curr_node = (struct OP_node*) calloc(INN_BLK_CNT*COL_BLK_CNT*chunk_num, sizeof(struct OP_node)); 
  int cnt = 0;
  for(int j = 0 ; j < INN_BLK_CNT ; j++){
    for(int k = 0 ; k < COL_BLK_CNT ; k++){
      for(int w_chunk_idx = start_chunk ; w_chunk_idx < chunk_num ; w_chunk_idx++){
        //model_id = j*COL_BLK_CNT*chunk_num + k*chunk_num + w_chunk_idx;
        model_id = (j*COL_BLK_CNT + k) * chunk_num + w_chunk_idx;
        curr_node[cnt].op            = mm_model;
        curr_node[cnt].model_id      = model_id;
        curr_node[cnt].a_feed        = a_feed;
        curr_node[cnt].partial_c     = partial_c;
        curr_node[cnt].A             = A;
        curr_node[cnt].B             = B;
        curr_node[cnt].C             = C;
        curr_node[cnt].j             = j;
        curr_node[cnt].k             = k;
        curr_node[cnt].w_chunk_idx   = w_chunk_idx;
        curr_node[cnt].start_chunk   = start_chunk;
        curr_node[cnt].blk_A         = blk_A;
        curr_node[cnt].blk_B         = blk_B;
        curr_node[cnt].blk_C         = blk_C;
        curr_node[cnt].ROW_BLK_CNT   = ROW_BLK_CNT;
        curr_node[cnt].INNER_BLK_CNT = INN_BLK_CNT;
        curr_node[cnt].COL_BLK_CNT   = COL_BLK_CNT;
        curr_node[cnt].partial_c     = partial_c;
        curr_node[cnt].SCALE         = SCALE;
        curr_node[cnt].mm256         = false;
        fifo_push(SPMC_fifo, &curr_node[cnt]);
        //std::cout << __func__ << ": model_id: " << model_id << ", A: " << A << "    , B: " << B << ", C: " << C << ", j: " << j << ", k: " << ", w_chunk_idx: " << w_chunk_idx << ", blk_A: " << blk_A << ", blk_B: " << blk_B << ", blk_C: " << blk_C << ", ROW_BLK_CNT: " << ROW_BLK_CNT << ", INN_BLK_CNT: " << INN_BLK_CNT << ", COL_BLK_CNT: " << COL_BLK_CNT << ", SCALE: " << SCALE << std::endl;
        cnt++;
        if(j == (INN_BLK_CNT-1) && k == (COL_BLK_CNT-1) && w_chunk_idx == (chunk_num-1)){ pthread_mutex_lock(&pmtx); done_enqueue = true; pthread_mutex_unlock(&pmtx); }
      }
    }
  }
  wait_queue_all_finished();
  free(curr_node);

//  pthread_t tid[dev_cnt];
//  struct arg_mm2conv_struct args[dev_cnt];
//  for(int tid_idx = 0 ; tid_idx < dev_cnt ; tid_idx++){
//    args[tid_idx].tid         = tid_idx;
//    args[tid_idx].a           = a_feed;
//    args[tid_idx].partial_c   = partial_c;
//    args[tid_idx].A           = A;
//    args[tid_idx].B           = B;
//    args[tid_idx].C           = C;
//    args[tid_idx].blk_A       = blk_A;
//    args[tid_idx].blk_B       = blk_B;
//    args[tid_idx].blk_C       = blk_C;
//    args[tid_idx].ROW_BLK_CNT = ROW_BLK_CNT;
//    args[tid_idx].INN_BLK_CNT = INN_BLK_CNT;
//    args[tid_idx].COL_BLK_CNT = COL_BLK_CNT;
//    args[tid_idx].SCALE       = IN_SCALE;
//    pthread_create(&tid[tid_idx], NULL, mm2conv_run, (void *)&args[tid_idx]);
//  }
//  for(int i = 0 ; i < dev_cnt ; i++){
//    pthread_join(tid[i], NULL);
//  }
#endif
  timing exe_e = clk::now();

  for(int i = 0 ; i < INN_BLK_CNT ; i++){
    std::cout << __func__ << ":[after] i: " << i << ", partiaL_c[" << i << "][0-3]: " << partial_c[i][0] << ", " << partial_c[i][1] << ", " << partial_c[i][2] << ", " << partial_c[i][3] << std::endl; 
  }

  // partial result summation
  sum_s = clk::now();
  int sum = 0;
  for(int i = 0; i < A*C ; i++){
    sum = 0;
    for(int j =  0; j< INN_BLK_CNT ; j++){  sum += partial_c[j][i]; }
    c[i] = sum;
  }
  sum_e = clk::now();
  // free up buffer
  free_s = clk::now();
  for(int i = 0 ; i < INN_BLK_CNT; i++){ free(partial_c[i]); }
  free(partial_c);
  for(int i = 0 ; i < ROW_BLK_CNT*INN_BLK_CNT*COL_BLK_CNT; i++){
    free(a_blk[i]);
    free(a_feed[i]);
  }
  free(a_blk);
  free(a_feed);
  free(data_array);
  free_e = clk::now();
  total_e = clk::now();
  double save_weight_us = std::chrono::duration_cast<std::chrono::nanoseconds>(save_weight_e - save_weight_s).count()/1000.0;
  double init_us        = std::chrono::duration_cast<std::chrono::nanoseconds>(init_e - init_s).count()/1000.0;
  double py_us          = std::chrono::duration_cast<std::chrono::nanoseconds>(py_e - py_s).count()/1000.0;
  double convert_us     = std::chrono::duration_cast<std::chrono::nanoseconds>(convert_e - convert_s).count()/1000.0;
  double set_data_us    = set_data_ns/1000.0;
  double create_us      = create_ns/1000.0;
  double build_m_us     = model_ns/1000.0;
  double build_itpr_us  = itpr_ns/1000.0;
//  double copy_us        = std::chrono::duration_cast<std::chrono::nanoseconds>(copy_e - copy_s).count()/1000.0;
//  double mm2conv_us     = std::chrono::duration_cast<std::chrono::nanoseconds>(mm2conv_e - mm2conv_s).count()/1000.0;
  double exe_us         = std::chrono::duration_cast<std::chrono::nanoseconds>(exe_e - exe_s).count()/1000.0;
         mem_us         = mem1_ns/(1000.0);
         run_us         = run1_ns/(1000.0*ITER);
         pop_us         = pop1_ns/(1000.0);
  double sum_us         = std::chrono::duration_cast<std::chrono::nanoseconds>(sum_e - sum_s).count()/1000.0;
  double free_us        = std::chrono::duration_cast<std::chrono::nanoseconds>(free_e - free_s).count()/1000.0;
  double total_us       = std::chrono::duration_cast<std::chrono::nanoseconds>(total_e - total_s).count()/1000.0;
  printf("| save weights : [%7.3f%%]\t%12.3f (us).|\n", (save_weight_us/total_us)*100, save_weight_us);  
  printf("| itpr init    : [%7.3f%%]\t%12.3f (us).|\n", (init_us/total_us)*100, init_us);  

  printf("| py create m  : [%7.3f%%]\t%12.3f (us).|\n", (py_us/total_us)*100, py_us);  
  printf("| convert model: [%7.3f%%]\t%12.3f (us).|\n", (convert_us/total_us)*100, convert_us);  

  
  printf("| set data     : [%7.3f%%]\t%12.3f (us).|\n", (set_data_us/total_us)*100, set_data_us);  
  printf("| create binary: [%7.3f%%]\t%12.3f (us).|\n", (create_us/total_us)*100, create_us);  

  printf("| build model  : [%7.3f%%]\t%12.3f (us).|\n", (build_m_us/total_us)*100, build_m_us);  
  printf("| build itpr   : [%7.3f%%]\t%12.3f (us).|\n", (build_itpr_us/total_us)*100, build_itpr_us);  
  printf("| copy input   : [%7.3f%%]\t%12.3f (us).|\n", (copy_us/total_us)*100, copy_us);  
  printf("| mm2conv      : [%7.3f%%]\t%12.3f (us).|\n", (mm2conv_us/total_us)*100, mm2conv_us);  
  printf("| mem          : [%7.3f%%]\t%12.3f (us).|\n", (mem_us/total_us)*100, mem_us);  
  printf("| avg run      : [%7.3f%%]\t%12.3f (us).|\n", (run_us/total_us)*100, run_us);  
  printf("| exe in total : [%7.3f%%]\t%12.3f (us).|\n", (exe_us/total_us)*100, exe_us);  
  printf("| pop          : [%7.3f%%]\t%12.3f (us).|\n", (pop_us/total_us)*100, pop_us);  
  printf("| sum          : [%7.3f%%]\t%12.3f (us).|\n", (sum_us/total_us)*100, sum_us);  
  printf("| free         : [%7.3f%%]\t%12.3f (us).|\n", (free_us/total_us)*100, free_us);  
  long long int term1=0, term2=0;
  term1 = 2*B-1;
  term2 = A*C;
  long long int op_cnt = term1 * term2;  
  printf("| invoke  GOPS: %10.4f, op_cnt: %13lld, us: %12.3f\t|\n", (float)op_cnt/(exe_us*1000), op_cnt, exe_us);   
  printf("| end2end GOPS: %10.4f, op_cnt: %13lld, us: %12.3f\t|\n", (float)op_cnt/(total_us*1000), op_cnt, total_us);   
  return 0;
}

int gptpu_mm2mul(int* a, int* b, int* c, int A, int B, int C, bool b_major){
  // for current design, only square matrix are allowed
  if(b_major != true){
    std::cout << __func__ << ", current impl. doesn't not support mm2mul with b_major = " << b_major << std::endl;
    exit(0);
  }
  set_chunk_size(mul_chunk_size);
  //===== init params =====
  timing total_start = clk::now();
  std::string model_name = mul_model;
  std::string matrix_path    = tempG_dir+"/"+model_name+"_tflite/"+model_name+"_quant.tflite";  
  timing set_start, set_end, new_start, new_end, save_start, save_end, py_start,py_end;
  double make_tflite_model_us, py_us, total_us, set_us, new_us, model_us, itpr_init_us, itpr_us, mem_us, run_us, pop_us, out_buf_us, exe_us, sum_us, trans_us;
  int ret, input_mode, model_id = 0;
  union scaling matrix_s, bias_s;
  unsigned long long int size, out_size;
  std::string mm_input_path_name;
  int chunk_num = CHAR_BIT/get_chunk_size();
  int mm2mul_blk_row = 128;
  int mm2mul_blk_col = 128;
  int blk_A = (A >= mm2mul_blk_row)?mm2mul_blk_row:A;
  int blk_B = (B >= mm2mul_blk_col)?mm2mul_blk_col:B;
  int ROW_BLK_CNT = sub_cnt(A, blk_A); //  (A / blk_A) + ((A % blk_A != 0)?1:0);  
  int COL_BLK_CNT = sub_cnt(B, blk_B); //  (B / blk_B) + ((B % blk_B != 0)?1:0);  
  int ROW_BLK_REM = A % blk_A;
  int COL_BLK_REM = B % blk_B;
// model name is by default: mul_model
  size     = (unsigned long long int)(blk_A*blk_B);
  out_size = (unsigned long long int)(blk_A*blk_B); 
  matrix_path    = data_dir+model_name+"_tflite/"+model_name+"_"+std::to_string(blk_A)+"x"+std::to_string(blk_B)+"_quant.tflite";  
  char* data_array = (char*)malloc(size*sizeof(char));
// ===== generate tflite =====
  timing make_start = clk::now();
  std::string sub_path =  model_name+"_temp.tflite";
  template_path  = temp_dir + sub_path;
  std::string command = "python3 "+PWD+"/src/create_model.py --platform=m2 --model="+model_name+" --outfile_name="+matrix_path+" --in_size="+std::to_string(blk_A)+" --out_size="+std::to_string(blk_B)+" --out_scale="+itoa(SCALE)+" --ramdisk="+std::to_string(ramdisk);
  py_start = clk::now();
  ret = system(command.c_str());
  py_end   = clk::now();
  new_end   = clk::now();
  //new_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(new_end - new_start).count(); 
  timing make_end = clk::now();

// prepare edgetpu run
  for(int i = 0 ; i < dev_cnt ; i++){
    model_ns += build_model(matrix_path, /*model_id*/i);
    itpr_ns  += build_interpreter(/*tpu_id*/i, /*model_id*/i);
  }
  timing exe_start = clk::now();
  int output_size = 0;
  int** tmp_c = (int**) malloc(A*sizeof(int*));
  for(int i = 0 ; i < A ; i++){
    tmp_c[i] = (int*) calloc(A*B, sizeof(int));
  }

// ===== start enqueue ======
  for(int idx = 0 ; idx < A ; idx++){  
    for(int i = 0 ; i < ROW_BLK_CNT ; i++){
      for(int j = 0 ; j < COL_BLK_CNT ; j++){
        for(int xi = 0 ; xi < sub_cnt(16, mul_chunk_size) ; xi++){
          for(int yi = 0 ; yi < sub_cnt(16, mul_chunk_size) ; yi++){
            pthread_mutex_lock(&pmtx);
            while((qin+1)%queue_size == qout){
              pthread_cond_wait(&out_CV, &pmtx);
            }
            // enqueue
            Q[qin].op = model_name;
            Q[qin].A = blk_A;
            Q[qin].B = blk_B;
            Q[qin].i = i;
            Q[qin].j = j;
            Q[qin].ROW_BLK_CNT = ROW_BLK_CNT;
            Q[qin].COL_BLK_CNT = COL_BLK_CNT;
            Q[qin].a = a;
            Q[qin].b = b;
            Q[qin].c = tmp_c[idx]; // for this version, we need proper result layout, need to skip the summation stage in consumer
            Q[qin].output_no_chunk = true;
            Q[qin].idx = idx;
            Q[qin].xi = xi;
            Q[qin].yi = yi;
            Q[qin].offset = (i*COL_BLK_CNT+j)*out_size;
            // done enqueue
            queue_cnt++;
            qin = (qin+1)%queue_size;
            pthread_cond_signal(&in_CV);
            if(idx == (A-1) && i == (ROW_BLK_CNT-1) && j == (COL_BLK_CNT-1) && xi == (sub_cnt(16, mul_chunk_size)-1) && yi == (sub_cnt(16, mul_chunk_size)-1)){ std::cout << "idx: " << idx << ", i: " << i << ", j: " << j << ", xi: " << xi << ", yi: " << yi << std::endl; done_enqueue = true;  }
            pthread_mutex_unlock(&pmtx);
          }
        }
      }
    }
  }

// ===== end enqueue =====
  wait_queue_all_finished();
// for each row [0:n] [n:2n] [2n:3n] ...
// it should be summed up into c[0,0] c[1,1] c[2,2] ...
  unsigned long long int sum = 0;
  for(int idx = 0 ; idx < A ; idx++){
    for(int i = 0 ; i < A ; i++){
      sum = 0;
      for(int j = 0 ; j < B ; j++){
        sum += tmp_c[idx][i*B+j];
      }
      c[((i+idx)%A)*B+i] = sum;
    }
  }
// ===== end exe =====
  for(int i = 0 ; i < A ; i++){
     free(tmp_c[i]);
  }
  free(tmp_c);

  timing exe_end   = clk::now();
  timing total_end = clk::now();
  if(BREAKDOWN == 1){
    std::cout << std::setw(12);
    make_tflite_model_us = std::chrono::duration_cast<std::chrono::nanoseconds>(make_end  - make_start ).count()/1000.0;
    py_us                = std::chrono::duration_cast<std::chrono::nanoseconds>(py_end    - py_start   ).count()/1000.0;
    total_us             = std::chrono::duration_cast<std::chrono::nanoseconds>(total_end - total_start).count()/1000.0;
    set_us               = std::chrono::duration_cast<std::chrono::nanoseconds>(set_end   - set_start  ).count()/1000.0;
    new_us               = std::chrono::duration_cast<std::chrono::nanoseconds>(new_end   - new_start  ).count()/1000.0;
    model_us             = model_ns/1000.0;
    itpr_init_us         = itpr_init_ns /1000.0;
    itpr_us              = itpr_ns /1000.0;
    mem_us               = mem_ns  /1000.0;
    run_us               = run_ns  /1000.0;

    FILE* fp;    
    fp = fopen("./record.txt", "a");
    fprintf(fp, "%s, A:%d, B:%d, blk_A:%d, blk_B: %d, run_us:%f, iter:%d\n", model_name.c_str(), A, B, blk_A, blk_B, run_us, ITER);
    fclose(fp);

    pop_us               = pop_ns  /1000.0;
    exe_us               = std::chrono::duration_cast<std::chrono::nanoseconds>(exe_end   - exe_start  ).count()/1000.0;
    printf("+----- gptpu_mm2mul timing breakdown ---------------------------+\n");
    printf("+----- make model breakdown below ------------------------------+ average, each time may differ \n");
    printf("| input data converting: [%7.3f%%]\t%12.3f (us).\t|\n", (set_us/total_us)*100, set_us);  
    printf("| create tflite lite   : [%7.3f%%]\t%12.3f (us).\t|\n", (new_us/total_us)*100, new_us);  
    printf("+----- run detail below ----------------------------------------+\n");
    printf("| build model          : [%7.3f%%]\t%12.3f (us).\t|\n", (model_us/total_us)*100, model_us);  
    printf("| build itpr           : [%7.3f%%]\t%12.3f (us).\t|\n", (itpr_us/total_us)*100, itpr_us);  
    printf("| transfer input       : [%7.3f%%]\t%12.3f (us).\t|\n", (mem_us/total_us)*100, mem_us);  
    printf("| run (invoke)         : [%7.3f%%]\t%12.3f (us).\t|\n", (run_us/total_us)*100, run_us);  
    printf("| populate out buffer  : [%7.3f%%]\t%12.3f (us).\t|\n", (pop_us/total_us)*100, pop_us);  
    printf("+---------------------------------------------------------------+\n");
    printf("| total                : [%7.3f%%]\t%12.3f (us).\t| (some impl. related overhead ignored for now)\n", (total_us/total_us)*100, total_us);  
    printf("+---------------------------------------------------------------+\n");
    long long int ns     = run_ns;
    long long int term1 = ITER * B;
    long long int term2 = (2 * A) - 1;
    long long int op_cnt = term1 * term2; 
    printf("| GOPS: %10.4f, op_cnt: %13lld, us: %12.3f\t| (valid for mm only now)\n", (float)op_cnt/ns, op_cnt, run_us);   
    printf("+---------------------------------------------------------------+\n");
    //std::cout << "|malloc partial       : " << std::chrono::duration_cast<std::chrono::nanoseconds>(malloc_e - malloc_s).count()/1000.0 << "\t(us).\t\t|" << std::endl; 
  }
  free(data_array);
  return run_us;
}

int gptpu_vs(int* a, int* c, int A , float scale){
  int* b; //dummy
  open_devices(0, 1);
  return (int)gptpu_main(b, a, c, 1, A, 0/*blk_row*/, 0/*blk_col*/, 0/*start_i*/, 0/*start_j*/, scale, vs_model, 0, false);
}

int gptpu_mac(int* a, int* b, int& c, int A){
  open_devices(0, 1);
  float scale = SCALE; // dummy
  if(A >= 8){
    set_block(1, A);
    return gptpu_block(a, b, true, &c, 1, A, 1, scale, mv_model, 0);
  }else{ //  small vector size handling 
    c = 0;
    for(int i = 0 ; i < A ; i++){
      c += a[i] * b[i];
    }
    return 0;
  }
}

float gptpu_conv2D(int* a, int* f, int* c, int A, int B, int M, int N, const std::string& padding){
// same padding
  open_devices(0, 1);
  float scale = 1;
  int padding_option = 0; // default as valid padding
  if(M%2 == 0 || N%2 == 0){
     std::cout << __func__ << ": current gptpu supports odd value filter size only, exit." << std::endl;
     exit(0);
  }
  if(!padding.compare("same")){
    padding_option = 1;
  }else if(!padding.compare("replication")){
    padding_option = 2;
  }
  return gptpu_main(a, f, c, A, B, M/*blk_row*/, N/*blk_col*/, padding_option/*start_i*/, 0/*start_j*/, scale, conv_model, 0/*tf*/, false);
}

int gptpu_black(int* a, int*b, int*c, int A, int B){
  float scale;
  std::string black_model = "black_model";
  gptpu_main(a, b, c, A, B, 0, 0, 0 , 0, scale, black_model, 0, false);
}

int gptpu_mv(int* a/*row-major: [A][B]*/, int* b/*vector: [B]*/, int* c/*vector: [A]*/, int A ,int B){
  open_devices(0, 1);
//  SCALE = get_auto_scale_factor_mm(a,b,A,B,1) ;
  return gptpu_block(a, b, true, c, A, B, 1, SCALE, mv_model, 0);
// ===== Note: mm2conv version for mv is slow, but worakble =====
//  return gptpu_mm2conv(b, a, c, 1, B, A, 1/*b_major*/);
// ==============================================================
}

int gptpu_imv(int* a/*row-major: [A][B]*/, int* b/*vector: [B]*/, int* c/*vector: [A]*/, int A ,int B/*, int iter*/){
  open_devices(0, 1);
#if MM_impl == 1 /* conv implementation*/
  return gptpu_mm2conv(b, a, c, 1, B, A, 1/*b_major*/);
#else
//  std::cout << "this A: " << A << ", B: " << B << std::endl;
  return gptpu_block(a, b, true, c, A, B, 1, SCALE, imv_model, 0);
#endif
}

int gptpu_smv(int* a/*row-major: [A][B]*/, int* b/*vector: [B]*/, int* c/*vector: [A]*/, int A ,int B, float scale){
  open_devices(0, 1);
#if MM_impl == 1 /* conv implementation*/
  return gptpu_mm2conv(b, a, c, 1, B, A, 1/*b_major*/);
#else
//  std::cout << "this A: " << A << ", B: " << B << std::endl;
  return gptpu_block(a, b, true, c, A, B, 1, scale, mv_model, 0);
#endif
}

// min-plus matrix multiplication
//int gptpu_mpmm(int* a, int* b, int* c, int A, int B, int C, bool b_major){ // 0 as row-major, 1 as col-major
//  open_devices(0, 1);
//  float scale = 1; // dummy
// design: use maxpooling2D to implement mpmm by transforming non-negative integer weights : yi = max(for all x) - xi

/*
for Aik, Bkj are non-negative integer weights
min(Aik+Bkj)   = min(Ai0+B0j, Ai1+Bj, ..., Ai(n-1)+B(n-1)j)
max(A'ik+B'kj) = max(A'i0+B'0j, A'i1+B'1j, ..., A'i(n-1)+B'(n-1)j) 
               = max(M-Ai0+M-B0j, ...)
               = 2M + max(-Aik-Bkj)
               = 2M + min(Aik+Bkj)
estimate: 1Kx1K max()=> 3.679 ms
          1K max()   =>

*/

//}

int gptpu_mm(int* a, int* b, int* c, int A, int B, int C, bool b_major){ // 0 as row-major, 1 as col-major
  open_devices(0, 1);
  float scale = 1; // dummy
//TODO; exact mode using 256mm block design
// ===== exact mode version (mm256conv) =======================================================================
//  return gptpu_256mm(a, b, c, A, B, C, b_major); 
// ===== exact mode version (mm256conv) =======================================================================

  //return gptpu_mm2mul(a, b, c, A, B, C, b_major);
// ===== baseline version (mm2conv) ===========================================================================
  return gptpu_mm2conv(a, b, c, A, B, C, b_major);
// ============================================================================================================

// ===== Note: the folowing is mv_stacking version, slow but workable =========================================
//  // CASE 1: for a is row-major, b is col-major:
//  // a as row-major placed matrix, b as vectors, and get col-major c before postprocessing
//  // CASE 2: for both a and b are row-major:  // use the fact that: AB = (BT)(AT)
//  // b as col-major placed matrix, a as vectors, and get col-major c before posrprocessing
//  if(b_major == true){ // b as col-major
//    return gptpu_block(a, b, true, c, A, B, C, scale, mm_model, 0); 
//  }else{               // b as row-major
//    return gptpu_block(b, a, false, c, A, B, C, scale, mm_model, 0); // a[A][B](row-major), b[B][C](col-major), c[A][C](row-major) 
//  }
// ===== Note end =============================================================================================
}

int gptpu_tanh(int* a, int* c, int A, int B){
  open_devices(0, 1);
  int* b; // dummy
  float scale = 1;
  gptpu_main(a, b, c, A, B, 0/*blk_row*/, 0/*blk_col*/, 0/*start_i*/, 0/*start_j*/, scale, tanh_model, 0, false);
}

int gptpu_relu(int* a, int* c, int A, int B){
  open_devices(0, 1);
  int* b; // dummy
  float scale = 1;
  gptpu_main(a, b, c, A, B, 0/*blk_row*/, 0/*blk_col*/, 0/*start_i*/, 0/*start_j*/, scale, relu_model, 0, false);
}

int gptpu_add(int* a, int* b, int* c, int A, int B){
  open_devices(0, 1);
  set_chunk_size(add_chunk_size);
  float scale = 1;
  gptpu_main(a, b, c, A, B, 0/*blk_row*/, 0/*blk_col*/, 0/*start_i*/, 0/*start_j*/, scale, add_model, 0, false);
}

int gptpu_sub(int* a, int* b, int* c, int A, int B){
  open_devices(0, 1);
  float scale = 1;
  gptpu_main(a, b, c, A, B, 0/*blk_row*/, 0/*blk_col*/, 0/*start_i*/, 0/*start_j*/, scale, sub_model, 0, false);
}

int gptpu_mul(int* a, int* b, int* c, int A, int B){
  open_devices(0, 1);
  set_chunk_size(mul_chunk_size);
  float scale = 1;
  gptpu_main(a, b, c, A, B, 0/*blk_row*/, 0/*blk_col*/, 0/*start_i*/, 0/*start_j*/, scale, mul_model, 0, false);
}

int gptpu_mean(int* a, int* c, int A, int B){
  open_devices(0, 1);
  set_chunk_size(mean_chunk_size);
  int* b ; // dummy
  float scale = 1;
  gptpu_main(a, b, c, A, B, 0/*blk_row*/, 0/*blk_col*/, 0/*start_i*/, 0/*start_j*/, SCALE, mean_model, 0, false);
}

int gptpu_max(int* a, int* c, int A, int B){
  open_devices(0, 1);
  int* b ; // dummy
  float scale = 1;
  gptpu_main(a, b, c, A, B, 0/*blk_row*/, 0/*blk_col*/, 0/*start_i*/, 0/*start_j*/, scale, max_model, 0, false);
}

int gptpu_maxpool(int* a, int* c, int A, int B){
  open_devices(0, 1);
  float scale = 0; // dummy
  int* b; //dummy
  gptpu_main(b, a, c, A, B, 0/*blk_row*/, 0/*blk_col*/, 0/*start_i*/, 0/*start_j*/, scale, maxpool_model, 0, false);
  return 0;
}

int gptpu_crop(int* a, int* c, int A, int B, int blk_row, int blk_col, int start_i, int start_j, bool transpose){
  open_devices(0, 1);
  float scale = 0; // dummy
  int* b; // dummy
  bool ret = crop_check(A, B, blk_row, blk_col, start_i, start_j); 
  if(ret == true){
    gptpu_main(a, b, c, A, B, blk_row, blk_col, start_i, start_j, scale, crop_model, 0, transpose);
  }else{
    return 0;
  }
}
int gptpu_ext(int* a, int* c, int blk_row, int blk_col, int A, int B, int start_i, int start_j, bool transpose){
  open_devices(0, 1);
  float scale = 0; // dummy
  int* b; // dummy
  bool ret = crop_check(A, B, blk_row, blk_col, start_i, start_j); // the same logic of checking as cropping
  if(ret == true){
    gptpu_main(a, b, c, A, B, blk_row, blk_col, start_i, start_j, scale, ext_model, 0, transpose);
  }else{
    return 0;
  }
}

// For comparison
int gptpu_tf_vs(int* a, int* c, int A, float scale){
  return 0;
}

int gptpu_tf_mm(int* a, int* b, int* c, int A, int B, int C, bool b_major){
  open_devices(0, 1);
  float scale = 0; // dummy
  // a[A][B]
  // b[B][C]
  // c[A][C]
  std::cout << "AxBxC: " << A << "x" << B << "x" << C << std::endl;
  std::cout << "a should be row-major; b should be col-major." << std::endl;
  if(b_major == true){
    gptpu_block(a, b, true, c, A, B, C, scale, mm_model, 1);
  }else{
    gptpu_block(b, a, false, c, A, B, C, scale, mm_model, 1);
  }
  return 0;
}

int gptpu_tf_mv(int* a, int* b, int* c, int A, int B){
  open_devices(0, 1);
  float scale = 0; // dummy
  gptpu_block(a, b, true, c, A , B, 1, scale, mv_model, 1);
  return 0;
}

int gptpu_tf_bmv(int* b, int* a, int* c, int A, int B){
  open_devices(0, 1);
  float scale = 0; // dummy
  gptpu_main(b, a, c, A, B, 0/*blk_row*/, 0/*blk_col*/, 0/*start_i*/, 0/*start_j*/, scale, bmv_model, 1, false);
  return 0;
}

int gptpu_tf_log(int* a, int* c, int A, int B){  
  open_devices(0, 1);
  float scale = 0;
  int* b;
  gptpu_main(a, b, c, A, B, 0/*blk_row*/, 0/*blk_col*/, 0/*start_i*/, 0/*start_j*/, scale, log_model, 1, false);
  return 0;
}

int gptpu_tf_add(int* a, int* b, int* c, int A, int B){
  open_devices(0, 1);
  float scale = 0; //dummy
  gptpu_main(a, b, c, A, B, 0/*blk_row*/, 0/*blk_col*/, 0/*start_i*/, 0/*start_j*/, scale, add_model, 1, false);
  return 0;
}

int gptpu_tf_sub(int* a, int* b, int* c, int A, int B){
  open_devices(0, 1);
  float scale = 0 ; // dummy
  gptpu_main(a, b, c, A, B, 0/*blk_row*/, 0/*blk_col*/, 0/*start_i*/, 0/*start_j*/, scale, sub_model, 1, false);
  return 0;
}

int gptpu_tf_mul(int* a, int*b, int* c, int A, int B){
  return 0;
}

int gptpu_tf_maxpool(int* a, int* c, int A, int B){
  open_devices(0, 1);
  float scale = 0; // dummy
  int* b; //dummy
  gptpu_main(b, a, c, A, B, 0/*blk_row*/, 0/*blk_col*/, 0/*start_i*/, 0/*start_j*/, scale, maxpool_model, 1, false);
  return 0;
}

int gptpu_tf_mean(int* a, int* c, int A , int B){
  return 0;
}

int gptpu_tf_crop(int* a, int* c, int A, int B, int blk_row, int blk_col, int start_i, int start_j, bool transpose){
  open_devices(0, 1);
  float scale = 0; // dummy
  int *b; // dummy
  gptpu_main(a, b, c, A, B, blk_row, blk_col, start_i, start_j, scale, crop_model, 1, transpose);  
  return 0;
}

void quantize_array_double2int(double* in, uint8_t* out, int size, uint8_t& mean, double& scale){
// for each out[i], in[i] = (out[i] - mean ) / scale
  double in_max = DBL_MIN;
  double in_min = DBL_MAX;
  for(int i = 0 ; i < size ; i++){
    if(in[i] > in_max){ in_max = in[i]; }
    if(in[i] < in_min){ in_min = in[i]; }
  }
  std::cout << __func__ << ": in_max: " << in_max << ", in_min: " << in_min << std::endl;
  if((in_max - in_min) != 0){
    scale = ((double)UCHAR_MAX/*255.0 - 0.0*/) / (in_max - in_min);
    mean  = (-1.0) * (int)(in_min * scale);
    for(int i = 0 ; i < size ; i++){
      out[i] = (uint8_t)(((in[i] - in_min)/(in_max - in_min)) * (double)UCHAR_MAX);
    }
  }else{
    scale = UCHAR_MAX / in_max;
    mean = 0;
    for(int i = 0 ; i < size ; i++){
      out[i] = UCHAR_MAX;
    }
  }
}

struct quantize_params{
  uint8_t mean;
  double scale;
};

void gptpu_pagerank(double* w, double* in_rank, double* out_rank, int size, int iter){
// scale down w and in_rank
  std::cout << __func__ << ": size: " << size << ", iter: " << iter << std::endl;
  quantize_params w_quantize_params;
  quantize_params in_rank_quantize_params;
  quantize_params out_rank_quantize_params;
  quantize_params per_layer_qunatize_params; // need oracle 
  uint8_t* w_int        = (uint8_t*)malloc(size*size*sizeof(uint8_t));
  uint8_t* in_rank_int  = (uint8_t*)malloc(size*sizeof(uint8_t));
  uint8_t* out_rank_int     = (uint8_t*)malloc(size*sizeof(uint8_t));
  double* tmp_rank_int     = (double*)malloc(size*sizeof(double));
  //int iter = (size == 1024)?5:((size == 2048)?4:(size == 4096)?4:(size == 8192)?3:1/*default value*/);  
//  if(iter == 1){
//    std::cout << __func__ << ": untested size = " << size << ", exit." << std::endl;
//    exit(0);
//  }
  //std::string command, model_path = PWD+"/data/imv_model_size_"+itoa(size)+"_iter_"+itoa(iter)+"_quant_edgetpu.tflite";
  std::string command, model_path = PWD+"/data/pagerank_1K_iter_"+itoa(iter)+"_edgetpu.tflite";

  quantize_array_double2int(      w,       w_int, size*size,       w_quantize_params.mean,       w_quantize_params.scale);
  quantize_array_double2int(in_rank, in_rank_int,      size, in_rank_quantize_params.mean, in_rank_quantize_params.scale);


  for(int i = 0 ; i < 10 ; i++){
    printf("w[%2d]: %f\n", i, w[i]);
  }
  for(int i = 0 ; i < 10 ; i++){
    printf("w_int[%2d]: %d\n", i, w_int[i]);
  }
  
  for(int i = 0 ; i < 10 ; i++){
    printf("in_rank[%2d]: %f\n", i, in_rank[i]);
  }
  for(int i = 0 ; i < 10 ; i++){
    printf("in_rank_int[%2d]: %d\n", i, in_rank_int[i]);
  }
  save_weight_uint8(w_int, PWD+"/data/tmp_pagerank_w_"+itoa(size)+".txt", size*size); 
// TODO: binary decipher
//  command = "python "+PWD+"/src/create_model.py --model=imv_model --in_size="+itoa(size)+" --out_size="+itoa(size)+" --ITER="+itoa(iter)+" --outfile="+model_path;
//  system(command.c_str()); 
  printf("CPU uint8 simulation, for precison metrics only\n");
  
  int sum = 0;  
  for(int i = 0 ; i < size ; i++){
    out_rank_int[i] = in_rank_int[i];
  }
  for(int it = 0 ; it < iter ; it++){
    for(int i = 0 ; i < size ; i++){
      sum = 0;
      for(int j = 0 ;  j <  size ; j++){
        sum += w_int[i*size+j] * out_rank_int[j];
      }
      tmp_rank_int[i] = sum;
    }
    for(int i = 0 ; i < 10 ; i ++){
      printf("iter:%d:  tmp_rank_int[%2d]: %f\n", it, i, tmp_rank_int[i]);
    }
    quantize_array_double2int(tmp_rank_int, out_rank_int, size, out_rank_quantize_params.mean, out_rank_quantize_params.scale);
    printf("scale: %f\n", out_rank_quantize_params.scale);
    for(int i = 0 ; i < 10 ; i ++){
      printf("iter:%d:  out_rank_int[%2d]: %d\n", it, i, out_rank_int[i]);
    }
  }

  long long int total = 0;
  for(int i = 0 ; i < size ; i++){
    total += out_rank_int[i];
  }
  printf("total: %lld\n", total);
  for(int i = 0 ; i < size ; i++){
    out_rank[i] = (double)out_rank_int[i] / (double)total;
  }


//===== run model
  set_dev_cnt(1);
  open_devices(0, 1);
  build_model(model_path, 0/*model_id*/);
  build_interpreter(0/*tpu_id*/, 0/*model_id*/);
  float the_scale = 1 / (w_quantize_params.scale * in_rank_quantize_params.scale );
  printf("the set scale: %f\n", the_scale);
  printf("weight scale: %f, in scale: %f\n", w_quantize_params.scale, in_rank_quantize_params.scale);
//  set_scale(the_scale);  
//  populate_input_uint8(in_rank_int, size, 0/*mdoel_id*/);
  timing s = clk::now();
//  invoke_model(0/*model_id*/, 0/*iter*/);
  timing e = clk::now();
  double us = std::chrono::duration_cast<std::chrono::nanoseconds>(e-s).count()/1000.0;
  std::cout << __func__ << ": invoke time: " << us << " (us)." << std::endl;
//  simple_populate_output(out_rank_int, 0, 0);
  for(int i = 0 ; i < 10 ; i ++){
    printf("out_rank[%2d]: %f\n", i, out_rank[i]);
  }
}

