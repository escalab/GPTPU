#include <stdio.h>
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
#include "offset.h"
#include "make_model.h"
//##include <dense.h> // created within edgetpu/
#include <complex>
#include <float.h>
#include <sys/sysinfo.h>
#include <math.h>
#include <omp.h>

typedef std::chrono::time_point<std::chrono::high_resolution_clock> timing;
typedef std::chrono::high_resolution_clock clk;

extern std::string temp_dir;
extern std::string template_path;

union scaling{
  float f;
  char c[sizeof(float)];
};

int reStride(int i){ // an special re-index func for 256x256 mm
  int n;
  if(i < 0 || i >= 64){
     std::cout << __func__ << ": special case rule violation" << std::endl;
     exit(0);
  }
  int s = i/8;
  int t = i%8;
  int x = t/2;
  int y = t%2;
  n = (x*16 + y) + s*2;  
//  std::cout << __func__ << ", n: " << n << ", i: " << i << std::endl;
  return n;
}

int create_mm2conv_tflite(const std::string& template_path, std::vector<Flatbufs>& flatbufs,/*const std::string& out_path*/ int model_id, char* data_array, int blk_A, int blk_B, int blk_C, float SCALE, int chunk_num/*for mm256conv use*/){
  std::cout << __func__ << ": template_path: " << template_path << ", blk_A: " << blk_A << " , blk_B: " << blk_B << ", blk_C: " << blk_C << ", SCALE: " << SCALE << ", chunk_num: " << chunk_num << std::endl;
  int temp_fd = open(template_path.c_str(), O_RDONLY);
  //int out_fd  = open(out_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0664); 
  char *src, *dst;
  struct stat st_temp;
  unsigned long long int out_file_size;
  if(temp_fd < 0){
    std::cout << __func__ << ": temp file opening fail. " << template_path << std::endl;
    exit(0);
  }
  if(out_fd < 0){
    std::cout << __func__ << ": out file opening fail. " << out_path << std::endl;
    exit(0);
  }
  unsigned long long int size; // the matrix length size
  unsigned long long int before_data;
  unsigned long long int after_data;
  unsigned long long int ext_len;
  unsigned long long int scale_loc;
  unsigned long long int scale_stride_len; 
  unsigned long long int scale_num; 
  unsigned long long int factor;
  unsigned long long int *f_locs;
  char* cf = (char*)malloc(8*sizeof(char)); // char of factor
  cf[0] = 0xe0;  
  cf[1] = 0xc1;
  cf[2] = 0xff;
  cf[3] = 0x03;
  cf[4] = 0x00;
  cf[5] = 0x00;
  cf[6] = 0xf0;
  cf[7] = 0x07;
  if(blk_A == blk_B && blk_B == blk_C && blk_C == 1024){
    size             = 1024; 
    before_data      = offset::mm2conv::oneKs::before_data;
    after_data       = offset::mm2conv::oneKs::after_data;
    ext_len          = offset::mm2conv::oneKs::ext_len; 
    scale_loc        = offset::mm2conv::oneKs::scale_loc;
    scale_stride_len = offset::mm2conv::oneKs::scale_stride_len; 
    scale_num        = offset::mm2conv::oneKs::scale_num; 
    factor           = offset::mm2conv::oneKs::f_loc;
    cf[0] = 0xe0;  
    cf[1] = 0xc1;
    cf[2] = 0xff;
    cf[3] = 0x03;
    cf[4] = 0x00;
    cf[5] = 0x00;
    cf[6] = 0xf0;
    cf[7] = 0x07;
  }else if(blk_A == 1 && blk_B == blk_C && blk_C == 1024){
    size             = 1024; 
    before_data      = offset::mm2conv::oneKmv::before_data;
    after_data       = offset::mm2conv::oneKmv::after_data;
    ext_len          = offset::mm2conv::oneKmv::ext_len; 
    scale_loc        = offset::mm2conv::oneKmv::scale_loc;
    scale_stride_len = offset::mm2conv::oneKmv::scale_stride_len; 
    scale_num        = offset::mm2conv::oneKmv::scale_num; 
  }else if(blk_A == blk_B && blk_B == blk_C && blk_C == 256){
    size             = 256; 
    before_data      = offset::mm2conv::mm_256::before_data;
    after_data       = offset::mm2conv::mm_256::after_data;
    ext_len          = offset::mm2conv::mm_256::ext_len; 
    scale_loc        = offset::mm2conv::mm_256::scale_loc;
    scale_stride_len = offset::mm2conv::mm_256::scale_stride_len; 
    scale_num        = offset::mm2conv::mm_256::scale_num; 
    factor           = offset::mm2conv::mm_256::f_loc; 
    cf[0] = 0xe0;  
    cf[1] = 0xc2;
    cf[2] = 0xff;
    cf[3] = 0x23;
    cf[4] = 0x10;
    cf[5] = 0x10;
    cf[6] = 0x70;
    cf[7] = 0x07;
//    cf[0] = 0xe0;  
//    cf[1] = 0xc2;
//    cf[2] = 0xff;
//    cf[3] = 0x03;
//    cf[4] = 0x00;
//    cf[5] = 0x00;
//    cf[6] = 0xf0;
//    cf[7] = 0x07;
  }else if(blk_A == blk_B && blk_B == blk_C && blk_C == 128){
    size             = 128; 
    before_data      = offset::mm2conv::mm_128::before_data;
    after_data       = offset::mm2conv::mm_128::after_data;
    ext_len          = offset::mm2conv::mm_128::ext_len; 
    scale_loc        = offset::mm2conv::mm_128::scale_loc;
    scale_stride_len = offset::mm2conv::mm_128::scale_stride_len; 
    scale_num        = offset::mm2conv::mm_128::scale_num; 
    cf[0] = 0xe0;  
    cf[1] = 0xc1;
    cf[2] = 0xff;
    cf[3] = 0x03;
    cf[4] = 0x00;
    cf[5] = 0x00;
    cf[6] = 0xf0;
    cf[7] = 0x07;
  }else if(blk_A == 4096 && blk_B == 256 && blk_C == 4096){ //for mm256conv exact mode design
    before_data      = offset::mm256conv::b16::before_data;
    after_data       = offset::mm256conv::b16::after_data;
    ext_len          = offset::mm256conv::b16::ext_len; 
    scale_loc        = offset::mm256conv::b16::scale_loc;
    scale_stride_len = offset::mm256conv::b16::scale_stride_len; 
    scale_num        = offset::mm256conv::b16::scale_num; 
    f_locs           = offset::mm256conv::b16::f_locs;
    cf[0] = 0xe0;  
    cf[1] = 0xc2;
    cf[2] = 0xff;
    cf[3] = 0x03;
    cf[4] = 0x00;
    cf[5] = 0x00;
    cf[6] = 0xf0;
    cf[7] = 0x07;
  }else if(blk_A == 2048 && blk_B == 256 && blk_C == 2048){ //for mm256conv exact mode design
    before_data      = offset::mm256conv::b8::before_data;
    after_data       = offset::mm256conv::b8::after_data;
    ext_len          = offset::mm256conv::b8::ext_len; 
    scale_loc        = offset::mm256conv::b8::scale_loc;
    scale_stride_len = offset::mm256conv::b8::scale_stride_len; 
    scale_num        = offset::mm256conv::b8::scale_num; 
    f_locs           = offset::mm256conv::b8::f_locs;
    cf[0] = 0xe0;  
    cf[1] = 0xc1;
    cf[2] = 0xff;
    cf[3] = 0x03;
    cf[4] = 0x00;
    cf[5] = 0x00;
    cf[6] = 0xf0;
    cf[7] = 0x07;
  }else{
    std::cout << __func__ << ": shape(" << blk_A << "x" << blk_B << "x" << blk_C << ") not supported yet." << std::endl;
    exit(0);
  }
  unsigned long long int blk_len     = 4;    //four consecutive elements in a row
  unsigned long long int stride_len  = size/blk_len/*256*/;  // how long is the size between two blks' starting
  unsigned long long int stride_zero_len = 0x100; // TODO: so far all are the same, make as a parameter if differ for others
  unsigned long long int num_strips  = 64 /*stride_len/blk_len*/; // how many consecutive striding blks in this section (between sections, the zeros are longer)
  unsigned long long int section_len = blk_len*num_strips*stride_len+/*zeros*/stride_zero_len; // 2^16 = 1024x64
// for each stride length, it can host 256/4 = 64 different strips for consecutive blocks

  int num_section = ((size*size) / section_len)+1; // 2^4 = 16
  char* zeros     = (char*)calloc(stride_zero_len, sizeof(char));
  char* ext_zeros = (char*)calloc(ext_len, sizeof(char));
  fstat(temp_fd, &st_temp);
  if(blk_A == 4096 && blk_B == 256 && blk_C == 4096){
    out_file_size = 0x117228;
  }else if(blk_A == 2048 && blk_B == 256 && blk_C == 2048){
    out_file_size = 0x8d228;
  }else{
    out_file_size = st_temp.st_size + size*size + stride_zero_len*num_section + ext_len;
  }
//  std::cout << "file_size: 0x" << std::hex << out_file_size << " = 0x" << st_temp.st_size << " + 0x" << size << "* 0x" << size << " + 0x" << stride_zero_len << "* 0x" << num_section << " + 0x" << ext_len << std::dec << std::endl;
  //std::cout << __func__ << ": st_temp.st_size: " << st_temp.st_size << std::endl;
  src = static_cast<char*>(mmap(NULL, st_temp.st_size, PROT_READ, MAP_SHARED, temp_fd, 0));
  assert(src != MAP_FAILED);
  if(flatbufs.size() < model_id + 1){
    flatbufs.resize(model_id + 1);
  }
  flatbufs[model_id].buf  = (char*) malloc(out_file_size*sizeof(char));
  flatbufs[model_id].size = out_file_size;
  if(flatbufs[model_id].buf == nullptr){
    std::cout << __func__ << ": malloc flatbufs[" << model_id << "].buf with size " << out_file_size << " fails, exit." << std::endl;
    exit(0);
  }
  dst = flatbufs[model_id].buf;
  //dst = static_cast<char*>(mmap(NULL, out_file_size, PROT_WRITE, MAP_SHARED, out_fd, 0));
  //assert(dst != MAP_FAILED);
  //if(ftruncate(out_fd, out_file_size) != 0){
  //  std::cout << __func__ <<  ": out file ftruncate fail." << std::endl;
  //}
// =====  copy the first section that before data section =====
  memcpy(dst, src, before_data);
// ===== copy data section =====
//std::cout << __func__ << ", num_section: " << num_section << ", num_strips: " << num_strips << ", stride_len: " << stride_len << ", blk_len: " << blk_len << ", blk_A: " << blk_A << ", blk_B: " << blk_B << ", blk_C: " << blk_C << std::endl;
//TODO; if 256x256 is an exception, then it could be just another general thing out of unknown rules
  if((blk_A == blk_B && blk_B == blk_C && blk_C == 256)){ // handle found expection
    for(int s = 0 ; s < num_section; s++){
    // copy the each 64 x 1024 elements
      for(int i  = 0 ; i < num_strips ; i++){ // 64 strips, and striding (with length = 4) on those strips. (so complicated)
        for(int j = 0 ; j < stride_len ; j++){ // 1024 / 4 = 256
          int n = reStride(j);
          memcpy(dst+before_data+(blk_len*i)+(stride_zero_len*j)+(s*section_len), 
                 data_array+(blk_len*stride_len*i)+(blk_len*n)+s*(blk_len*stride_len*num_strips), 
                 blk_len*sizeof(char));
        }
      }  
    // fill in zeros between two sections
      memcpy(dst+before_data+blk_len*num_strips+stride_zero_len*stride_len+s*section_len, zeros, stride_len*sizeof(char));
    }
  }else if(blk_A == 4096 && blk_B == 256 && blk_C ==4096){
    blk_len = 4;
    stride_len = 0x100; // == 256
/*
the data sequence:

b,n: (0,0) (1,16) (2, 16*2) (3, 16*3) (4,1) (5, 16*1+1) (6, 16*2+1)
b = 0,1,2,...,63
n = f(b) = (b%4)*16+(b/4)

------------------         b  n             
0x320c: 0x00 01 02 03      0  0
0x330c: 0x10 11 12 13      1  4
...
        0xf0 f1 f2 f3     15 60
------------------ 16*256=4096=0x1000
0x420c: 0x04 05 06 07     16  1
        0x14 15 16 17     17  5  
...
        0xf4 f5 f6 f7     31 61
------------------
        0x08 09 0a 0b     32  2
...
        0xf8 f9 fa fb     47 62
------------------
        0x0c 0d 0e 0f     48  3
...
        0xfc fd fe ff     63 63
------------------
*/
    int blk_cnt = 64; // 0xf * 4
    num_strips = 64;
    int num_curr_j_sections = 4; // curr_j = num_strips * num_curr_j_sections
//blk_len    = 4
//stride_len = 0x100
// 64 blocks in length (blk_r) within (curr_j) blocks are consecutive
// every 64 blocks are totally consecutive
// every chunk_idx for 256x256 mm are consecutive
//    for(int i = 0 ; i < 256*256*16 ; i++){
//      if(data_array[i] != 0){
//        int x = i / (256*16);
//        int y = i % (256*16);
//        int z = y/256;
//         y = y % 256;
//        std::cout << __func__ << ": data_array " << i << " is: " << (unsigned)data_array[i] << ",x: " << x << ", y: " << y << ", z: " << z << std::endl;
//      }
//    }
    int cnt = 0; 
    for(int w = 0; w < chunk_num ; w++){
      for(int j = 0 ; j < num_curr_j_sections/*4*/ ; j++){
        for(int i = 0 ; i < num_strips/*64*/ ; i++){
          for(int b = 0 ; b < blk_cnt/*64*/ ; b++){
            int n = (b%4)*16+(b/4);
            memcpy(dst+before_data+(w*num_curr_j_sections+j)*(blk_cnt*(stride_zero_len+blk_len))+n*(stride_len)+i*(blk_len), 
                   data_array+w*(256)+(j*num_strips+i)*(256*chunk_num)+b*blk_len, 
                   //data_array+cnt*blk_len/*+ data in col_major with shape (256, 4096)*/, 
                   blk_len*sizeof(char));
            cnt++;
          }
        }
      }
    }
  }else if(blk_A == 2048 && blk_B == 256 && blk_C ==2048){
    for(int w = 0 ; w < chunk_num ; w++){
      for(int idx_r = 0 ; idx_r < blk_B ; idx_r++){
        for(int idx_c = 0 ; idx_c < (blk_C/chunk_num)/*256*/ ; idx_c++){
          memcpy(dst+before_data+w*(0x10400)+((idx_r/4)*0x100)+(idx_r%4)+(idx_c%64)*4+(idx_c/64)*0x4100,
                 data_array+w*(blk_B)+idx_r+idx_c*(blk_C),
                 1*sizeof(char)); 
        }
      }
    }
  }else{ // all other general cases
    for(int s = 0 ; s < num_section; s++){
    // copy the each 64 x 1024 elements
      for(int i  = 0 ; i < num_strips ; i++){ // 64 strips
        for(int j = 0 ; j < stride_len ; j++){ // 1024 / 4 = 256
          memcpy(dst+before_data+(blk_len*i)+(stride_zero_len*j)+(s*section_len), 
                 data_array+(blk_len*stride_len*i)+(blk_len*j)+s*(blk_len*stride_len*num_strips), 
                 blk_len*sizeof(char));
        }
      }  
    // fill in zeros between two sections
      memcpy(dst+before_data+blk_len*num_strips+stride_zero_len*stride_len+s*section_len, zeros, stride_len*sizeof(char));
    }
  }
// ===== fill in extended zeros =====
  unsigned long long int ext_loc;
  if(blk_A == 4096 && blk_B == 256 && blk_C == 4096){
    ext_loc = 0x10710c;
  }else if(blk_A == 2048 && blk_B == 256 && blk_C == 2048){
    ext_loc = 0x8510c;
  }else{
    ext_loc = before_data + num_section * section_len;
  }
  memcpy(dst+ext_loc, ext_zeros, ext_len);
// ===== copy the section section after data =====
  unsigned long long int sec_start = ext_loc + ext_len;
  //std::cout << "ext_loc = 0x" << std::hex << ext_loc << " = 0x" << before_data << " + 0x" << num_section << " * 0x" << section_len << std::dec << std::endl;
//  std::cout << "after_data = 0x" << std::hex << after_data << " = (ext_loc)0x" << ext_loc << " + (ext_len)0x" << ext_len << std::dec << std::endl;
  assert(after_data == ext_loc + ext_len);
  memcpy(dst+after_data, src+before_data, st_temp.st_size - before_data);
// ===== set scale(s), some shapes need duplicated scales =====
  union scaling the_scale;
  the_scale.f = SCALE;
  for(int i = 0 ; i < scale_num ; i++){
//    memcpy(dst+scale_loc+(i*scale_stride_len), scale, 3*sizeof(char) );
//    std::cout << __func__ << ":scale offset: " << std::hex << scale_loc+(i*scale_stride_len) << ", scal_loc: " << scale_loc << ", i: " << i << ", stride_len: " << stride_len << std::dec << std::endl;
    memcpy(dst+scale_loc+(i*scale_stride_len), reinterpret_cast<char*>(&the_scale.c[0]), sizeof(float) );
  }
// ===== set the factor ==========================
  if((blk_A == 4096 && blk_B == 256 && blk_C == 4096) ||
     (blk_A == 2048 && blk_B == 256 && blk_C == 2048)){ // handle found exception for mm256conv exact mode design
    for(int i = 0 ; i < (sizeof(f_locs)/sizeof(f_locs[0])) ; i ++){
      memcpy(dst+f_locs[i], reinterpret_cast<char*>(&cf[0]), 8*sizeof(char));
    }
  }else{ // all other general case
    memcpy(dst+factor, reinterpret_cast<char*>(&cf[0]), 8*sizeof(char));
  }


// TODO: developing
//  union scaling the_scale;
//  the_scale.f = SCALE;
//  memcpy(dst+scale_loc, reinterpret_cast<char*>(&the_scale.c[0]), sizeof(float));
//  std::cout << "the SCALE is: " << SCALE << std::endl;
//  std::cout << std::hex << "scale in binary: " << (unsigned)*(dst+scale_loc) << " " << (unsigned)*(dst+scale_loc+1) << " " << (unsigned)*(dst+scale_loc+2) << " " << (unsigned)*(dst+scale_loc+3) << std::dec << std::endl;
//  std::cout << "scale location in binary: 0x" << std::hex << scale_loc << " = " << std::dec << scale_loc << std::endl;
/*
for 1024x1024x1024:

i=0,...,1023:
0x320c  ~ 0x320f
0x330c  ~ 0x330f
...
0x1310c ~ 0x1310f

i=1024,...,2047:
0x3210  ~ 0x3213
0x3310  ~ 0x3313
...
0x13110 ~ 0x13113

i=2048,...,3072:
0x3214  ~ 0x3217
0x3314  ~ 0x3317
...
0x13114 ~ 0x13117
*/
  munmap(src, st_temp.st_size);
  munmap(dst, out_file_size);
  close(temp_fd);
  //close(out_fd);
  free(cf);
  return 0;
}

int create_dense_tflite(const std::string& out_path, int A, int B, char* data, char* scaling, char* bias_scaling, char* zero_point, const std::string& data_type, int verbose){
// TODO: implement int8 
  if(!data_type.compare("int8")){ std::cout << __func__ << ": data type " << data_type << " not implemented for mv_stacking method. Try use mm2conv instead." << std::endl; exit(1); }
  timing setup_start = clk::now();
  int temp_fd = open(template_path.c_str(), O_RDONLY);
  int out_fd  = open(out_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0664); 
  char *src, *dst;
  struct stat st_temp;
  unsigned long long int out_file_size;
  uint32_t size_buffer[2] = {(uint32_t)A, (uint32_t)B}, fourOut[1] = {4 * size_buffer[0]}; // 0 for input, 1 for output  
  unsigned long long int bias_size = (unsigned long long int)(size_buffer[0]*4);
  unsigned long long int data_size = (unsigned long long int)(A * B);
  unsigned long long int curr_len = 0;  

  if(temp_fd < 0){
    std::cout << "create_new_tflite: temp file opening fail. " << template_path << std::endl;
  }
  if(out_fd < 0){
    std::cout << "create_new_tflite: out file opening fail. " << out_path << std::endl;
  }

  // get file size
  fstat(temp_fd, &st_temp);
  out_file_size = st_temp.st_size + data_size + bias_size;

  // mmap
  src = static_cast<char*>(mmap(NULL, st_temp.st_size, PROT_READ, MAP_SHARED, temp_fd, 0));
  assert(src != MAP_FAILED);
  dst = static_cast<char*>(mmap(NULL, out_file_size, PROT_WRITE, MAP_SHARED, out_fd, 0));
  assert(dst != MAP_FAILED);
  if(ftruncate(out_fd, st_temp.st_size + data_size + bias_size) != 0){
    std::cout << "out file ftruncate fail." << std::endl;
  }
  timing setup_end = clk::now();

  unsigned long long int dense_data, dense_4out, dense_bias;

  if(A == 1){
    dense_data = offset::dense_data + 0x4;
    dense_4out = offset::dense_4out + 0x4;
    dense_bias = offset::dense_bias + 0x4;
  }else{
    dense_data = offset::dense_data;
    dense_4out = offset::dense_4out;
    dense_bias = offset::dense_bias;
  }

  timing fst_start = clk::now();
// ========== first section =========u
  //copy first section (before data) from tempolate
  memcpy(dst, src, dense_data);

  // rewrite data size
  memcpy(dst+offset::dense_size , src+offset::dense_size, sizeof(size_buffer));
// ========== first section ==========
  timing fst_end   = clk::now();

  // append data
  timing data_start = clk::now();
  memcpy(dst+dense_data, data, data_size);
  timing data_end   = clk::now();
  
  timing snd_start = clk::now();
// ========== second section ==========
  // copy second section (between data and bias) from tempolate
  curr_len = dense_data + data_size; 
  memcpy(dst+curr_len, src+dense_data, dense_bias - dense_data);

  // rewrite 4 out size
  memcpy(dst+dense_4out+data_size, reinterpret_cast<char*>(&fourOut[0]), sizeof(uint32_t));
// ========== second section ==========
  timing snd_end   = clk::now();
  // append bias
  timing zero_start = clk::now();
  char *zeros = new char[size_buffer[0] * 4]();

  curr_len = dense_bias + data_size;
  memcpy(dst+curr_len, zeros, sizeof(zeros));
  timing zero_end   = clk::now();

  timing third_start = clk::now();
// ========== third section ==========
  // copy third section (everything after actual bias data) from tempolate
  curr_len += bias_size;
  memcpy(dst+curr_len, src+dense_bias, st_temp.st_size - dense_bias);

  // rewrite dense_matmul = output size
  // TODO : BUG HERE
  curr_len = out_file_size - offset::dense_matmul;
  memcpy(dst+curr_len, reinterpret_cast<char*>(&size_buffer[1]), sizeof(uint32_t));

  // rewrite dense_flatten = input size
  curr_len = out_file_size - offset::dense_flatten;
  memcpy(dst+curr_len, reinterpret_cast<char*>(&size_buffer[1]) , sizeof(uint32_t));

 // rewrite dense_meta
  curr_len = out_file_size - offset::dense_meta;
  memcpy(dst+curr_len, reinterpret_cast<char*>(&size_buffer[0]), sizeof(size_buffer) );

  // rewrite zero point
  curr_len = out_file_size - offset::dense_zerop;
  memcpy(dst+curr_len, &zero_point[0], sizeof(char));

  // rewrite scale
  curr_len = out_file_size - offset::dense_scale;
  memcpy(dst+curr_len, &scaling[0], sizeof(float));

  // rewrite bias size
  curr_len = out_file_size - (offset::dense_bias_size);
  memcpy(dst+curr_len, reinterpret_cast<char*>(&size_buffer[0]), sizeof(uint32_t));

  // rewrite bias scale
  curr_len = out_file_size - offset::dense_bias_scale;
  memcpy(dst+curr_len, &bias_scaling[0], sizeof(float));

// ========== third section ==========
  timing third_end   = clk::now();
  // closing
  timing close1_start  = clk::now();
  munmap(src, st_temp.st_size);
  munmap(dst, out_file_size);
  timing close1_end    = clk::now();
  timing close2_start  = clk::now();
  close(temp_fd);
  close(out_fd);
  timing close2_end    = clk::now();
  timing close3_start  = clk::now();
  //delete src, dst;
  delete [] zeros;
  timing close3_end    = clk::now();

  if(verbose >= 0){
    std::cout << "+===== create_dense_tflite timing breakdown =====" << std::endl;
    std::cout << "| setup         : " << std::chrono::duration_cast<std::chrono::nanoseconds>(setup_end - setup_start).count()/1000.0 << "\t(us)." << std::endl; 
    std::cout << "| first section : " << std::chrono::duration_cast<std::chrono::nanoseconds>(fst_end - fst_start).count()/1000.0 << "\t(us)." << std::endl; 
    std::cout << "| data          : " << std::chrono::duration_cast<std::chrono::nanoseconds>(data_end - data_start).count()/1000.0 << "\t(us)." << std::endl; 
    std::cout << "| second section: " << std::chrono::duration_cast<std::chrono::nanoseconds>(snd_end - snd_start).count()/1000.0 << "\t(us)." << std::endl; 
    std::cout << "| bias          : " << std::chrono::duration_cast<std::chrono::nanoseconds>(zero_end - zero_start).count()/1000.0 << "\t(us)." << std::endl; 
    std::cout << "| third section : " << std::chrono::duration_cast<std::chrono::nanoseconds>(third_end - third_start).count()/1000.0 << "\t(us)." << std::endl; 
    std::cout << "| closing1      : " << std::chrono::duration_cast<std::chrono::nanoseconds>(close1_end - close1_start).count()/1000.0 << "\t(us)." << std::endl; 
    std::cout << "| closing2      : " << std::chrono::duration_cast<std::chrono::nanoseconds>(close2_end - close2_start).count()/1000.0 << "\t(us)." << std::endl; 
    std::cout << "| closing3      : " << std::chrono::duration_cast<std::chrono::nanoseconds>(close3_end - close3_start).count()/1000.0 << "\t(us)." << std::endl; 
    std::cout << "+==============================================" << std::endl;
  }
  return 0;
}


int create_tflite(const std::string& out_path, std::string model_name, uint32_t* size_buffer, int para_cnt, int verbose){
  timing setup_start = clk::now();
  int temp_fd = open(template_path.c_str(), O_RDONLY);
  int out_fd  = open(out_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0664); 
  char *src, *dst;
  struct stat st_temp;
  unsigned long long int out_file_size;

  if(temp_fd < 0){
    std::cout << "create_new_tflite: temp file opening fail. " << template_path << std::endl;
  }
  if(out_fd < 0){
    std::cout << "create_new_tflite: out file opening fail. " << out_path << std::endl;
  }

  // get file size
  fstat(temp_fd, &st_temp);
  out_file_size = st_temp.st_size;

  // mmap
  src = static_cast<char*>(mmap(NULL, st_temp.st_size, PROT_READ, MAP_SHARED, temp_fd, 0));
  assert(src != MAP_FAILED);
  dst = static_cast<char*>(mmap(NULL, out_file_size, PROT_WRITE, MAP_SHARED, out_fd, 0));
  assert(dst != MAP_FAILED);
  if(ftruncate(out_fd, out_file_size) != 0){
    std::cout << "out file ftruncate fail." << std::endl;
  }
  timing setup_end = clk::now();

  timing size_start = clk::now();
// ========== first section ==========
  //copy whole file from template
  memcpy(dst, src, out_file_size);
  if(model_name == "crop"){
    for(int i = 0 ; i < para_cnt ; i++){
      memcpy(dst+offset::crop_offsets[i], reinterpret_cast<char*>(&size_buffer[i]), sizeof(uint32_t));
    } 
  }else if(model_name == "sub"){
    for(int i = 0 ; i < para_cnt ; i++){
      memcpy(dst+offset::sub_offsets[i],  reinterpret_cast<char*>(&size_buffer[i]), sizeof(uint32_t));
    } 
  }
  timing size_end = clk::now();
// ========== clean up ==========
  timing clean_s = clk::now();
  munmap(src, out_file_size);
  munmap(dst, out_file_size);
  close(temp_fd);
  close(out_fd);
  timing clean_e = clk::now();

  if(verbose >= 0){
    std::cout << "+===== create_" << model_name << "_tflite timing breakdown =====" << std::endl;
    std::cout << "| setup     : " << std::chrono::duration_cast<std::chrono::nanoseconds>(setup_end - setup_start).count()/1000.0 << "\t(us)." << std::endl; 
    std::cout << "| set sizes : " << std::chrono::duration_cast<std::chrono::nanoseconds>(size_end - size_start).count()/1000.0 << "\t(us)." << std::endl; 
    std::cout << "| clean up  : " << std::chrono::duration_cast<std::chrono::nanoseconds>(clean_e - clean_s).count()/1000.0 << "\t(us)." << std::endl; 
    std::cout << "+==============================================" << std::endl;
  }
  return 0;
}

int create_crop_tflite(const std::string& out_path, int A, int B, int blk_row, int blk_col, int start_i, int start_j, int verbose){
  uint32_t size_buffer[CROP_OFFSETS_CNT] = {(uint32_t)start_i, (uint32_t)start_j, (uint32_t)(blk_row+start_i), (uint32_t)(blk_col+start_j), (uint32_t)A, (uint32_t)B, (uint32_t)blk_row, (uint32_t)blk_col}; // 0 for input, 1 for output  
  return create_tflite(out_path, "crop", size_buffer, CROP_OFFSETS_CNT, verbose);
}

int create_sub_tflite(const std::string& out_path, int A, int B, int verbose){
  uint32_t size_buffer[SUB_OFFSETS_CNT] = {(uint32_t)(A+B), (uint32_t)B, (uint32_t)A, (uint32_t)B, (uint32_t)A, (uint32_t)B, (uint32_t)A, (uint32_t)B}; // 0 for input, 1 for output  
  return create_tflite(out_path, "sub", size_buffer, SUB_OFFSETS_CNT, verbose);
}






