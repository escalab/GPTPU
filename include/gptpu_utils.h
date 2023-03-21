#ifndef __GPTPU_UTILS_H__
#define __GPTPU_UTILS_H__
#include <stdarg.h>
#include <string.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <unordered_map>
#include "gptpu_buffer.h"

namespace gptpu_utils{

union scaling{
    float f;
    uint8_t c[sizeof(float)];
};

bool file_exist(std::string& file_path);
void get_array_minmax(float* data,
                      float& max,
                      float& min,
                      int m,
                      int n,
                      int ldm);
void ChooseQuantizationParams(float max, 
		              float min, 
			      float& scale, 
			      uint8_t& mean/*nudged_zero_point*/);

/* dequantize sequential in array to out array with leading dimension. */
void dequantization(uint8_t* in, 
                    float* out, 
                    int depth, 
                    int m, 
                    int n, 
                    int ldn,
                    float scale,
                    uint8_t mean);
void array_casting(float* input, 
		   uint8_t* output,
	           float scale,
	 	   uint8_t mean,
		   int m,
		   int n,
		   int ldn,
		   bool transpose);
/*
 For any supported operation, there are at most 3 dimensions.
    Ex: for GEMM, (M,N,K)
 */
std::string select_example_model_path(std::string app_name, int M, int N, int K);
std::string select_template_path(std::string app_name, int M, int N, int K);
std::string get_params_string(std::string app_name, int M, int N, int K);
std::string define_kernel_path(std::string app_name, int M, int N, int K);
void create_file_parent_path(std::string& file_path);
void create_template_from_example_model(openctpu_buffer* buf,
                				        std::string op,
                                        std::string example_path,
                                        std::string template_path,
                                        int m,
                                        int n,
                                        int k);
void open_file_with_check(std::fstream& fd, 
                          std::string& file_path, 
                          std::ios_base::openmode flag);
//void open_file_with_check(int& fd, std::string& file_path, int flag);
//void open_file_with_check(int& fd, std::string& file_path, int flag, int mode);
//void mmap_with_check(char* ptr, int size, int mode, int fd);
//void set_file_size_with_check(int& fd, int size);
void copy_tflite_data_section(uint8_t* out, uint8_t* in, int before_data, int m, int n);
void set_scale_in_tflite_model(uint8_t* dst, 
		                       float scale, 
			                   int scale_loc,
            			       int scale_num,
			                   int scale_stride_len);
void reorder_mm2conv_array(uint8_t* in, 
                           uint8_t* out,
                           int A,
                           int B,
                           int IN_W,
                           int IN_H,
                           int F_W,
                           int F_H,
                           int IN_C);

std::vector<std::string> split_params_str(const std::string& params_str);
void save_mm2conv_weight(float* data,
		         const std::string out_file_path,
			 int B,
			 int C,
			 int blk_r,
			 int blk_c,
			 int i,
			 int j,
			 int row_blk_cnt,
			 int col_blk_cnt,
			 int inn_blk_rem,
			 int col_blk_rem);
std::string replace_delimiter(std::string in_string, 
		              std::string old_delimiter, 
		              std::string new_delimiter);
}
#endif
