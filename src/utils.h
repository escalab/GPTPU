#include <string.h>
#include <dense.h>

int   get_chunk_size();
void  set_chunk_size(int size);
bool  crop_check(int A, int B, int blk_row, int blk_col, int start_i, int start_j);
void  open_devices(int opening_order, int wanted_dev_cnt);
void  close_devices();
int   print_buffer(char* buf, unsigned long long int size);
float set_data_array(int* b, char* data_array, unsigned long long int size);
void  set_block_array(int* a, bool b_major, char* data_array, int row_idx, int col_idx, 
                      int A, int B, int ROW_BLK_REM, int COL_BLK_REM, int chunk_idx);
int   save_input(const std::string& intput_path, int* a, int A);
int   read_output(const std::string& output_path, int* c, int N);
void  conv_save_weight(int* f, const std::string& weight_file_name, int A,int B);
void  save_weight_uint8(uint8_t* f, const std::string& weight_file_name, int size);
void  partition_conv_input(int* a, int* a_blk, int A, int B, int blk_A, int blk_B,
                           int start_i, int start_j);
void  pad_input(int* a, int* a_blk, int* a_pad, int GA, int GB, int A, int B, int A_pad,  
                int B_pad, int padding, int blk_i ,int blk_j, int ROW_BLK_CNT, int COL_BLK_CNT);
void  save_partial(int A, int B, int INNER_BLK_CNT);
int   HPof2(int n);
void  select_blk_shape(int A, int in, int B, int& MM_BLK_ROW, int& MM_BLK_COL, bool b_major);
void  data_mm2conv(int* in, int *out, int A, int B, int IN_W, int IN_H, int F_W, int F_H, int IN_C);
void  data_mm256conv(int* in, int *out, int A, int B, int IN_W, int IN_H, int F_W, int F_H, int IN_C, int chunk_num);
void  set_mm2conv_array(int* b, bool b_major, char* data_array, int B, int C, int i, int j, 
                        int ROW_BLK_CNT, int COL_BLK_CNT, int blk_r, int blk_c, 
                        int INN_BLK_REM, int COL_BLK_REM, int chunk_idx, int exact_mode);
void  set_mm256conv_array(int* b, bool b_major, char* data_array, int B, int C, int i, int j, 
                        int ROW_BLK_CNT, int COL_BLK_CNT, int blk_r, int blk_c, 
                        int INN_BLK_REM, int COL_BLK_REM, int chunk_num, int exact_mode);
void  mm256blk_save_weight(int* b, bool b_major, const std::string& weight_file_name, int B, int C, 
                          int i, int j, int ROW_BLK_CNT, int COL_BLK_CNT, int blk_r, int blk_c, 
                          int INN_BLK_REM, int COL_BLK_REM, int chunk_num);
void  mm2conv_save_weight(int* b, bool b_major, const std::string& weight_file_name, int B, int C, 
                          int i, int j, int ROW_BLK_CNT, int COL_BLK_CNT, int blk_r, int blk_c, 
                          int INN_BLK_REM, int COL_BLK_REM, int chunk_idx);
void  basic_block_mapping(int A, int B, int C, int& AA, int& BB, int& CC);
void  mm2conv_shape_mapping(int A, int B, int C, int& AA, int& BB, int& CC, int exact_mode,      
                            int& IN_W, int& IN_H, int& IN_C, int& F_W, int& F_H, 
                            int& S_W, int& S_H, int& OUT_C);
float get_auto_scale_factor_mm(int*a, int*b, int A, int B, int C);
void  search_conv_optimal(std::string& in_dir, int iter);
void  search_random_conv_optimal(std::string& in_dir, int iter, int sec);
void quantize(float* in, int* out, float& scale, int& mean);









