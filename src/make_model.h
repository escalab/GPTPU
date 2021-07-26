#include "gptpu.h"
#include <string>
int create_mm2conv_tflite(const std::string& template_path, std::vector<Flatbufs>& flatbufs,/*const std::string& out_path*/int model_id, char* data_array, int blk_A, int blk_B, int blk_C, float SCALE, int chunk_num/*for mm256conv use*/);
int create_dense_tflite(const std::string& out_path, int A, int B, char* data, char* scaling, char* bias_scaling, char* zero_point, const std::string& data_type, int verbose);
int create_crop_tflite( const std::string& out_path, int A, int B, int blk_row, int blk_col, int start_i, int start_j, int verbose);
int create_sub_tflite( const std::string& out_path, int A, int B, int verbose);
