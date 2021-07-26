#include "gptpu.h"
#include <string>
int make_mm2conv_temp(const std::string& in_path, const std::string& out_path, int blk_A, int blk_B, int blk_C);
int make_dense_temp(const std::string& in_path, const std::string& out_path);
int make_temp(std::string model_name, const std::string& in_path, const std::string& out_path);

