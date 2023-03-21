// Example to run a model using one Edge TPU.
// It depends only on tflite and edgetpu.h

// Copyright 2020 Google LLC
// Modified by Nam Vu 2020

#include <algorithm>
#include <chrono>  // NOLINT
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <ostream>
#include <regex>
#include <string>

#include "edgetpu.h"
#include "model_utils.h"
//#include "tensorflow/lite/interpreter.h"
//#include "tensorflow/lite/model.h"

#include "gptpu.h"

typedef std::chrono::time_point<std::chrono::high_resolution_clock> timing;
typedef std::chrono::high_resolution_clock clk;

//namespace gptpu_utils {
//
//std::vector<uint8_t> decode_bmp(
//    const uint8_t* input, int row_size, int width, int height, int channels, bool top_down) {
//  std::vector<uint8_t> output(height * width * channels);
//  for (int i = 0; i < height; i++) {
//    int src_pos;
//    int dst_pos;
//
//    for (int j = 0; j < width; j++) {
//      if (!top_down) {
//        src_pos = ((height - 1 - i) * row_size) + j * channels;
//      } else {
//        src_pos = i * row_size + j * channels;
//      }
//
//      dst_pos = (i * width + j) * channels;
//
//      switch (channels) {
//        case 1:
//          output[dst_pos] = input[src_pos];
//          break;
//        case 3:
//          // BGR -> RGB
//          output[dst_pos] = input[src_pos + 2];
//          output[dst_pos + 1] = input[src_pos + 1];
//          output[dst_pos + 2] = input[src_pos];
//          break;
//        case 4:
//          // BGRA -> RGBA
//          output[dst_pos] = input[src_pos + 2];
//          output[dst_pos + 1] = input[src_pos + 1];
//          output[dst_pos + 2] = input[src_pos];
//          output[dst_pos + 3] = input[src_pos + 3];
//          break;
//        default:
//          std::cerr << "Unexpected number of channels: " << channels << std::endl;
//          std::abort();
//          break;
//      }
//    }
//  }
//  return output;
//}
//
//std::vector<uint8_t> read_bmp(
//    const std::string& input_bmp_name, int* width, int* height, int* channels) {
//  int begin, end;
//
//  std::ifstream file(input_bmp_name, std::ios::in | std::ios::binary);
//  if (!file) {
//    std::cerr << "input file " << input_bmp_name << " not found\n";
//    std::abort();
//  }
//
//  begin = file.tellg();
//  file.seekg(0, std::ios::end);
//  end = file.tellg();
//  size_t len = end - begin;
//
//  std::vector<uint8_t> img_bytes(len);
//  file.seekg(0, std::ios::beg);
//  file.read(reinterpret_cast<char*>(img_bytes.data()), len);
//  const int32_t header_size = *(reinterpret_cast<const int32_t*>(img_bytes.data() + 10));
//  *width = *(reinterpret_cast<const int32_t*>(img_bytes.data() + 18));
//  *height = *(reinterpret_cast<const int32_t*>(img_bytes.data() + 22));
//  const int32_t bpp = *(reinterpret_cast<const int32_t*>(img_bytes.data() + 28));
//  *channels = bpp / 8;
//
//  // there may be padding bytes when the width is not a multiple of 4 bytes
//  // 8 * channels == bits per pixel
//  const int row_size = (8 * *channels * *width + 31) / 32 * 4;
//
//  // if height is negative, data layout is top down
//  // otherwise, it's bottom up
//  bool top_down = (*height < 0);
//
//  // Decode image, allocating tensor once the image size is known
//  const uint8_t* bmp_pixels = &img_bytes[header_size];
//  return decode_bmp(bmp_pixels, row_size, *width, abs(*height), *channels, top_down);
//}
//
//std::map<int, std::string> ParseLabel(const std::string& label_path) {
//  std::map<int, std::string> ret;
//  std::ifstream label_file(label_path);
//  if (!label_file.good()) return ret;
//  for (std::string line; std::getline(label_file, line);) {
//    std::istringstream ss(line);
//    int id;
//    ss >> id;
//    line = std::regex_replace(line, std::regex("^ +[0-9]+ +"), "");
//    ret.emplace(id, line);
//  }
//  return ret;
//}
//
//}  // namespace gptpu_utils
//
void run_model(const std::string model_path, int idx, uint8_t* input, int input_size, uint8_t* output, int output_size, int iter){
    gptpu_utils::DeviceHandler device_handler;

    int total_dev_cnt = device_handler.list_devices(0/*verbose*/);
    assert(idx < total_dev_cnt);
    device_handler.open_device(idx/*tpu_id*/, 1/*verbose*/);

    // Build model.
    device_handler.build_model(model_path, idx/*model_id*/);

    // Build interpreter.
    device_handler.build_interpreter(idx/*tpu_id*/, idx/*model_id*/);

    device_handler.populate_input(input, input_size, idx/*model_id*/);

    // Inference.
    device_handler.model_invoke(idx/*model_id*/, iter/*iter*/);

    // Get output.
    uint8_t zero_point;
    float scale;

    device_handler.get_raw_output(output, output_size, idx/*model_id*/, zero_point, scale);
    std::cout << "zero_point: " << (unsigned)zero_point << ", scale: " << scale << std::endl;
}


int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cout << "Usage: " << argv[0] << " <chain model> <seg_model> <input size> <output size> <iter>" << std::endl;
        return 1;
    }

    const std::string chain_model_path = argv[1];
    const std::string seg_model_path   = argv[2];
    int input_size                     = atoi(argv[3]);
    int output_size                    = atoi(argv[4]);
    int iter                           = atoi(argv[5]);

    // random input data generation
    uint8_t* input = (uint8_t*) malloc(input_size * sizeof(uint8_t));
    for(int i = 0 ; i < input_size ; i++){
        input[i] = rand()%256;
    }
    uint8_t* chain_output = (uint8_t*) malloc(output_size * sizeof(uint8_t));
    uint8_t* seg_interm   = (uint8_t*) malloc(output_size * sizeof(uint8_t));
    uint8_t* seg_output   = (uint8_t*) malloc(output_size * sizeof(uint8_t));

    // Valid condition for seg testing
    assert(input_size == output_size);

    // Run chain model
    run_model(chain_model_path, 0, input, input_size, chain_output, output_size, iter);

    // Run seg model two times
    run_model(seg_model_path, 1, input, input_size, seg_interm, output_size, iter);
    run_model(seg_model_path, 2, seg_interm, output_size, seg_output, output_size, iter);

    // default data checking
    int print_count = 20;
    assert(output_size >= print_count);
    for(int i = 0 ; i < print_count ; i++){
        std::cout << "input[" << i << "]: " 
                  << (unsigned)input[i] 
                  << ", chain output[" << i << "]: " 
                  << (unsigned)chain_output[i]
                  << ", intermediate activatin[" << i << "]: "
                  << (unsigned)seg_interm[i]
                  << ", seg output[" << i << "]: "
                  << (unsigned)seg_output[i]  << std::endl;
    }
    return 0;
}
