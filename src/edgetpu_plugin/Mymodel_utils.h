#ifndef EDGETPU_CPP_EXAMPLES_MODEL_UTILS_H_
#define EDGETPU_CPP_EXAMPLES_MODEL_UTILS_H_

#include <array>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "edgetpu.h"
//#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"

namespace coral {

// Builds tflite Interpreter capable of running Edge TPU model.
std::unique_ptr<tflite::Interpreter> BuildEdgeTpuInterpreter(
    const tflite::FlatBufferModel& model,
    edgetpu::EdgeTpuContext* edgetpu_context);

// For stride quantization deisgn
void util_set_chunk_size(int chunk_size);
int  util_get_chunk_size();

// My run Inference
int MyRunInference(int** output_data, const std::vector<uint8_t>& input_data, int iter, tflite::Interpreter* interpreter, int verbose);

int MyRunInferenceV2(int* output_data, int iter, tflite::Interpreter* interpreter, int w_chunk_idx, int in_chunk_idx, int verbose, long long int&, long long int&, bool enable_chunking);

int MyRunInferenceV3(int* output_data, int A, int C, int blk_A, int blk_C, int i, int k, int iter, tflite::Interpreter* interpreter, int verbose, long long int&, long long int&);
int My_simple_output(int* output_data, tflite::Interpreter* interpreter, int verbose);
int My_populate_output_exact(int* output_data, int A, int C, int blk_A, int blk_C, int i, int k, tflite::Interpreter* interpreter, int chunk_num, float SCALE);
int My_populate_output_chunking(int* output_data, int A, int C, int blk_A, int blk_C, int i, int k, tflite::Interpreter* interpreter, int in_chunk_idx, int w_chunk_idx, float SCALE);
int My_populate_element_wise_output(int* output_data, int size, tflite::Interpreter* interpreter, float SCALE);
int My_populate_element_wise_output_chunking(int* output_data, int size, tflite::Interpreter* interpreter, int in_chunk_idx, int w_chunk_idx, float SCALE);

// Runs inference using given `interpreter`
std::vector<float> RunInference(const std::vector<uint8_t>& input_data, tflite::Interpreter* interpreter);

// Returns input tensor shape in the form {height, width, channels}.
std::array<int, 3> GetInputShape(const tflite::Interpreter& interpreter,
                                 int index);

}  // namespace coral
#endif  // EDGETPU_CPP_EXAMPLES_MODEL_UTILS_H_
