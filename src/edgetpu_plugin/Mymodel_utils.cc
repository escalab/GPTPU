#include "src/cpp/examples/Mymodel_utils.h"

#include <memory>

#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/kernels/register.h"

//#include "tensorflow/lite/core/api/profiler.h"

#include <chrono>
#include <fcntl.h>
#include <unistd.h>
//#include <pthread.h>
#include <mutex>

typedef std::chrono::time_point<std::chrono::high_resolution_clock> timing;
typedef std::chrono::high_resolution_clock clk;
std::mutex mtx;
//pthread_mutex_t pmtx = PTHREAD_MUTEX_INITIALIZER;
namespace coral {

// For stride quantization design
int CORAL_CHUNK_SIZE = CHAR_BIT; // default as 8 bits

void util_set_chunk_size(int size){
  if(size == 1 || size == 2 || size == 4 || size == 7/*for add model*/ || size == 8 || size == 16){
    CORAL_CHUNK_SIZE = size;
  }else{
    std::cout << "util_set_chunk_size: invalid chunk size: " << size << std::endl;
    exit(1);
  }
}

int util_get_chunk_size(){
  return CORAL_CHUNK_SIZE;
}

std::unique_ptr<tflite::Interpreter> BuildEdgeTpuInterpreter(const tflite::FlatBufferModel& model, edgetpu::EdgeTpuContext* edgetpu_context) {
  tflite::ops::builtin::BuiltinOpResolver resolver;
  resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
  std::unique_ptr<tflite::Interpreter> interpreter;
  if (tflite::InterpreterBuilder(model, resolver)(&interpreter) != kTfLiteOk) {
    std::cerr << "Failed to build interpreter." << std::endl;
  }
  // Bind given context with interpreter.
  interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context);
  interpreter->SetNumThreads(2);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    std::cerr << "Failed to allocate tensors." << std::endl;
  }
  return interpreter;
}

int MyRunInference(int** output_data, const std::vector<uint8_t>& input_data, int iter, tflite::Interpreter* interpreter, int verbose) {
  uint8_t* input = interpreter->typed_input_tensor<uint8_t>(0);
  
  timing mem_start = clk::now();
  std::memcpy(input, input_data.data(), input_data.size());
  timing mem_end   = clk::now();

  
//  std::cout << "try to play around some functions" << std::endl;
//  tflite::Profiler *p;
//  ProfileSummarizer *s;
//  interpreter->SetProfiler(p);

  timing invoke_start = clk::now();
  timing first_start  = clk::now();
  interpreter->Invoke();
  timing first_end   = clk::now();
  if(iter > 1){
    for(int i = 0 ; i < iter-1 ; i++){
      interpreter->Invoke();
    }
  }
  timing invoke_end   = clk::now();

//  p = interpreter->GetProfiler();
//  auto events = p->GetProfileEvents();
//  std::cout << "p: " << p << std::endl;
  

  const auto& output_indices = interpreter->outputs();
  const int num_outputs = output_indices.size();
  int num_values = 0;
  int out_idx = 0;
 
  timing populate_start = clk::now();
  for (int i = 0; i < num_outputs; ++i) {
    const auto* out_tensor = interpreter->tensor(output_indices[i]);
    assert(out_tensor != nullptr);
    if (out_tensor->type == kTfLiteUInt8) {
      num_values = out_tensor->bytes;
      
      // allocate output array
      if(i == 0){ *output_data = (int*) malloc(num_values*sizeof(int));} 
      else{ *output_data = (int*) realloc(*output_data, num_values*sizeof(int)*(i+1)); }	
      //output_data.resize(out_idx + num_values);
//      std::cout << "here" << std::endl;
//      std::cout << "out_tensor quant paprams: " << out_tensor->params.scale << ", " << out_tensor->params.zero_point << std::endl;
//      std::cout << "out_tensor dim size: " << out_tensor->dims->size << std::endl;
//      for(int j = 0 ; j < out_tensor->dims->size; j++){
//        std::cout << out_tensor->dims->data[j] << std::endl;
//      }
//      for(int j = 0 ; j < num_values ; j++){
//        std::cout << (unsigned)out_tensor->data.int8[j];
//      }

      const uint8_t* output = interpreter->typed_output_tensor<uint8_t>(i);
      for (int j = 0; j < num_values; ++j) { // output dim.
//        std::cout << "i: " << i << ", j: " << j << ", out_idx: " << out_idx << ", num_values: " << num_values << std::endl;
        (*output_data)[out_idx++] = (output[j] - out_tensor->params.zero_point) *
                                 out_tensor->params.scale;
//        if(j < 10){
//          std::cout << "output: " << (unsigned)output[j] << ", output_data: " << (*output_data)[out_idx-1] << ", scale: " << out_tensor->params.scale << std::endl; 
//        }
      }
    } else {
      std::cerr << "[MyInference]Tensor " << out_tensor->name
                << " has unsupported output type: " << out_tensor->type
                << std::endl;
    }
  }
  timing populate_end   = clk::now();
  
  if(1){
    std::cout << "+===== MyRunInference timing breakdown =====" << std::endl;
    std::cout << "| memcpy    : " << std::chrono::duration_cast<std::chrono::nanoseconds>(mem_end - mem_start).count()/1000.0 << "\t\t(us)." << std::endl; 
    std::cout << "| 1st invoke: " << std::chrono::duration_cast<std::chrono::nanoseconds>(first_end - first_start).count()/1000.0 << "\t\t(us)." << std::endl; 
    std::cout << "| all invoke: " << std::chrono::duration_cast<std::chrono::nanoseconds>(invoke_end - invoke_start).count()/1000.0 << "\t\t(us). (" << iter << ") times." << std::endl; 
    std::cout << "| populate  : " << std::chrono::duration_cast<std::chrono::nanoseconds>(populate_end - populate_start).count()/1000.0 << "\t\t(us)." << std::endl; 
//    int us = std::chrono::duration_cast<std::chrono::microseconds>(populate_end-mem_start).count();
//    int  op_cnt = num_outputs*num_values*(2*input_data.size()-1);
//    if(interpreter->tensor(output_indices[0])->name == "dense/MatMul"){
//    std::cout << "TOPS: " << ((float)op_cnt / (float)us) / 1000.0 << std::endl;
  //  }    
    std::cout << "+===========================================" << std::endl;
  }
  return num_values;
}

int MyRunInferenceV2(int* output_data, int iter, tflite::Interpreter* interpreter, int w_chunk_idx, int in_chunk_idx, int verbose, long long int& run_ns, long long int& pop_ns, bool enable_chunking) {
  int chunk_size = util_get_chunk_size();

  timing invoke_start = clk::now();
  timing first_start  = clk::now();
  interpreter->Invoke();
  timing first_end   = clk::now();
  if(iter > 1){
    for(int i = 0 ; i < iter-1 ; i++){
      interpreter->Invoke();
    }
  }
  timing invoke_end   = clk::now();

  const auto& output_indices = interpreter->outputs();
  const int num_outputs = output_indices.size();
  int num_values = 0;
  int out_idx = 0;
  int temp_value = 0;
  timing s, e;
  timing populate_start = clk::now();
  for (int i = 0; i < num_outputs; ++i) {
    const auto* out_tensor = interpreter->tensor(output_indices[i]);
    assert(out_tensor != nullptr);
    if (out_tensor->type == kTfLiteUInt8) {
      num_values = out_tensor->bytes;
      const uint8_t* output = interpreter->typed_output_tensor<uint8_t>(i);
      s = clk::now();

 //     pthread_mutex_lock(&pmtx);
      if(enable_chunking == true){
        for (int j = 0; j < num_values; ++j) { // output dim.
         //int chunk_mask = (~(0xffffffff << chunk_size)) << ((in_chunk_idx+w_chunk_idx) * chunk_size); 
         temp_value = (output[j] - out_tensor->params.zero_point) * out_tensor->params.scale;
         output_data[out_idx++] += (temp_value << ((in_chunk_idx+w_chunk_idx) * chunk_size)); 
  //        if(j == 0)std::cout << "output_data: " << output_data[j] << ", output: " << (unsigned)output[j] << ", temp_value: " << temp_value << ", out_idx: " << out_idx << std::endl;
        }
      }else{ // no chunking
        for (int j = 0; j < num_values; ++j) { // output dim.
         output_data[out_idx++] = (output[j] - out_tensor->params.zero_point) * out_tensor->params.scale;
        }
      }
//      pthread_mutex_unlock(&pmtx);
      e = clk::now();
    } else {
      std::cerr << "[MyInference]Tensor " << out_tensor->name
                << " has unsupported output type: " << out_tensor->type
                << std::endl;
    }
  }
  timing populate_end   = clk::now();
  
  if(verbose > 0){
    std::cout << "+===== MyRunInferenceV2 timing breakdown =====" << std::endl;
    std::cout << "| 1st invoke: " << std::chrono::duration_cast<std::chrono::nanoseconds>(first_end - first_start).count()/1000.0 << "\t\t(us)." << std::endl; 
    std::cout << "| all invoke: " << std::chrono::duration_cast<std::chrono::nanoseconds>(invoke_end - invoke_start).count()/1000.0 << "\t\t(us). (" << iter << ") times." << std::endl; 
    std::cout << "| populate  : " << std::chrono::duration_cast<std::chrono::nanoseconds>(populate_end - populate_start).count()/1000.0 << "\t\t(us)." << std::endl; 
    std::cout << "+=============================================" << std::endl;
  }

  run_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(invoke_end - invoke_start).count();
  pop_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(populate_end - populate_start).count();
  
  return num_values;
}

int MyRunInferenceV3(int* output_data, int A, int C, int blk_A, int blk_C, int blk_idx_i, int blk_idx_k, int iter, tflite::Interpreter* interpreter, int verbose, long long int& run_ns, long long int& pop_ns) {
  timing invoke_start = clk::now();
  timing first_start  = clk::now();
  interpreter->Invoke();
  timing first_end   = clk::now();
  if(iter > 1){
    for(int i = 0 ; i < iter-1 ; i++){
      interpreter->Invoke();
    }
  }
  timing invoke_end   = clk::now();

  const auto& output_indices = interpreter->outputs();
  const int num_outputs = output_indices.size();
  int num_values = 0;
  int offset = 0;
  timing populate_start = clk::now();
  for (int i = 0; i < num_outputs; ++i) {
    const auto* out_tensor = interpreter->tensor(output_indices[i]);
    assert(out_tensor != nullptr);
    if (out_tensor->type == kTfLiteUInt8) {
      num_values = out_tensor->bytes;
      const uint8_t* output = interpreter->typed_output_tensor<uint8_t>(i);
      for (int j = 0; j < num_values/* == blk_A * blk_C */; ++j) { // output dim.
        int idx_r = j/blk_C;
        int idx_c = j%blk_C;
        offset = blk_idx_i*(blk_A*C)+blk_idx_k*blk_C/*init offset*/+ idx_r*(C)+idx_c/*in block offset*/;
        output_data[offset] = (output[j] - out_tensor->params.zero_point) * out_tensor->params.scale;
//        std::cout << "output_data: " << output_data[j] << ", output: " << (unsigned)output[j] << std::endl;
      }
    } else {
      std::cerr << "[MyInference]Tensor " << out_tensor->name
                << " has unsupported output type: " << out_tensor->type
                << std::endl;
    }
  }
  timing populate_end   = clk::now();
  
  if(verbose > 0){
    std::cout << "+===== MyRunInferenceV2 timing breakdown =====" << std::endl;
    std::cout << "| 1st invoke: " << std::chrono::duration_cast<std::chrono::nanoseconds>(first_end - first_start).count()/1000.0 << "\t\t(us)." << std::endl; 
    std::cout << "| all invoke: " << std::chrono::duration_cast<std::chrono::nanoseconds>(invoke_end - invoke_start).count()/1000.0 << "\t\t(us). (" << iter << ") times." << std::endl; 
    std::cout << "| populate  : " << std::chrono::duration_cast<std::chrono::nanoseconds>(populate_end - populate_start).count()/1000.0 << "\t\t(us)." << std::endl; 
    std::cout << "+=============================================" << std::endl;
  }

  run_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(invoke_end - invoke_start).count();
  pop_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(populate_end - populate_start).count();
  
  return num_values;
}

int My_simple_output(int* output_data, tflite::Interpreter* interpreter, int verbose) {
  const auto& output_indices = interpreter->outputs();
  const int num_outputs = output_indices.size();
  int num_values = 0;
  int offset = 0;
  for (int i = 0; i < num_outputs; ++i) {
    const auto* out_tensor = interpreter->tensor(output_indices[i]);
    assert(out_tensor != nullptr);
    if (out_tensor->type == kTfLiteUInt8) {
      num_values = out_tensor->bytes;
      const uint8_t* output = interpreter->typed_output_tensor<uint8_t>(i);
//      std::cout << __func__ << ": num_value: " << num_values << std::endl;
      for (int j = 0; j < num_values; ++j) { // output dim.
        output_data[j] = (output[j] - out_tensor->params.zero_point) * out_tensor->params.scale;
//        if(j < 10)std::cout << __func__ << ": output_data[" << j << "]: " << output_data[j] << ", output[" << j << "]: " <<(unsigned)output[j] << ", z: " << out_tensor->params.zero_point << ", s: " << out_tensor->params.scale << std::endl;
      }
    } else {
      std::cerr << "[MyInference]Tensor " << out_tensor->name
                << " has unsupported output type: " << out_tensor->type
                << std::endl;
    }
  }
  return num_values;
}
int My_populate_output_chunking(int* output_data, int A, int C, int blk_A, int blk_C, int blk_idx_i, int blk_idx_k, tflite::Interpreter* interpreter, int in_chunk_idx, int w_chunk_idx, float IN_SCALE) {
  const auto& output_indices = interpreter->outputs();
  const int num_outputs = output_indices.size();
  int num_values = 0;
  int offset = 0;
  int chunk_size = util_get_chunk_size();
  int temp_value = 0;
  for (int i = 0; i < num_outputs; ++i) {
    const auto* out_tensor = interpreter->tensor(output_indices[i]);
    assert(out_tensor != nullptr);
    if (out_tensor->type == kTfLiteUInt8) {
      num_values = out_tensor->bytes;
      const uint8_t* output = interpreter->typed_output_tensor<uint8_t>(i);
//TODO: optimization : check "num_values = blk_A*blk_C" and then avoid "/" and "%" operation by 2 loops
      for (int j = 0; j < num_values/* == blk_A * blk_C */; ++j) { // output dim.
        int idx_r = j/blk_C;
        int idx_c = j%blk_C;
        offset = blk_idx_i*(blk_A*C)+blk_idx_k*blk_C/*init offset*/+ idx_r*(C)+idx_c/*in block offset*/;
        temp_value = (output[j] - out_tensor->params.zero_point) * out_tensor->params.scale * (float)(1.0/float(IN_SCALE)); // * (1.0/float(SCALE));// * (1/0.12451171875);
//* stride quantization re-accumulation design */
        int chunk_mask = (~(0xffffffff << chunk_size)) << ((in_chunk_idx+w_chunk_idx) * chunk_size);
// should be atomic among two chunk_idxs
//        if(j == 0){
//          std::cout << "pre output_data: " << output_data[offset] << ", ";
//        }
// TODO: first term: not just overwrite, need correct adding here
        //output_data[offset] = ((~chunk_mask) & output_data[offset]) + ((temp_value << ((in_chunk_idx+w_chunk_idx) * chunk_size)) & chunk_mask);
        output_data[offset] = (output_data[offset]) + ((temp_value << ((in_chunk_idx+w_chunk_idx) * chunk_size)) /*& chunk_mask*/);
        //if(offset  == 0){
        //   std::cout << "IN_SCALE: " << IN_SCALE << ", params.scale: " << out_tensor->params.scale << ", output_data[" << offset << "]: " << output_data[offset] << ", second term: " << ((temp_value << ((in_chunk_idx+w_chunk_idx) * chunk_size)) & chunk_mask) << ", temp_value: " << temp_value << ", in_chunk_idx: " << in_chunk_idx << ", w_chunk_idx: " << w_chunk_idx << ", output[j]: " << (unsigned)output[j] << std::endl;
        //   std::cout << std::hex << "mask: " << chunk_mask << std::dec << std::endl;
        // }
      }
    } else {
      std::cerr << "[MyInference]Tensor " << out_tensor->name
                << " has unsupported output type: " << out_tensor->type
                << std::endl;
    }
  }
  return num_values;
}
int My_populate_output_exact(int* output_data, int A, int C, int blk_A, int blk_C, int blk_idx_i, int blk_idx_k, tflite::Interpreter* interpreter, int chunk_num, float IN_SCALE) {
  const auto& output_indices = interpreter->outputs();
  const int num_outputs = output_indices.size();
  int num_values = 0;
  int offset = 0;
  int chunk_size = util_get_chunk_size();
  int temp_value = 0;

  int bA_bC_cn = blk_A*blk_C*chunk_num;
  int bC_cn    = blk_C*chunk_num;
  int A_C_cn   = A*C*chunk_num;
  int bA_C_cn  = blk_A*C*chunk_num;
  int C_cn     = C*chunk_num;

  for (int i = 0; i < num_outputs; ++i) {
    const auto* out_tensor = interpreter->tensor(output_indices[i]);
    assert(out_tensor != nullptr);
    if (out_tensor->type == kTfLiteUInt8) {
      num_values = out_tensor->bytes;
      const uint8_t* output = interpreter->typed_output_tensor<uint8_t>(i);
      for (int j = 0; j < num_values/* == blk_A*chunk_num * blk_C*chunk_num */; ++j) { // output dim.
//        int idx_r = j/(blk_C*chunk_num);
//        int idx_c = j%(blk_C*chunk_num);
//        int chunk_r = j/(bA_bC_cn); // the ith 256x256 block within A*C in row direction that this element belongs to.
//        int chunk_c = (j%(bC_cn))/blk_C; // the jth 256x256 block within A*C in col direction that this element belongs to.
//        int inblk_r = (j/(bC_cn))/*idx_r*/%blk_A; // the ith element within a 256x256 block
//        int inblk_c = (j%(bC_cn))/*idx_c*/%blk_C; // the jth element within a 256x256 block
/*
input: "output: array,       shape: (256*16)*(256*16)
output: "output_data: array, shape: ((n*256)*16)*((n*256)*16)
==> matrix_dim: n*256
*/
// TODO: delay the indexing calculation until summation stage for pure CPU optimization globally.

//        offset = chunk_r*(A_C_cn)+chunk_c*C /*starting offset of a A*C block within the A*chunk_num*C*chunk_num block (diff chunks) */ 
//                 + blk_idx_i*(bA_C_cn)+blk_idx_k*(blk_C) /*starting offset of the 256x256 block within a A*C block  */
//                 + inblk_r*(C_cn)+inblk_c/*element offset within the 256x256 block*/; 
        /*output_data is partial_c[INN_BLK_idx]*/
        //output_data[j/*offset*/] = (int)((output[j] - out_tensor->params.zero_point) * out_tensor->params.scale * (float)(1.0/float(IN_SCALE))) << (chunk_r + chunk_c); 
        output_data[j] = (int)output[j]; 
      }
    } else {
      std::cerr << "[MyInference]Tensor " << out_tensor->name
                << " has unsupported output type: " << out_tensor->type
                << std::endl;
    }
  }
  return num_values;
}

int My_populate_element_wise_output(int* output_data, int size, tflite::Interpreter* interpreter, float SCALE) {
  int chunk_size = util_get_chunk_size();
  const auto& output_indices = interpreter->outputs();
  const int num_outputs = output_indices.size();
  int num_values = 0;
  int out_idx = 0;
  int temp_value = 0;
  for (int i = 0; i < num_outputs; ++i) {
    const auto* out_tensor = interpreter->tensor(output_indices[i]);
    assert(out_tensor != nullptr);
    if (out_tensor->type == kTfLiteUInt8) {
      num_values = out_tensor->bytes;
      const uint8_t* output = interpreter->typed_output_tensor<uint8_t>(i);
      for (int j = 0; j < num_values; ++j) { // output dim.
       output_data[out_idx++] = (output[j] - out_tensor->params.zero_point) * out_tensor->params.scale;
      }
    } else {
      std::cerr << "[MyInference]Tensor " << out_tensor->name
                << " has unsupported output type: " << out_tensor->type
                << std::endl;
    }
  }
  return num_values;
}
int My_populate_element_wise_output_chunking(int* output_data, int size, tflite::Interpreter* interpreter, int w_chunk_idx, int in_chunk_idx, float SCALE) {
  int chunk_size = util_get_chunk_size();
  const auto& output_indices = interpreter->outputs();
  const int num_outputs = output_indices.size();
  int num_values = 0;
  int out_idx = 0;
  int temp_value = 0;
  for (int i = 0; i < num_outputs; ++i) {
    const auto* out_tensor = interpreter->tensor(output_indices[i]);
    assert(out_tensor != nullptr);
    if (out_tensor->type == kTfLiteUInt8) {
      num_values = out_tensor->bytes;
      const uint8_t* output = interpreter->typed_output_tensor<uint8_t>(i);
      for (int j = 0; j < num_values; ++j) { // output dim.
       temp_value = (output[j] - out_tensor->params.zero_point) * out_tensor->params.scale;
       output_data[out_idx++] += (temp_value << ((in_chunk_idx+w_chunk_idx) * chunk_size)); 
      }
    } else {
      std::cerr << "[MyInference]Tensor " << out_tensor->name
                << " has unsupported output type: " << out_tensor->type
                << std::endl;
    }
  }
  return num_values;
}
std::vector<float> RunInference(const std::vector<uint8_t>& input_data,
                                tflite::Interpreter* interpreter) {
  std::vector<float> output_data;
  uint8_t* input = interpreter->typed_input_tensor<uint8_t>(0);
  std::memcpy(input, input_data.data(), input_data.size());

  interpreter->Invoke();

  //std::cerr << " ===== My insertion code =====" << std::endl;
  //auto profiler = interpreter->GetProfiler();
  //std::cerr << "profiler type: " << typeid(profiler).name() << std::endl;
  //std::cerr << profiler << std::endl;  
  //interpreter->SetProfiler(profiler);
  //std::cerr << " ===== My insertion code =====" << std::endl;

  const auto& output_indices = interpreter->outputs();
  const int num_outputs = output_indices.size();
  int out_idx = 0;
  for (int i = 0; i < num_outputs; ++i) {
    const auto* out_tensor = interpreter->tensor(output_indices[i]);
    assert(out_tensor != nullptr);
    if (out_tensor->type == kTfLiteUInt8) {
      const int num_values = out_tensor->bytes;
      output_data.resize(out_idx + num_values);
      const uint8_t* output = interpreter->typed_output_tensor<uint8_t>(i);
      for (int j = 0; j < num_values; ++j) {
        output_data[out_idx++] = (output[j] - out_tensor->params.zero_point) *
                                 out_tensor->params.scale;
      }
    } else if (out_tensor->type == kTfLiteFloat32) {
      const int num_values = out_tensor->bytes / sizeof(float);
      output_data.resize(out_idx + num_values);
      const float* output = interpreter->typed_output_tensor<float>(i);
      for (int j = 0; j < num_values; ++j) {
        output_data[out_idx++] = output[j];
      }
    } else {
      std::cerr << "Tensor " << out_tensor->name
                << " has unsupported output type: " << out_tensor->type
                << std::endl;
    }
  }
  return output_data;
}

std::array<int, 3> GetInputShape(const tflite::Interpreter& interpreter,
                                 int index) {
  const int tensor_index = interpreter.inputs()[index];
  const TfLiteIntArray* dims = interpreter.tensor(tensor_index)->dims;
  return std::array<int, 3>{dims->data[1], dims->data[2], dims->data[3]};
}

}  // namespace coral
