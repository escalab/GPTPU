// Copyright 2020 Google LLC
// Modified by Nam Vu 2020

#include "model_utils.h"
#include <iostream>
#include <assert.h>

#include <memory>

#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/kernels/register.h"

namespace model_utils {
	DeviceHandler::DeviceHandler(){
		this->device_listed = false;
        this->total_dev_cnt = 0;
    }
	DeviceHandler::~DeviceHandler(){
		// Reset all vectors to size zero.
		this->interpreters       = 
            std::vector<std::shared_ptr<tflite::Interpreter>>();
		this->edgetpu_contexts   = 
            std::vector<std::shared_ptr<edgetpu::EdgeTpuContext>>();
		this->enumerate_edgetpus = 
            std::vector<edgetpu::EdgeTpuManager::DeviceEnumerationRecord>();
		this->models             = 
            std::vector<std::shared_ptr<tflite::FlatBufferModel>>();
    }

	int DeviceHandler::list_devices(int verbose){
		if(this->device_listed == true){
            return this->total_dev_cnt;
        }
        this->device_listed = true;
        this->enumerate_edgetpus = 
            edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();
		this->total_dev_cnt = this->enumerate_edgetpus.size();
		// at least 'dev_cnt' number of models are allowed.
		this->models.resize(this->total_dev_cnt); 
		this->edgetpu_contexts.resize(this->total_dev_cnt);
		this->interpreters.resize(this->total_dev_cnt);
		assert( this->enumerate_edgetpus.size() == 
                this->edgetpu_contexts.size() );
		if(verbose){
			std::cout << "enumerated edgetpu: " << std::endl;
			for(auto it = this->enumerate_edgetpus.begin() ; 
                it != this->enumerate_edgetpus.end() ; it++){
				std::cout << it->path << std::endl;
			}
		}
		return this->total_dev_cnt;
	}
	void DeviceHandler::open_device(int tpuid, int verbose){
		assert(tpuid >= 0 && tpuid < (int)this->edgetpu_contexts.size());
		this->edgetpu_contexts[tpuid] = 
            edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice(
					this->enumerate_edgetpus[tpuid].type,
					this->enumerate_edgetpus[tpuid].path
				);
		if(verbose){
			std::cout << "opened device: ";
			std::cout << "type: " << this->enumerate_edgetpus[tpuid].type;
			std::cout << ", path: " << this->enumerate_edgetpus[tpuid].path;
			std::cout << std::endl;
		}
	}

	void DeviceHandler::build_model(const std::string& model_path, int model_id){
		std::shared_ptr<tflite::FlatBufferModel> local_model_tmp;
		local_model_tmp = 
            tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
		if(local_model_tmp == nullptr){
			std::cout << "Fail to build FlatBufferModel from file: " 
                      << model_path << std::endl;
			std::abort();
		}	
		assert(model_id >= 0 && model_id < (int)this->models.size());
		this->models[model_id] = local_model_tmp;
	}

	void DeviceHandler::build_interpreter(int tpuid, int model_id){
		std::shared_ptr<tflite::Interpreter> tmp;
		tmp = BuildEdgeTpuInterpreter(
				*this->models[model_id],
				this->edgetpu_contexts[tpuid].get()
				);
		if(tmp == nullptr){
			std::cout << "Fail to build interpreter ( ";
			std::cout << "tpu_id: " << tpuid;
			std::cout << ", model_id: " << model_id;
			std::cout << " )" << std::endl;
			std::abort();
		}
		assert(model_id >= 0 && model_id < (int)this->interpreters.size());
		this->interpreters[model_id] = tmp;
	}
	
	void DeviceHandler::populate_input(uint8_t* data, int size, int model_id){
		assert(model_id >= 0 && model_id < (int)this->interpreters.size());
		uint8_t* input = 
            this->interpreters[model_id].get()->typed_input_tensor<uint8_t>(0);
        std::memcpy(input, data, size);
    }

	void DeviceHandler::model_invoke(int model_id, int iter){
		assert(model_id >= 0 && model_id < (int)this->interpreters.size());
		std::shared_ptr<tflite::Interpreter> tmp = this->interpreters[model_id];
		for(int i = 0 ; i < iter ; i++){
			tmp.get()->Invoke();
		}
	}

	void DeviceHandler::get_output(int* data, int model_id){
		assert(model_id >= 0 && model_id < (int)this->interpreters.size());
		std::shared_ptr<tflite::Interpreter> tmp = this->interpreters[model_id];
		const auto& output_indices = tmp->outputs();
		const int num_outputs = output_indices.size();
		int num_values = 0;
		for(int i = 0 ; i < num_outputs ; ++i){
			const auto* out_tensor = tmp->tensor(output_indices[i]);
			assert(out_tensor != nullptr);
			if(out_tensor->type == kTfLiteUInt8){
				num_values = out_tensor->bytes;
				const uint8_t* output = tmp->typed_output_tensor<uint8_t>(i);
				for(int j = 0 ; j < num_values ; ++j){
					data[j] = (output[j] - out_tensor->params.zero_point) *
							out_tensor->params.scale;
                }
			}else{
				std::cout << "Tensor " << out_tensor->name
					  << "has unsupported output type: " << out_tensor->type
					  << std::endl;
				std::abort();
			}
		}
	}
	
    /*   
        Caller has to pre-allocate it's output data array in uint8_t pointer 
        type before calling this function.
    
        Caller has it's own full control over the output array since it's 
        conceptually copied from the internal interpreter. 
    
        This new output design avoids unnessacery data copy and the latency is 
        reduced 100x to be minor compared to the main model invokation time.
    */
    void DeviceHandler::get_raw_output(uint8_t* data, 
                                       int size, 
                                       int model_id, 
                                       uint8_t& zero_point, 
                                       float& scale){
        assert(model_id >= 0 && model_id < (int)this->interpreters.size());
		std::shared_ptr<tflite::Interpreter> tmp = this->interpreters[model_id];
		const auto& output_indices = tmp->outputs();
		const int num_outputs = output_indices.size();
		int num_values = 0;
		for(int i = 0 ; i < num_outputs ; ++i){
			const auto* out_tensor = tmp->tensor(output_indices[i]);
			assert(out_tensor != nullptr);
			if(out_tensor->type == kTfLiteUInt8){
				num_values = out_tensor->bytes;
                std::memcpy(data, 
                            tmp->typed_output_tensor<uint8_t>(i), 
                            size);
				zero_point = out_tensor->params.zero_point;
                scale      = out_tensor->params.scale;
			}else{
				std::cout << "Tensor " << out_tensor->name
					  << "has unsupported output type: " << out_tensor->type
					  << std::endl;
				std::abort();
			}
		}
	}

std::unique_ptr<tflite::Interpreter> BuildEdgeTpuInterpreter(
    const tflite::FlatBufferModel& model, edgetpu::EdgeTpuContext* edgetpu_context) {
  tflite::ops::builtin::BuiltinOpResolver resolver;
  resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
  std::unique_ptr<tflite::Interpreter> interpreter;
  if (tflite::InterpreterBuilder(model, resolver)(&interpreter) != kTfLiteOk) {
    std::cerr << "Failed to build interpreter." << std::endl;
  }
  // Bind given context with interpreter.
  interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context);
  interpreter->SetNumThreads(1);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    std::cerr << "Failed to allocate tensors." << std::endl;
  }
  return interpreter;
}

std::unique_ptr<tflite::Interpreter> BuildInterpreter(const tflite::FlatBufferModel& model) {
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolver resolver;
  if (tflite::InterpreterBuilder(model, resolver)(&interpreter) != kTfLiteOk) {
    std::cerr << "Failed to build interpreter." << std::endl;
  }
  interpreter->SetNumThreads(1);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    std::cerr << "Failed to allocate tensors." << std::endl;
  }
  return interpreter;
}

std::vector<float> RunInference(
    const std::vector<uint8_t>& input_data, tflite::Interpreter* interpreter) {
  std::vector<float> output_data;
  uint8_t* input = interpreter->typed_input_tensor<uint8_t>(0);
  std::memcpy(input, input_data.data(), input_data.size());

  interpreter->Invoke();

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
        output_data[out_idx++] =
            (output[j] - out_tensor->params.zero_point) * out_tensor->params.scale;
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
                << " has unsupported output type: " << out_tensor->type << std::endl;
    }
  }
  return output_data;
}

std::array<int, 3> GetInputShape(const tflite::Interpreter& interpreter, int index) {
  const int tensor_index = interpreter.inputs()[index];
  const TfLiteIntArray* dims = interpreter.tensor(tensor_index)->dims;
  return std::array<int, 3>{dims->data[1], dims->data[2], dims->data[3]};
}

}  // namespace gptpu_utils
