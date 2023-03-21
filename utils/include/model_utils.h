// Copyright 2020 Google LLC
// Modified by Nam Vu 2020

#ifndef EDGETPU_CPP_EXAMPLES_MODEL_UTILS_H_
#define EDGETPU_CPP_EXAMPLES_MODEL_UTILS_H_

#include <array>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "edgetpu.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"

namespace model_utils {
	class DeviceHandler{
		public:
            DeviceHandler();
            ~DeviceHandler();

            // List all edgetpu devices and returns the number of them.
            int list_devices(int verbose);

            // Open 'tpuid' edgetpu device.
            void open_device(int tpuid, int verbose);

            // Separate out the original 'BuildInterpreter' into 'build_model' and 'build_interpreter' two functions
            // Build edgetpu model.
            void build_model(const std::string& model_path, int model_id);

            // Build edgetpu interpreter.
            void build_interpreter(int tpuid, int model_id);

            // Separate out the original 'RunInfernece' into populate_input invoke populate_output
            // Populate input array.
            void populate_input(uint8_t* data, int size, int model_id);

            // Actual invoke calling.
            void model_invoke(int model_id, int iter);

            // Get output array.
            void get_output(int* data, int model_id);
            
            // Get uin8_t raw output array.
	    // It's caller's duty to make sure size is valid for data pointer.
            void get_raw_output(uint8_t* data, int size, int model_id, uint8_t& zero_point, float& scale);

        private:
            bool device_listed;
            unsigned int total_dev_cnt;
            std::vector<std::shared_ptr<tflite::Interpreter>> interpreters;
			std::vector<std::shared_ptr<edgetpu::EdgeTpuContext>> edgetpu_contexts;
			std::vector<edgetpu::EdgeTpuManager::DeviceEnumerationRecord> enumerate_edgetpus;
			std::vector<std::shared_ptr<tflite::FlatBufferModel>> models;
    };


// Builds tflite Interpreter capable of running Edge TPU model.
std::unique_ptr<tflite::Interpreter> BuildEdgeTpuInterpreter(
    const tflite::FlatBufferModel& model, edgetpu::EdgeTpuContext* edgetpu_context);

// Builds tflite Interpreter for normal CPU model.
std::unique_ptr<tflite::Interpreter> BuildInterpreter(const tflite::FlatBufferModel& model);

// Runs inference using given `interpreter`
std::vector<float> RunInference(
    const std::vector<uint8_t>& input_data, tflite::Interpreter* interpreter);

// Returns input tensor shape in the form {height, width, channels}.
std::array<int, 3> GetInputShape(const tflite::Interpreter& interpreter, int index);

}  // namespace gptpu_utils
#endif  // EDGETPU_CPP_EXAMPLES_MODEL_UTILS_H_
