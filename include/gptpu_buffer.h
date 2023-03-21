#ifndef __GPTPU_BUFFER_H__
#define __GPTPU_BUFFER_H__

#include "gptpu_dim.h"
    
struct TemplateOffset{
    unsigned long long int before_data;
    unsigned long long int after_data;
};

/* Type of openctpu_buffer */
enum BufferType{
    MODEL,
    DATA,
    OUTPUT
};

class openctpu_buffer{
public:
    
    openctpu_buffer(openctpu_dimension* dim);
    ~openctpu_buffer();

    void get_dims(int& rows, int& cols, int& ldn);

    // input data type of functions
    void config_input_partition();
    void convert_data_to_tflite_pattern(std::string op, float* in); 

    // model type of functions
    void create_model_file(std::string template_path,
		           std::string out_path,
			   void* data);

    // output data type of functions
    void config_output_partition();
    void allocate_output_tflite_array();

    // data pointers
    int** input_arrays;
    void* output;
    float* float_output_array;
    uint8_t* tflite_data_array;

    void set_template_offsets(unsigned long long int before_data, 
		              unsigned long long int after_data){
        this->template_offset.before_data = before_data;
        this->template_offset.after_data  = after_data;
    };

    void get_template_offsets(unsigned long long int& before_data,
		    	      unsigned long long int& after_data){
        before_data = this->template_offset.before_data;
        after_data  = this->template_offset.after_data;
    };

    bool get_transpose(){ return this->dim->get_transpose(); };    
    
    void set_params_str(std::string str){this->params_str = str; };
    std::string get_params_str(){return this->params_str; };

    void set_buffer_type(BufferType buffer_type){ 
	    this->buffer_type = buffer_type; 
    };
    BufferType get_buffer_type(){ return this->buffer_type; };
    void set_scaling_params(float scale, uint8_t mean){
        this->scaling.scale = scale;
        this->scaling.mean = mean;
    };

    void get_scaling_params(float& scale, uint8_t& mean){
        scale = this->scaling.scale;
        mean = this->scaling.mean;
    };

private:
    void cast_to_tflite_data_serial(float* input, 
                                    uint8_t* output, 
                                    float& scale, 
                                    uint8_t& mean);
    BufferType buffer_type;
    struct Scaling{
        float scale = 1.0;
        uint8_t mean = 0.0;
    };
    Scaling scaling;
    openctpu_dimension* dim;
    struct TemplateOffset template_offset;
    std::string params_str; // separated by space
};

#endif
