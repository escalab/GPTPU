#include <math.h>
#include <fcntl.h>
#include <float.h>
#include <assert.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "gptpu_utils.h"
#include "gptpu_buffer.h"

openctpu_buffer::openctpu_buffer(openctpu_dimension* dim){
	this->dim = dim;
}

openctpu_buffer::~openctpu_buffer(){
    free(this->tflite_data_array);
}

void openctpu_buffer::get_dims(int& rows, int& cols, int& ldn){
	this->dim->get_dims(rows, cols, ldn);
}

void openctpu_buffer::config_input_partition(){
		
}

void openctpu_buffer::cast_to_tflite_data_serial(float* input, uint8_t* output, float& scale, uint8_t& mean){
    float max = FLT_MIN;
    float min = FLT_MAX;
    int m, n, ldn;
    this->get_dims(m, n, ldn);
    gptpu_utils::get_array_minmax((float*)input, max, min, m, n, ldn);
    gptpu_utils::ChooseQuantizationParams(max, min, scale, mean);
    gptpu_utils::array_casting((float*)input, 
                               (uint8_t*)output,
                               scale,
                               mean,
                               m,
                               n,
                               ldn,
                               this->get_transpose());
    this->set_scaling_params(scale, mean);
}

void openctpu_buffer::convert_data_to_tflite_pattern(std::string op, float* in){
    int m, n, ldn;
    float scale;
    uint8_t mean;
    this->get_dims(m, n, ldn);
    /*
     This is a temporary internal array. Due to data layout pattern's complexity,
     two steps are used here. Leave this for future optimization.
    TODO: merge the behavior of 
        1. cast_to_tflite_serial 
        2. reorder_mm2conv_array
        to save data copy & movement time (if needed)
     */
    uint8_t* array_serial = (uint8_t*) malloc(m * n * sizeof(uint8_t));
    this->tflite_data_array = (uint8_t*) malloc(m * n * sizeof(uint8_t));
    this->cast_to_tflite_data_serial(in, array_serial, scale, mean);
    this->set_scaling_params(scale, mean);
    if(op == "conv2d"){
	    std::vector<std::string> params = 
		    gptpu_utils::split_params_str(this->params_str);
        assert(params.size() == 9);
        gptpu_utils::reorder_mm2conv_array(array_serial, 
   	  	                                   this->tflite_data_array, 
        	    	        	           m, 
		            	                   n, 
			                               atoi(params[0].c_str()), // IN_W
			                               atoi(params[1].c_str()), // IN_H
			                               atoi(params[4].c_str()), // F_W
			                               atoi(params[5].c_str()), // F_H
			                               atoi(params[2].c_str())); // IN_C
    }else{
        std::cout << __func__ << ": op: " << op 
		  << " not suported yet." << std::endl;
    }
    free(array_serial);
}

void openctpu_buffer::config_output_partition(){
		
}

void openctpu_buffer::allocate_output_tflite_array(){
    int m, n, ldn;
    this->get_dims(m, n, ldn);
    this->tflite_data_array = (uint8_t*) malloc(m * n * sizeof(uint8_t));
}

void openctpu_buffer::create_model_file(std::string template_path,
		                        std::string out_path,
					void* data){
    // Init 
    int m, n, ldm;
    this->get_dims(m, n, ldm);
    int temp_fd, out_fd;
    uint8_t *src, *dst; // for mmap
    float scale;
    uint8_t mean;
    struct stat st_temp;
    unsigned long long int before_data, after_data;
    // TODO: replace 1Kx1K temp parameters here
    this->set_template_offsets(0x3128, 0x103cfc);
    this->get_template_offsets(before_data, after_data);
    unsigned long long int ext_loc = before_data + m * n;
    unsigned long long int ext_len = 0xbd4;
    unsigned long long int out_file_size = 0x10a260;
    unsigned long long int scale_loc = 0x107aac;
    unsigned long long int scale_stride_len = 0x2738;
    unsigned long long int scale_num = 2;
 
    unsigned long long int blk_len     = 4;    //four consecutive elements in a row
    unsigned long long int stride_len  = 0x100; //size/blk_len/*256*/;  // how long is the size between two blks' starting
    unsigned long long int stride_zero_len = 0x100; // TODO: so far all are the same, make as a parameter if differ for others
    unsigned long long int num_strips  = 64 /*stride_len/blk_len*/; // how many consecutive striding blks in this section (between sections, the zeros are longer)
    unsigned long long int section_len = blk_len*num_strips*stride_len+/*zeros*/stride_zero_len;
    int data_array_size = m*n;
    unsigned long long int num_section = ((data_array_size) / section_len)+1; // 2^4 = 16

    uint8_t* ext_zeros = (uint8_t*) calloc(ext_len, sizeof(uint8_t));
    uint8_t* data_char = (uint8_t*) malloc(data_array_size * sizeof(uint8_t));
 
    // Setup
    gptpu_utils::create_file_parent_path(out_path);
    assert((temp_fd = open(template_path.c_str(), O_RDONLY)) >= 0);
    assert((out_fd = open(out_path.c_str(), 
                          O_RDWR | O_CREAT | O_TRUNC, 
                          0777)) >= 0);
    fstat(temp_fd, &st_temp);
    assert((src = static_cast<uint8_t*>(mmap(NULL, 
                                            st_temp.st_size, 
                                            PROT_READ, 
                                            MAP_SHARED, 
                                            temp_fd, 
                                            0))) != MAP_FAILED);
    out_file_size = st_temp.st_size + data_array_size + stride_zero_len * num_section + ext_len;
    assert((dst = static_cast<uint8_t*>(mmap(NULL, 
                                            out_file_size, 
                                            PROT_WRITE,
                                            MAP_SHARED, 
                                            out_fd, 
                                            0))) != MAP_FAILED);
    std::cout << __func__ << ": output kernel file size: " << out_file_size << std::endl;
    assert(ftruncate(out_fd, out_file_size) == 0); // set out file size

    /* Main logic */
    // 1. Copy the first section before data section
    memcpy(dst, src, before_data);
    // 2. Cast float input array to uint8_t array
    this->cast_to_tflite_data_serial((float*)data, 
                                     (uint8_t*)data_char, 
                                     scale, 
                                     mean);
    // 3. Copy data section following tflite's pattern
    gptpu_utils::copy_tflite_data_section(dst, data_char, before_data, m, n);
    // 4. Fill in extended zero
    memcpy(dst+ext_loc, ext_zeros, ext_len);
    // 5. Copy the after data section
    memcpy(dst+after_data, src+before_data, st_temp.st_size - before_data);
    // 6. Set Scale(s), some shapes need duplicated scales
    gptpu_utils::set_scale_in_tflite_model(dst, 
                                           scale, 
                                           scale_loc, 
                                           scale_num, 
                                           scale_stride_len);
    // Clean up
    munmap(src, st_temp.st_size);
    munmap(dst, out_file_size);
    close(temp_fd);
    close(out_fd);
    free(ext_zeros);
    free(data_char);
}


