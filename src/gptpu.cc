#include "fifo.h"
#include "gptpu.h"
#include "gptpu_utils.h"
#include "model_utils.h"
#include "gptpu_buffer.h"
#include "gptpu_runtime.h"
#include <stdlib.h>
#include <iostream>

namespace openctpu{

#define itoa(x) std::to_string(x)

using namespace gptpu_utils;

// The GPTPU runtime class pointer 
gptpu_runtime::Runtime* gptpu_runtime;

void openctpu_init(unsigned int desired_dev_cnt, bool use_python, int verbose){
    gptpu_runtime = new gptpu_runtime::Runtime(verbose);
    gptpu_runtime->set_use_python_flag(use_python);

    unsigned int total_dev_cnt = gptpu_runtime->get_total_dev_cnt();
    assert(desired_dev_cnt > 0 && desired_dev_cnt <= total_dev_cnt);

    gptpu_runtime->span_consumers(desired_dev_cnt, verbose);
}

openctpu_dimension* openctpu_alloc_dimension(int rows, int cols, int ldn, bool transpose){
    openctpu_dimension* ret = new openctpu_dimension(transpose);
    ret->set_dims(rows, cols, ldn);
    return ret;
}

std::string inline python_based_model_creation(void* data,
					       int rows,
					       int cols,
					       std::string op,
					       std::string template_model_base,
					       std::string params_str){
    std::string out_path;        
    // save data weight for python-based tflite model generation
    // These two varaibles are random names
    std::string weight_file_path = "/home/mm2conv_weight.txt";
    std::string model_name = op + "_" + 
	    		     gptpu_utils::replace_delimiter(params_str, 
					     		    " ", 
							    "-"); 
    gptpu_utils::create_file_parent_path(weight_file_path);
    save_mm2conv_weight((float*)data, 
    	                weight_file_path,
		        rows,/*B*/
		        cols,/*C*/
		        rows,/*blk_r*/
		        cols,/*blk_c*/
		        0,/*i*/
		        0,/*j*/
		        1,/*row_blk_cnt*/
		        1,/*col_blk_cnt*/
		        0,/*inn_blk_rem*/
		        0/*col_blk_rem*/);
    /* create model file from python-based workflow: */
    std::string command = "python /home/src/Python/create_tflite_model.py \
                           --model "+op+" \
                           --saved_model_base "+template_model_base+" \
                           --in_shape "+itoa(rows)+" "+itoa(cols)+" \
                           --params "+params_str+"\
			   --weight_file_path "+weight_file_path;
    system(command.c_str());
    out_path = template_model_base + "/" + 
	       model_name + "/" + model_name + "_edgetpu.tflite";
    return out_path;
}

void inline CXX_based_model_creation(std::string template_path,
		                     std::string op,
				     std::string template_model_base,
				     int rows,
				     int cols,
				     std::string params_str,
				     openctpu_buffer* buf,
				     std::string example_path,
				     void* data,
				     std::string out_path){
    if(file_exist(template_path) == false){
	/* create template file: */
	std::cout << __func__ << ": template file: " << template_path
		  << " does not exist, creating one..." << std::endl;

   	// 1. generate a full example model with data
        std::string command = "python /home/src/Python/create_tflite_model.py \
                               --model "+op+" \
                               --saved_model_base "+template_model_base+" \
                               --in_shape "+itoa(rows)+" "+itoa(cols)+" \
                               --params "+params_str;
	system(command.c_str());
        // 2. create the template file from example model
        // TODO: separate out TemplateOffset setting as a member func. of 
        // openctpu_buffer, so that this func can be a static one.
        create_template_from_example_model(buf,
       		    		           op, 
                                           example_path, 
                                           template_path, 
                                           rows, 
                                           rows, 
                                           cols);
    }
    // create edgetpu file as a kernel model by filling template file with data
    // TODO: think of separating out giving ret 
    buf->create_model_file(template_path, out_path, data);
}

openctpu_buffer* openctpu_create_buffer(openctpu_dimension* dim,
                                        BufferType tensor_type,
                                        void* data){
    openctpu_buffer* ret = new openctpu_buffer(dim);
    int rows, cols, ldn;
    ret->get_dims(rows, cols, ldn);

    /*
     * op has to be known at this moment in order to determine 
     * proper buffer allocation and preparation.
     * As a demo project, now only mm2conv operator is supported.
     * */
    std::string op = "conv2d";
    std::string template_model_base = 
        "/home/kernels/templates/mm2conv/"+itoa(rows)+"x"+itoa(rows)+"x"+itoa(cols); 
    
    ret->set_buffer_type(tensor_type);
    std::string params_str    = get_params_string(op, rows, rows, cols);
    ret->set_params_str(params_str);

    if(tensor_type == MODEL){
    	/* Debug only: to enable Python-base model buffer generation for debugging, set this flag. */
    	bool use_python = gptpu_runtime->get_use_python_flag();

	/* potential arguments for selecting template:
           app_name, rows, cols, ldn.
           => a mm2conv mapping function
        */
        // TODO: need to think of a way to inform API that what is
        // the given M dimension
        // A Python-generated example model
        std::string example_path  = select_example_model_path(op, rows, rows, cols);
        // The template file without data section
        std::string template_path = select_template_path(op, rows, rows, cols);        
        // The output tflite file as kernel model
        std::string out_path      = define_kernel_path(op, rows, rows, cols); 
	
	if(use_python == true){
	    out_path = python_based_model_creation(data,
					           rows,
					           cols,
					           op,
					           template_model_base,
					           params_str);
	}else{
	    CXX_based_model_creation(template_path,
		                     op,
				     template_model_base,
				     rows,
				     cols,
				     params_str,
				     ret,
				     example_path,
				     data,
				     out_path);
	}
        assert(gptpu_runtime != nullptr);
	gptpu_runtime->device_handler->build_model(out_path, 0/*model_id*/);

    }else if(tensor_type == DATA){
        ret->config_input_partition();
        ret->convert_data_to_tflite_pattern(op, (float*)data);

    }else if(tensor_type == OUTPUT){
        ret->config_output_partition();
        ret->allocate_output_tflite_array(); // allocate placeholder for tflite output
        ret->float_output_array = (float*)data; // keep the pointer
    
    }else{
        std::cout << __func__
                  << ": undefined tensor_type: " 
                  << tensor_type << std::endl;
        std::abort();
    }
    return ret;
}

void openctpu_invoke_operator(const std::string op, 
                              openctpu_buffer* a,
                              openctpu_buffer* b, 
                              openctpu_buffer* c){

    assert(gptpu_runtime != nullptr);
    unsigned int idx = 0; // 0 ~ queue_size
    struct fifo* fifo_ptr       = gptpu_runtime->get_fifo();
    struct FifoNode *fifo_nodes = gptpu_runtime->get_fifo_nodes();
    
    // set out-tensor's params
    float a_scale, b_scale, c_scale;
    uint8_t a_mean, b_mean, c_mean = 0;
    a->get_scaling_params(a_scale, a_mean);
    b->get_scaling_params(b_scale, b_mean);
    c_scale = a_scale * b_scale;
    c->set_scaling_params(c_scale, c_mean);

    std::string kernel = gptpu_runtime->get_impl_kernel(op);

    // assign node attributes
    fifo_nodes[idx].op_name    = kernel;
    fifo_nodes[idx].model_id   = idx;
    fifo_nodes[idx].tensor_in  = b;
    fifo_nodes[idx].tensor_out = c;
    // more...
    // Enqueue
    fifo_push(fifo_ptr, &fifo_nodes[idx]);
    
    // signal the done enqueueing signal
    gptpu_runtime->set_done_enqueue_signal_atomic(true);

}

void openctpu_sync(){
    assert(gptpu_runtime != nullptr);
    gptpu_runtime->wait_until_queue_is_finished();
}

} // end namespace
