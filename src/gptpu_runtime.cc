#include <assert.h>
#include "fifo.h"
#include "gptpu_utils.h"
#include "gptpu_runtime.h"

namespace gptpu_runtime{
    Runtime::Runtime(int verbose){
        // create FIFO with assertion
        assert((fifo_spmc = fifo_new(this->fifo_queue_size)) != nullptr);

        // Create node structure as fifo element
        this->fifo_nodes = new FifoNode[this->fifo_queue_size];

        // Init
        this->device_handler = new model_utils::DeviceHandler();
        this->total_dev_cnt = this->device_handler->list_devices(verbose); 
        pthread_cond_init(&this->end_cv, NULL);
        this->ack_cnt = 0;
        this->stop_all = false;
        this->done_enqueue = false;
    }
    
    Runtime::~Runtime(){
        delete this->fifo_nodes;
        fifo_close(this->fifo_spmc);
        delete this->device_handler;
        delete this->threads_args;
    }

    // Open devices and thier corresponding pthread. 
    void Runtime::span_consumers(unsigned int desired_dev_cnt, 
                                 int verbose){
        pthread_t thread_id[desired_dev_cnt];
        this->threads_args = new ThreadArgs[desired_dev_cnt];
        this->dev_cnt = desired_dev_cnt;

        for(unsigned int i = 0 ; i < desired_dev_cnt ; i++){
            this->device_handler->open_device(i, verbose);
            this->threads_args[i].tid = i; // naive assignment
            this->threads_args[i].stop_all = &this->stop_all;
            this->threads_args[i].done_enqueue = &this->done_enqueue;
            this->threads_args[i].fifo_spmc = this->fifo_spmc;
            this->threads_args[i].pmtx = &this->pmtx;
            this->threads_args[i].end_cv = &this->end_cv;
            this->threads_args[i].ack_cnt = &this->ack_cnt;
            this->threads_args[i].dev_cnt = &this->dev_cnt;
            this->threads_args[i].device_handler = this->device_handler;
            pthread_create(&thread_id[i], NULL, consumer_func, (void *)&this->threads_args[i]);
            //TODO: how to design pthread join
        }
    }
   
    /* signal end_cv on condition that if ack_cnt >= dev_cnt */ 
    void inline signaling_on_cond(int* ack_cnt, 
                                  int* dev_cnt, 
                                  pthread_cond_t* end_cv){
        (*ack_cnt)++; // has to be atomic across threads
        if((*ack_cnt) >= (*dev_cnt)){
            pthread_cond_signal(end_cv);
        }
    }
   
    /* signal handler after each fifo popping (atomic checking) */
    void inline signal_handling(int* ack_cnt,
                                int* dev_cnt,
                                pthread_cond_t* end_cv,
                                bool* done_enqueue,
                                bool* stop_all,
                                struct fifo* fifo_spmc,
				pthread_mutex_t* pmtx){
        pthread_mutex_lock(pmtx);
        if((*stop_all) == true){
            signaling_on_cond(ack_cnt, dev_cnt, end_cv);
        }else if(fifo_empty(fifo_spmc) && (*done_enqueue) == true){
            (*stop_all) = true;
	    signaling_on_cond(ack_cnt, dev_cnt, end_cv);
        }
        pthread_mutex_unlock(pmtx);
    }
    
    void* Runtime::consumer_func(void* _arg){
        // Getting args
        ThreadArgs* arg = ((ThreadArgs*)_arg);
        unsigned int tid = arg->tid;
        struct fifo* fifo_spmc = arg->fifo_spmc;
        pthread_mutex_t* pmtx = arg->pmtx;
        pthread_cond_t* end_cv = arg->end_cv;
        int* ack_cnt = arg->ack_cnt;
        int* dev_cnt = arg->dev_cnt;
        bool* stop_all = arg->stop_all;
        bool* done_enqueue = arg->done_enqueue;
        model_utils::DeviceHandler* device_handler = arg->device_handler;
        //init
        struct FifoNode* curr_node;

        while(true){
            // Found a valid node in enqueue that is ready to be consumed.
            if((curr_node = (struct FifoNode*)fifo_pop(fifo_spmc)) != NULL ){
		if(curr_node->op_name == "conv2d"){
                    device_handler->build_interpreter(tid, curr_node->model_id); 
                    // populate_input_chunking
                    int m, n, ldn;
                    curr_node->tensor_in->get_dims(m, n, ldn);
                    device_handler->populate_input(curr_node->tensor_in->tflite_data_array,
                                                   (m* n)/*array size*/,
                                                   curr_node->model_id);
                    // invoke_model
                    device_handler->model_invoke(curr_node->model_id, 1/*iter*/);
                    // simple_populate_output
                    curr_node->tensor_out->get_dims(m, n, ldn);
                    float scale;
                    uint8_t zero_point;
                    device_handler->get_raw_output(curr_node->tensor_out->
				    		       tflite_data_array,
                                                   (m * n),
                                                   curr_node->model_id,
                                                   zero_point,
                                                   scale);
                    // dequantization
                    uint8_t mean;
		    curr_node->tensor_out->get_scaling_params(scale, mean);
		    gptpu_utils::dequantization(curr_node->tensor_out->
				    		    tflite_data_array,
		                                curr_node->tensor_out->
						    float_output_array,
		 		                1/*depth*/,
					        m,
						n,
						ldn,
						scale,
						mean);
		}
            }else if(curr_node->op_name == "stall"){
                continue;
            }else{
                std::cout << __func__ << ": consumer " << tid << " receives "
                          << "unknown op name: " << curr_node->op_name
                          << std::endl;
                pthread_exit(NULL);
            }
            // ===== end OP signal handling (atomic) =====
            signal_handling(ack_cnt, 
                            dev_cnt, 
                            end_cv, 
                            done_enqueue, 
                            stop_all, 
                            fifo_spmc,
			    pmtx);
        } // end while
    }

    struct fifo* Runtime::get_fifo(){ return this->fifo_spmc; }
    struct FifoNode* Runtime::get_fifo_nodes(){ return this->fifo_nodes; }
    unsigned int Runtime::get_total_dev_cnt(){ return this->total_dev_cnt; }
    void Runtime::set_flag_atomic(bool* flag, bool val){
        pthread_mutex_lock(&this->pmtx);
        (*flag) = val;
        pthread_mutex_unlock(&this->pmtx);
    }
    bool Runtime::get_flag_atomic(bool* flag){
        bool ret;
        pthread_mutex_lock(&this->pmtx);
        ret = *flag;
        pthread_mutex_unlock(&this->pmtx);
        return ret;
    }

    void Runtime::wait_until_queue_is_finished(){
        pthread_mutex_lock(&this->pmtx);
        while(this->ack_cnt < this->dev_cnt){
            pthread_cond_wait(&this->end_cv, &this->pmtx);
        }
        pthread_mutex_unlock(&this->pmtx);
    }

    std::string Runtime::get_impl_kernel(std::string op){
        std::unordered_map<std::string, std::string>::const_iterator got = 
            this->kernel_table.find(op);
        if(got == this->kernel_table.end()){
            std::cout << __func__ << ": op: " << op 
                      << " has no implementation (yet)." << std::endl;
            std::exit(0);
        }else{
            return got->second;
        }
    }
}

