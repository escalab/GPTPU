#ifndef __GPTPU_RUNTIME_H__
#define __GPTPU_RUNTIME_H__
#include "fifo.h"
#include "model_utils.h"
#include "gptpu_runtime_node.h"
#include <pthread.h>
#include <unordered_map>

namespace gptpu_runtime{
#define QUEUE_SIZE 65536

    struct ThreadArgs{
        unsigned int tid;
        // pointers to common flags/structures
        bool* stop_all;
        bool* done_enqueue;
        int* ack_cnt;
        int* dev_cnt; // desired(opened) dev cnt
        struct fifo* fifo_spmc; // common pointer
        pthread_mutex_t* pmtx;
        pthread_cond_t* end_cv;
        model_utils::DeviceHandler* device_handler;
    };

    class Runtime{
        public:
            Runtime(int verbose);
    	    ~Runtime();
            model_utils::DeviceHandler* device_handler;
            void span_consumers(unsigned int desired_dev_cnt, 
                                int verbose);
            struct fifo* get_fifo();
            struct FifoNode* get_fifo_nodes();
    	    unsigned int get_total_dev_cnt();
            /* 
               To inform runtime that producer is done with enqueuing task(s). 
             */
            void set_done_enqueue_signal_atomic(bool flag){
                this->set_flag_atomic(&this->done_enqueue, flag); 
            };
            void wait_until_queue_is_finished();
	    void set_use_python_flag(bool val){ this->use_python = val; };
	    bool get_use_python_flag(){ return this->use_python; };
        std::string get_impl_kernel(std::string op);
        
        private:
	    // To generate tflite modele using python-base script or not
	    bool use_python = false;
	    unsigned int fifo_queue_size = QUEUE_SIZE;
            unsigned int total_dev_cnt;
            static void* consumer_func(void* arg);
            void set_flag_atomic(bool* flag, bool val);
            bool get_flag_atomic(bool* flag);
            struct fifo *fifo_spmc; //FIFO strcture 
            struct FifoNode* fifo_nodes; // per FIFO node structure

	    // Signals
            int ack_cnt; // running device count that are done with consuming.
            int dev_cnt; // desired(opened) dev cnt
            bool stop_all; // To force all consumers to finish if set.
            bool done_enqueue; // Producer to signal runtime that it's done
            pthread_mutex_t pmtx;
            pthread_cond_t end_cv; // conditional variable
            ThreadArgs* threads_args;

	    /* The op-to-kernel mapping table:
	     * Some op's underlying implementation kernel is different than
	     * op itself.
	     * Ex: "mm_model" is implmented by "conv2d" which uses the concept
	     * of mm2conv transformation.
	     * */	    
            std::unordered_map<std::string, std::string> kernel_table = {
		{"mm_model", "conv2d"}
            };
    };
} // end namespace

#endif

