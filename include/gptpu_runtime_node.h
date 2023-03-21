#ifndef __GPTPU_RUNTIME_NODE_H__
#define __GPTPU_RUNTIME_NODE_H__

#include <string>
#include "gptpu_buffer.h"

struct FifoNode{
	std::string op_name;
    int model_id;
    	openctpu_buffer *tensor_in;
    	openctpu_buffer *tensor_out;
};

#endif
