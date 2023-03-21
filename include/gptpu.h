#ifndef __GPTPU_H__
#define __GPTPU_H__

#include <string>
#include "gptpu_dim.h"
#include "gptpu_buffer.h"

// A list of public openctpu APIs

namespace openctpu{

void openctpu_init(unsigned int desired_dev_cnt, bool use_python, int verbose);

openctpu_dimension* openctpu_alloc_dimension(int rows, int cols, int ldn, bool transpose);

openctpu_buffer* openctpu_create_buffer(openctpu_dimension* dim,
                                        BufferType tensor_type,
					                    void* data);

template <typename Func, typename ... Args>
void openctpu_enqueue(Func func, Args&&... args){ func(std::forward<Args>(args)...); };

/*
   For generalizability:
    
   template <typename... Args>
    void openctpu_invoke_operator(const std::string op, Args... args);
*/
void openctpu_invoke_operator(const std::string op, openctpu_buffer* a, openctpu_buffer* b, openctpu_buffer* c);

void openctpu_sync();

}
#endif
