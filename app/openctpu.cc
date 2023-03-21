#include <chrono>
#include <math.h>
#include <time.h>
#include <cblas.h>
#include <fcntl.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include "gptpu.h"
#include "quality.h"

typedef std::chrono::time_point<std::chrono::high_resolution_clock> timing;
typedef std::chrono::high_resolution_clock clk;

using namespace openctpu;

// self-defined GPTPU kernels
void matrix_mul(openctpu_buffer *matrix_a,
                openctpu_buffer *matrix_b,
                openctpu_buffer *matrix_c, float coef){
    openctpu_invoke_operator("mm_model", matrix_a, matrix_b, matrix_c);
}

// CPU baseline implementation
void gemm_kernel_cpu(float* a, float* b, float* c, int A, int B, int C, bool b_major){
    timing start = clk::now();
    cblas_sgemm(CblasRowMajor, 
                CblasNoTrans, 
                CblasNoTrans, 
                A, B, C, 
                1/*alpha*/, a, B/*lda*/, b, C/*ldb*/, 1/*beta*/, c, C/*ldc*/);
    timing end = clk::now();
    double ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()/1000000.0;
    std::cout << "[demo] CPU baseline(cblas_gemm) time  : " << ms << " (ms)" << std::endl;
}

int main(int argc, char **argv){
    float *a, *b, *c, *ref_c;

    openctpu_dimension *matrix_a_d, *matrix_b_d, *matrix_c_d;
    openctpu_buffer    *tensor_a,   *tensor_b,   *tensor_c;
    if(argc != 4){
        printf("usage: %s [problem size] [scale] [use_python]\n", argv[0]);
        exit(0);
    }
    int size = atoi(argv[1]);
    float scale = atof(argv[2]);
    bool use_python = atoi(argv[3]);

    // for int type: range is at most 256 since uint8_t input limitation
    int a_m = 0, a_range = 256, b_m = 0, b_range = 256;
    a     = (float*) malloc(size*size*sizeof(float));
    b     = (float*) malloc(size*size*sizeof(float));
    c     = (float*) malloc(size*size*sizeof(float));
    ref_c = (float*) malloc(size*size*sizeof(float));

    // random seed
    time_t t;
    srand((unsigned) time(&t));
    
    std::cout << "[demo] generating random input data..." << std::endl; 
    // generate random input data
    for(int i = 0 ; i < size; i++){
        for(int j = 0 ; j < size ; j++){
            a[i*size+j] = (float)rand()/(float)(RAND_MAX/a_range);
            b[i*size+j] = (float)rand()/(float)(RAND_MAX/b_range);
        }
    }
    std::cout << "[demo] input matrix data range: 0 ~ " << a_range << std::endl; 

    openctpu_init(1/*desired tpu count*/, 
		  use_python/*use python script to generate tflite model as MODEL buffer*/, 
		  0/*verbose*/); 

    std::cout << "[demo] allocating tensor dim..." << std::endl;
    matrix_a_d = openctpu_alloc_dimension(size, size, size/*ldn*/, false);
    matrix_b_d = openctpu_alloc_dimension(size, size, size       , false);
    matrix_c_d = openctpu_alloc_dimension(size, size, size       , false);

    std::cout << "[demo] allocating tensor buffer..." << std::endl;
    tensor_a = openctpu_create_buffer(matrix_a_d, MODEL, a); 
    tensor_b = openctpu_create_buffer(matrix_b_d, DATA, b);
    tensor_c = openctpu_create_buffer(matrix_c_d, OUTPUT, c);

    // ===== Main kernel calling =====
    std::cout << "[demo] enqueue kernel..." << std::endl;
    timing start = clk::now();
    
    openctpu_enqueue(matrix_mul, tensor_a, tensor_b, tensor_c, scale);
    
    openctpu_sync(); // wait until GPTPU completes the task
    
    timing end = clk::now();
    double ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()/1000000.0;
    std::cout << "[demo] EdgeTPU end-to-end latency time: " << ms << " (ms)" << std::endl;;
    // ===============================

    // reference - call baseline implementation
    gemm_kernel_cpu(a, b, ref_c, size, size, size, false/*b_major*/);

    // verify
    Quality* quality = new Quality(size, size, size/*ldn*/, c, ref_c);
    quality->print_results(1/*verbose*/);
    delete quality;
    return 0;
}
