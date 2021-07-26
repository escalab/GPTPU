#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <gptpu.h>

typedef int TYPE;

float SCALE = 1.0;

// custom-defined kernels
void matrix_mul(openctpu_buffer *matrix_a,
                 openctpu_buffer *matrix_b,
                 openctpu_buffer *matrix_c){
  openctpu_invoke_operator("mm_model", matrix_a, matrix_b, matrix_c);
}

void matrix_add(openctpu_buffer *matrix_a,
                 openctpu_buffer *matrix_b,
                 openctpu_buffer *matrix_c){
  openctpu_invoke_operator("add_model", matrix_a, matrix_b, matrix_c);
}

template <class T>
void GEMM(T* a, T* b, T* c, int A, int B, int C, bool b_major){
  T sum;
  int b_offset;
  for(int i = 0 ; i < A; i++){
    for(int j = 0 ; j < C ; j++){
      sum = 0;
      for(int k = 0 ; k < B ; k++){
        b_offset = (b_major == true)?(j*B+k):(k*C+j);
        sum += a[i*B+k]*b[b_offset];
      }
      c[i*C+j] = sum;
    }
  }
}

int main(int argc, char **argv){
  TYPE *a, *b, *c, *ref_c;

  openctpu_dimension *matrix_a_d, *matrix_b_d, *matrix_c_d;
  openctpu_buffer    *tensor_a,   *tensor_b,   *tensor_c;
  int size = 1024;

// for int type: range is at most 256 since uint8_t input limitation
  int a_m = 0, a_range = 256, b_m = 0, b_range = 256;
  a     = (TYPE*) malloc(size*size*sizeof(TYPE));
  b     = (TYPE*) malloc(size*size*sizeof(TYPE));
  c     = (TYPE*) malloc(size*size*sizeof(TYPE));
  ref_c = (TYPE*) malloc(size*size*sizeof(TYPE));

// generate random input data
  for(int i = 0 ; i < size*size; i++){
    a[i] = rand()%a_range+a_m;
    b[i] = rand()%b_range+b_m;
//    a[i] = (float)rand()/(float)(RAND_MAX/a_range);
//    b[i] = (float)rand()/(float)(RAND_MAX/b_range);
  }  

// open device
/* 1. opening mode.  0: open device(s) sequentially starting from 1st dev.
                     1: open device(s) sequentially starting from a random dev (circular).
   2. # of devices you want to open. (constrainted by max # of devices available)
   Note: You cannot re-call this function to open other devices in system. (TODO: enable this feature)
*/
  openctpu_init(1, 1); 

// kernel-wise config
  // arguments 2 - 4 are only effective if data type is 0
  // if data tpe is 1, must be approx. mode with quantization
  auto config = openctpu_setConfig(0/*data type: 0: int, 1: float*/, true/*exact_mode*/, true/*mm256_mode*/, 8/*chunk_num*/);

  matrix_a_d = openctpu_alloc_dimension(2, size, size);
  matrix_b_d = openctpu_alloc_dimension(2, size, size);
  matrix_c_d = openctpu_alloc_dimension(2, size, size);

// optional flags are required for now, this API should detect type of array and do quantization for float type
  tensor_a = openctpu_create_buffer(matrix_a_d, a, config, false/*b_major*/, 0/*tensor_type*/); // tensor-wise argument
  tensor_b = openctpu_create_buffer(matrix_b_d, b, config, true            , 1);
  tensor_c = openctpu_create_buffer(matrix_c_d, c, config, false           , 2);

  openctpu_enqueue(matrix_mul/*kernel name*/, tensor_a, tensor_b, tensor_c);
  openctpu_sync(tensor_c); // wait until data is in c buffer, ready to peek the values

// reference
  GEMM<TYPE>(a, b, ref_c, size, size ,size, true/*b_major*/);

// verify
  openctpu_verify(tensor_c, ref_c, 2, size, size);

  return 0;
}
