

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
//#include <cuda.h>
#include <sys/time.h>
#include <chrono>
// includes, kernels
//#include "backprop_cuda_kernel.cu"
#include "backprop.h"
typedef std::chrono::time_point<std::chrono::high_resolution_clock> timing;
typedef std::chrono::high_resolution_clock clk;

////////////////////////////////////////////////////////////////////////////////

//extern "C"
void bpnn_layerforward(float *l1, float *l2, float **conn, int n1, int n2);

//extern "C"
void bpnn_output_error(float *delta, float *target, float *output, int nj, float *err);

//extern "C"
void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no, float **who, float *hidden, float *err);

//extern "C" 
void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly, float **w, float **oldw);
void bpnn_adjust_weights_int(float *delta, int ndelta, float *ly, int nly, float **w, float **oldw);


//extern "C"
int setup(int argc, char** argv);

extern "C"
float **alloc_2d_dbl(int m, int n);

extern "C"
float squash(float x);

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

unsigned int num_threads = 0;
unsigned int num_blocks = 0;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
	setup(argc, argv);
}


//extern "C"
void bpnn_train_cuda(BPNN *net, float *eo, float *eh)
{
  BPNN* net_int = new BPNN(*net); // deep copy

  int in, hid, out;
  float out_err, hid_err;
  float out_err_int, hid_err_int;
  
  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;   
  

  double cpu_us = 0, bpnn_us = 0, adj_us = 0, gpu_us = 0;  
   
//#ifdef GPU  
  int m = 0;
  float *input_hidden_cuda;
  float *input_cuda;
  float *output_hidden_cuda;
  float *partial_sum;
  float *hidden_partial_sum;
  float *hidden_delta_cuda;
  float *input_prev_weights_cuda;
  float sum;
  float *input_weights_one_dim;
  float *input_weights_prev_one_dim;
  num_blocks = in / 16;  
  dim3  grid( 1 , num_blocks);
  dim3  threads(16 , 16);
  
  input_weights_one_dim = (float *) malloc((in + 1)* (hid + 1) * sizeof(float));
  input_weights_prev_one_dim = (float *) malloc((in + 1)* (hid + 1) * sizeof(float));
  partial_sum = (float *) malloc(num_blocks * WIDTH * sizeof(float));
 
  // this preprocessing stage is added to correct the bugs of wrong memcopy using two-dimensional net->inputweights
  for (int k = 0; k <= in; k++) {	
   for (int j = 0; j <= hid; j++) {
	  input_weights_one_dim[m] = net->input_weights[k][j];
	  input_weights_prev_one_dim[m] = net-> input_prev_weights[k][j];
	  m++;
    }
  }
  
//  cudaMalloc((void**) &input_cuda, (in + 1) * sizeof(float));
//  cudaMalloc((void**) &output_hidden_cuda, (hid + 1) * sizeof(float));
//  cudaMalloc((void**) &input_hidden_cuda, (in + 1) * (hid + 1) * sizeof(float));
//  cudaMalloc((void**) &hidden_partial_sum, num_blocks * WIDTH * sizeof(float));
  
  
//#endif


#ifdef GPU
 
  printf("Performing GPU computation\n");
  
  //printf("in= %d, hid = %d, numblocks = %d\n", in, hid, num_blocks);
  
  cudaMemcpy(input_cuda, net->input_units, (in + 1) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(input_hidden_cuda, input_weights_one_dim, (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);

  
  timing gpu_s = clk::now();
  bpnn_layerforward_CUDA<<< grid, threads >>>(input_cuda,
	                                          output_hidden_cuda,
											  input_hidden_cuda,
											  hidden_partial_sum,
											  in,
											  hid);
 
  cudaThreadSynchronize();
  timing gpu_e = clk::now();
  gpu_us += std::chrono::duration_cast<std::chrono::nanoseconds>(gpu_e-gpu_s).count()/1000.0;  
  
//  cudaError_t error = cudaGetLastError();
//	if (error != cudaSuccess) {
//		printf("bpnn kernel error: %s\n", cudaGetErrorString(error));
//		exit(EXIT_FAILURE);
//	}
  
  cudaMemcpy(partial_sum, hidden_partial_sum, num_blocks * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);
     
  gpu_s = clk::now();
  for (int j = 1; j <= hid; j++) {
    sum = 0.0;
    for (int k = 0; k < num_blocks; k++) {	
      sum += partial_sum[k * hid + j-1] ;
    }
	sum += net->input_weights[0][j];
	net-> hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
  }
  gpu_e = clk::now();
  gpu_us += std::chrono::duration_cast<std::chrono::nanoseconds>(gpu_e-gpu_s).count()/1000.0;  
#endif

//#ifdef CPU

  printf("Performing CPU computation\n");
  timing cpu_s = clk::now();
  timing bpnn_s = clk::now();
  bpnn_layerforward(net->input_units, net->hidden_units,net->input_weights, in, hid);
//#endif
  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
  timing bpnn_e = clk::now();
  bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);  
  timing adj_s = clk::now();
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);

//#ifdef CPU

  bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in, net->input_weights, net->input_prev_weights);
  timing adj_e = clk::now();

  timing cpu_e = clk::now();

  printf("Performing CPU_int computation\n");
  bpnn_layerforward(net_int->input_units, net_int->hidden_units,net_int->input_weights, in, hid);
//#endif
  bpnn_layerforward(net_int->hidden_units, net_int->output_units, net_int->hidden_weights, hid, out);
  bpnn_output_error(net_int->output_delta, net_int->target, net_int->output_units, out, &out_err_int);
  bpnn_hidden_error(net_int->hidden_delta, hid, net_int->output_delta, out, net_int->hidden_weights, net_int->hidden_units, &hid_err_int);  
  bpnn_adjust_weights_int(net_int->output_delta, out, net_int->hidden_units, hid, net_int->hidden_weights, net_int->hidden_prev_weights);

//#ifdef CPU

  bpnn_adjust_weights_int(net_int->hidden_delta, hid, net_int->input_units, in, net_int->input_weights, net_int->input_prev_weights);

  bpnn_us = std::chrono::duration_cast<std::chrono::nanoseconds>(bpnn_e-bpnn_s).count()/1000.0;  
  adj_us = std::chrono::duration_cast<std::chrono::nanoseconds>(adj_e-adj_s).count()/1000.0;  
  cpu_us = std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_e-cpu_s).count()/1000.0;  
//#endif  


#ifdef GPU

  cudaMalloc((void**) &hidden_delta_cuda, (hid + 1) * sizeof(float));
  cudaMalloc((void**) &input_prev_weights_cuda, (in + 1) * (hid + 1) * sizeof(float));

  cudaMemcpy(hidden_delta_cuda, net->hidden_delta, (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(input_prev_weights_cuda, input_weights_prev_one_dim, (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(input_hidden_cuda, input_weights_one_dim, (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);

  gpu_s = clk::now();
  bpnn_adjust_weights_cuda<<< grid, threads >>>(hidden_delta_cuda,  
												hid, 
												input_cuda, 
												in,
												input_hidden_cuda, 
												input_prev_weights_cuda
												);

  gpu_e = clk::now();
  gpu_us += std::chrono::duration_cast<std::chrono::nanoseconds>(gpu_e-gpu_s).count()/1000.0;  
  cudaMemcpy(net->input_units, input_cuda, (in + 1) * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(input_weights_one_dim, input_hidden_cuda, (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyDeviceToHost);

// ===== error measurement =====================
// size: hid, in
// CPU: net->input_weights
// GPU: input_hidden_cuda 
  double avg = 0 ;
  double rate = 0;
  double square_sum_avg = 0;
  int idx = 0;
  for(int i = 0 ; i < in ; i++){
    for(int j = 0 ; j < hid ; j++){
      idx = i*hid+j;
      avg = (avg*idx + net->input_weights[i][j])/(idx+1);
      rate = (rate*idx+(fabs(net->input_weights[i][j] - net_int->input_weights[i][j] )))/(idx+1);
      square_sum_avg = (square_sum_avg * idx + pow((net->input_weights[i][j] - net_int->input_weights[i][j]), 2)) / (idx+1);
 //      printf("idx: %d, avg: %f, rate: %f, square_sum_avg: %f\n", idx, avg, rate, square_sum_avg);
    }
  }
  double RMSE = sqrt(square_sum_avg);
  printf("RMSE: %f, CPU avg: %f, RMSE%%: %f %%, error rate: %f %%\n", RMSE, avg, (RMSE/avg)*100, (rate/avg)*100);
    
  cudaFree(input_cuda);
  cudaFree(output_hidden_cuda);
  cudaFree(input_hidden_cuda);
  cudaFree(hidden_partial_sum);
  cudaFree(input_prev_weights_cuda);
  cudaFree(hidden_delta_cuda);
  
  free(partial_sum);
  free(input_weights_one_dim);
  free(input_weights_prev_one_dim);

#endif   
  
  
  printf("CPU  time: %12.3f (us)\n", cpu_us);
  printf("bpnn time: %12.3f (us)\n", bpnn_us);
  printf("adj  time: %12.3f (us)\n", adj_us);
  printf("ther time: %12.3f (us)\n", cpu_us - bpnn_us - adj_us);
  printf("GPU  time: %12.3f (us)\n", gpu_us);
}
