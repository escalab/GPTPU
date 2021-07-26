#include <float.h>
#include <omp.h>
#include <string.h>
#include <fstream>
//#include <algorithm>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <cblas.h>
#include <typeinfo>
#include <vector>
#include <climits>
#include <gptpu.h>
#include <math.h>

// ===== 'double' for testing integer behavior; 'int' for actual gptpu run ===== //
#define TYPE int
unsigned int seed = 9487; //time(NULL);  // seed for generating adj. matrix
int threshold = 30;
typedef std::chrono::time_point<std::chrono::high_resolution_clock> timing;
typedef std::chrono::high_resolution_clock clk;

double scale = 1;

template <class T>
T get_max(T* rank, int size){
  T max = 0;
  for(int i = 0 ; i < size ; i++){
    if(rank[i] > max){ max = rank[i]; }
  }
  return max;
}

template <class T>
T get_min(T* rank, int size){
  T min = 1;
  for(int i = 0 ; i < size ; i++){
    if(rank[i] < min){ min = rank[i]; }
  }
  return min;
}

double get_mean(double* rank, int size){
  double mean = 0;
  for(int i = 0 ; i < size ; i++){
    mean += rank[i];
  }
  scale = (double)mean/size;
  return (double)mean/size;
}

template <class T>
void print_rank(T* rank, int size){
 // std::cout << "rank: " << std::endl;
  T max_rank  = 0, min_rank = 256;
  for(int i = 0 ; i < size ; i ++){
//    if(i < threshold){ std::cout << rank[i] <<  " " ; }
    if(rank[i] > max_rank){
      max_rank = rank[i];
    }
    if(rank[i] < min_rank){
      min_rank = rank[i];
    }
    
  }
//  std::cout << std::endl;
//  std::cout << "max rank: " << max_rank << ", min rank: " << min_rank << ", diff: " << max_rank - min_rank << std::endl;
}

template <class T>
void scale_rank(T* rank, int size){
  T max_rank = 0;
  for(int i = 0 ; i < size; i++){
    if(rank[i] > max_rank){
      max_rank = rank[i];
    }
  } 
  int MAX = CHAR_MAX ; // CHAAR_MAX
  if(max_rank > MAX){
    for(int i = 0 ; i < size; i++){
      rank[i] = (int)(rank[i] * MAX / max_rank);
    }
//    std::cout << "scale rank: " << MAX/max_rank << std::endl;
  }
//  std::cout << "no scale, max: " << max_rank << std::endl;
}

template <class T>
void print_matrix(T* adj, int size){
  std::cout << "matrix: " << std::endl;
  for(int i = 0; i < size ; i++){
    for(int j = 0 ; j < size ; j++){
      std::cout << adj[i*size+j] << " ";
    }
    std::cout << std::endl;
  }
}

template <class T>
void initial_pagerank(double* rank, T* irank, int size){
  for(int i = 0 ; i < size ; i++){
    rank[i]  = (double)1/size;
    irank[i] = rank[i]; //1;
  }
}

template <class T>
void generate_random_adj_matrix(double* adj_matrix, T* iadj_matrix, int size, unsigned int seed){
  // assume in row-major
  // randomly mark 1, indicating exist edge
  srand(seed);
  for(int i = 0 ; i < size*size ; i++){ 
    adj_matrix[i] = iadj_matrix[i] = ((rand()%20) == 0)? 1 : 0; 
//    if(rand()%10 != 0){
//      adj_matrix[i] = iadj_matrix[i] = 0; 
//    }
  }
  // count how many edges for each node, and update to correct weight
  int cnt = 0;
  for(int j = 0 ; j < size ; j++){ // loop over column
    cnt = 0;
    for(int i = 0 ; i < size ; i++){ // count how many edges out of this node
       if( adj_matrix[i*size+j] != 0 ){
         cnt += 1;
       }
    }
    for(int i = 0 ; i < size ; i++){ // update every 1's to  '1/cnt'
      if(adj_matrix[i*size+j] != 0){
        adj_matrix[i*size+j] = (double)1/cnt;
      }
    }
  } 
}

void eval_clustering(double* rank1, int* rank2, int size){
  std::cout << __func__ << std::endl;
  float* rank2_scale  = (float*) malloc(size*sizeof(float));
  bool touch; 
  int num_overlap = 0;
  double previous_max;
  double max_rank1;
  double min_rank1;
  int max_rank2 = get_max<int>(rank2, size);
  int min_rank2 = get_min<int>(rank2, size);
  float* cluster_rank = (float*) malloc((max_rank2-min_rank2+1)*sizeof(float)); 

  for(int i = min_rank2 ;  i < max_rank2 ; i++){
    touch = false;
    max_rank1 = get_min<double>(rank1, size);
    min_rank1 = get_max<double>(rank1, size);
    for(int idx =  0 ; idx < size ; idx++){
      if(rank2[idx] == i){
        touch = true;
        if(rank1[idx] > max_rank1){ max_rank1 = rank1[idx]; } 
        if(rank1[idx] < min_rank1){ min_rank1 = rank1[idx]; } 
      }
    }
    if(touch){
      if(min_rank1 < previous_max){ 
        num_overlap += 1; 
//        std::cout << "min rank1: " << min_rank1 << ", previous max rank1: " << previous_max << std::endl;
      }
      //std::cout << "for cluster " << i << ": max rank1: " << max_rank1 << ", min rank1: " << min_rank1 << std::endl;
      printf("for cluster %3d: max rank1: %12.9f, min rank1: %12.9f\n", i, max_rank1, min_rank1);
      cluster_rank[i] = (float)(max_rank1 + min_rank1)/(float)2;
      previous_max = max_rank1;
    }
  }
  for(int i = min_rank2 ;  i < max_rank2 ; i++){
    for(int idx =  0 ; idx < size ; idx++){
      if(rank2[idx] == i){
        rank2_scale[idx] = cluster_rank[i];
      }
    }
  }

//  std::cout << "# of cluster range overlap: " << num_overlap << std::endl;
//TODO: calculate RMSE between rank1, rank2_scale
  double MSE = 0;
  double rate = 0;
  double rank1_mean = 0;
  for(int i = 0 ; i < size ; i++){
//printf("rank1: %f, rank2_scale: %f\n", rank1[i], rank2_scale[i]);
    MSE = (MSE * i + pow(rank1[i] - rank2_scale[i], 2)) / (i + 1);
    rank1_mean = (rank1_mean * i + rank1[i]) / (i + 1);
    rate = (rate * (double)i + fabs(rank1[i] - rank2_scale[i])) / (i + 1);
    if(i < 10){
      printf("rank1[%d]: %f, rank2_scale[%d]: %f, diff: %f, rate = %f\n", i, rank1[i], i, rank2_scale[i], fabs(rank1[i] - rank2_scale[i]), rate);
    }
  }
  printf("RMSE: %f, rank1_avg: %f, RMSE%%: %f %%, error rate: %f %% (rate: %ff)\n", sqrt(MSE), rank1_mean, (sqrt(MSE)/rank1_mean)*100, (rate/rank1_mean)*100, rate);

}

void get_error(double* a, double* b, int size){
  double MSE = 0;
  double rate = 0;
  double rank1_mean = 0;
  for(int i = 0 ; i < size ; i++){
//printf("rank1: %f, rank2_scale: %f\n", rank1[i], rank2_scale[i]);
    MSE = (MSE * i + pow(a[i] - b[i], 2)) / (i + 1);
    rank1_mean = (rank1_mean * i + a[i]) / (i + 1);
    rate = (rate * (double)i + fabs(a[i] - b[i])) / (i + 1);
    if(i < 10){
//      printf("a[%d]: %f, b[%d]: %f, diff: %f, rate = %f\n", i, a[i], i, b[i], fabs(a[i] - b[i]), rate);
    }
  }
  printf("RMSE: %f, rank1_avg: %f, RMSE%%: %f %%, error rate: %f %% (rate: %ff)\n", sqrt(MSE), rank1_mean, (sqrt(MSE)/rank1_mean)*100, (rate/rank1_mean)*100, rate);

}


template <class T>
double compute_norm(double* rank1, T* rank2, int size, int degree){
  double sum = 0;
  for(int i = 0 ; i < size ; i++){
    sum += std::pow(abs(rank1[i] - (double)rank2[i]*scale), degree);
//    std::cout << "rank1 :" << rank1[i] << ", rank2: " << rank2[i] << ", scale: " << scale << std::endl;
  }
  return std::pow(sum, 1.0/degree);
}

void auto_scaling(int* rank, int size, double& scale){
// find max/min
  int max = INT_MIN;
  int min = INT_MAX;
  for(int i = 0 ; i < size ; i++){
    if(rank[i] > max){ max = rank[i]; }
    if(rank[i] < min){ min = rank[i]; }
  }
//  std::cout << __func__ << ": max: " << max << ", min: " << min << ", diff: " << (max-min) << ", scale: " << scale << std::endl;
// ===== scaling ===== 
  scale = 0; //scale / 255;
//  std::cout << __func__ << ": max: " << max << ", min: " << min << ", diff: " << (max-min) << ", scale: " << scale << std::endl;
}

struct quantize_params{
  uint8_t mean;
  double scale;
};

void quantize_array_double2int(double* in, double* out, int size, uint8_t& mean, double& scale){
 // for each out[i], in[i] = (out[i] - mean ) / scale
  double in_max = DBL_MIN;
  double in_min = DBL_MAX;
  for(int i = 0 ; i < size ; i++){
    if(in[i] > in_max){ in_max = in[i]; }
    if(in[i] < in_min){ in_min = in[i]; }
  }
//  std::cout << __func__ << ": in_max: " << in_max << ", in_min: " << in_min << std::endl;
  if((in_max - in_min) != 0){
    scale = ((double)UCHAR_MAX/*255.0 - 0.0*/) / (in_max - in_min);
    mean  = (-1.0) * (int)(in_min * scale);
    for(int i = 0 ; i < size ; i++){
      out[i] = (double)(((in[i] - in_min)/(in_max - in_min)) * (double)UCHAR_MAX);
    }
  }else{
    scale = UCHAR_MAX / in_max;
    mean = 0;
    for(int i = 0 ; i < size ; i++){
      out[i] = UCHAR_MAX;
    }
  }
}

void array_dequan(double* in, double* out, int size){
  double max = DBL_MIN;
  double min = DBL_MAX;
  double range = 0;
  for(int i = 0 ;  i < size ; i++){
    if(in[i] > max){ max = in[i]; }
    if(in[i] < min){ min = in[i]; }
  }
  range = max - min;
  if(range == 0){
    for(int i =  0 ; i < size; i++){
      out[i] = in[i];
    }
  }else{
    for(int i =  0 ; i < size; i++){
      out[i] = (double)((int)(((in[i] - min) / range) * UCHAR_MAX));
    }
  }
}

double PageRank(int size, int iter, int degree/*norm degree*/){
  double* in_rank1 = (double*) malloc(size*sizeof(double));
  double* in_rank1_dup = (double*) malloc(size*sizeof(double));
  double* out_rank1= (double*) malloc(size*sizeof(double));
  double* temp1;
  double* w1       = (double*) malloc(size*size*sizeof(double));
  double* w1_int   = (double*) malloc(size*size*sizeof(double));
  
  double* in_rank_tpu  = (double*) malloc(size*sizeof(double));
  double* out_rank_tpu = (double*) malloc(size*sizeof(double));
  double* tmp_rank_tpu = (double*) malloc(size*sizeof(double));

  TYPE* in_rank2   = (TYPE*) malloc(size*sizeof(TYPE));
  TYPE* out_rank2  = (TYPE*) malloc(size*sizeof(TYPE));
  TYPE* temp2;
  TYPE* w2         = (TYPE*) malloc(size*size*sizeof(TYPE));
  //int iter = (size == 1024)?5:((size == 2048)?4:(size == 4096)?4:(size == 8192)?3:1/*default value*/); 

  initial_pagerank<TYPE>(in_rank1, in_rank2, size);
// duplicate in rank
  for(int i = 0 ; i < size ; i++){
    in_rank1_dup[i] = in_rank1[i];
  }
  generate_random_adj_matrix<TYPE>(w1, w2, size, seed); 
// ===== read weight matirx from binary, which created from src/create_model.py =====
//  std::ifstream fin("./pagerank_1K_iter5_weight.txt", std::ios::binary);
//  if(!fin.is_open()){
//    std::cout << " file is not opened." << std::endl;
//    exit(0);
//  }
//  fin.read(reinterpret_cast<char*>(&w1), size*size*sizeof(float));

//  FILE *ptr;
//  float* tmp = (float*) malloc(size*size*sizeof(float));
//  ptr = fopen("./pagerank_1K_iter5_weight.txt", "rb");
//  fread(tmp, sizeof(float), size*size, ptr);
//  for(int i = 0 ; i < size*size ; i++){
//    w1[i] = tmp[i];
//  }

//  print_matrix<double>(w1, size);
// ===== iterative pagerank calculation (power method) =====
//  printf("w1 (for cblas_dgemv):\n");
//  for(int i = 0 ; i < 10 ; i++){
//    for(int j = 0 ; j < 10 ; j++){
//      std::cout << w1[i*size+j] << " ";
//    }
//    std::cout << std::endl;
//  }
  timing ref_s = clk::now();


  
//  omp_set_num_threads(8);
  //printf("# threads: %d\n", omp_get_thread_num());
//  for(int j = 0 ; j < 10 ; j++){
//    std::cout <<" in_rank1[" << j << "]: " << in_rank1[j] << std::endl;
//  }
  for(int i = 0 ; i < iter ; i++ ){   // rank =  matrix * rank;
//#pragma omp parallel
  //  for(int j = 0 ; j < 8 ; j++){
      cblas_dgemv(CblasRowMajor, CblasNoTrans, size, size, 1, w1, size, in_rank1, 1, 0, out_rank1, 1);
  //  }
    temp1 = in_rank1; in_rank1 = out_rank1; out_rank1 = temp1;
//    for(int j = 0 ; j < 10 ; j++){
//      std::cout << "iter " << i << ": cblas_dgemv out_rank[" << j << "]: " << in_rank1[j] << std::endl;
//    }
  }
  timing ref_e = clk::now();

  double MEAN  = 0;
  double RANGE = 0;
  double total = 0;
  double max  = DBL_MIN;
  double min = DBL_MAX;
  for(int i = 0 ;  i< size; i++){
    if(in_rank1[i] > max){ max = in_rank1[i]; }
    if(in_rank1[i] < min){ min = in_rank1[i]; }
    total += in_rank1[i];
  }
  MEAN = total / (double)size;
  RANGE = max - min;

// ===== end reference =====================================
  quantize_params in_rank_params;
  quantize_params w_rank_params;
//  quantize_array_double2int(w1, w1_int, size*size, w_rank_params.mean, w_rank_params.scale);


//  for(int i = 0 ;  i < 10 ; i++){
//    std::cout << "in_rank1_dup[" << i << "]: " << in_rank1_dup[i] << std::endl;
//  }
//  quantize_array_double2int(in_rank1_dup, in_rank_tpu, size, in_rank_params.mean, in_rank_params.scale);
//  for(int it = 0 ; it < iter ; it++){
//    for(int i = 0 ;  i < 10 ; i++){
//      std::cout << "in_rank_tpu[" << i << "]: " << in_rank_tpu[i] << std::endl;
//    }
//    cblas_dgemv(CblasRowMajor, CblasNoTrans, size, size, 1, w1_int, size, in_rank_tpu, 1, 0, out_rank_tpu, 1);
//    for(int i = 0 ; i < size ; i++){
//      tmp_rank_tpu[i] = out_rank_tpu[i];
//    }
///    array_dequan(tmp_rank_tpu, in_rank_tpu, size);
//    for(int i = 0 ; i < size ; i++){
//      in_rank_tpu[i] = tmp_rank_tpu[i];
//    }
//  }
  for(int i = 0 ; i < size ; i++){
    tmp_rank_tpu[i] = in_rank_tpu[i];
  }
// rescael to range===========
  array_dequan(tmp_rank_tpu, out_rank_tpu, size);
//  for(int i =  0 ; i < 10 ; i++){
//    std::cout << "[before scaling]out_rank_tpu[" << i << "]: " << out_rank_tpu[i] << std::endl;
//  }
  double _MEAN  = 0;
  double _RANGE = 0;
  double _total = 0;
  double _max  = DBL_MIN;
  double _min = DBL_MAX;
  for(int i = 0 ;  i< size; i++){
    if(out_rank_tpu[i] > _max){ _max = out_rank_tpu[i]; }
    if(out_rank_tpu[i] < _min){ _min = out_rank_tpu[i]; }
    _total += out_rank_tpu[i];
  }
  _MEAN = _total / (double)size;
  _RANGE = _max - _min;
  for(int i =  0 ; i < size ; i++){
    out_rank_tpu[i] = ((out_rank_tpu[i] - _MEAN) / _RANGE)*RANGE + MEAN;
  }
//  double total = 0;
//  for(int i = 0 ; i < size ; i++){
//    total += out_rank_tpu[i];
//  }
//  for(int i = 0 ; i < size ; i++){
//    out_rank_tpu[i] = out_rank_tpu[i]/total;
//  }

  for(int i =  0 ; i < 10 ; i++){
 //   std::cout << "[after scaling]out_rank_tpu[" << i << "]: " << out_rank_tpu[i] << std::endl;
  }


// ===== multi-layer model design ==========================
//  std::cout << std::endl;
//  timing ms = clk::now();
//  gptpu_pagerank(w1, in_rank_tpu, out_rank_tpu, size, iter); // # of iteration is determined offline.  # of iteration = f(w2, in_rank2, size)
//  std::cout << "out_rank_tpu:" << std::endl;
//  for(int i = 0 ; i < 10 ; i++){
//    std::cout << out_rank_tpu[i] << " " << std::endl;
//  }
//  timing me = clk::now();
//  double multi_us = std::chrono::duration_cast<std::chrono::nanoseconds>(me - ms).count()/1000.0;
//  printf("gptpu_pagerank time: %12.3f (us)\n", multi_us);
// =========================================================
// ===== GPTPU impl. part ==================================
  timing s = clk::now();
  double scale = 0.1; //(double)(4/(1.2*size));
  set_dev_cnt(2);
  set_block(1024, 1024);
  if(size >= 4096){
    set_block(4096, 4096);
  }
  set_breakdown(1);
  for(int i = 0 ; i < iter ; i++ ){   // rank =  matrix * rank;
//    cblas_dgemv(CblasRowMajor, CblasNoTrans, size, size, 1, w1, size, in_rank1, 1, 0, out_rank1, 1);
    gptpu_mv(w2, in_rank2, out_rank2, size ,size /*magic number*/);
//    scale_rank<TYPE>(out_rank2, size);
//    print_rank<double>(out_rank1, size);
//    print_rank<TYPE>(out_rank2, size);
    // swap in_rank and out_rank
//    temp2 = in_rank2; in_rank2 = out_rank2; out_rank2 = temp2;
//    auto_scaling(in_rank2, size, scale);
  }
  timing e = clk::now();
// ===== GPTPU impl. part end ==============================
  double ref_us = std::chrono::duration_cast<std::chrono::nanoseconds>(ref_e - ref_s).count()/1000.0;
  double the_us = std::chrono::duration_cast<std::chrono::nanoseconds>(e - s).count()/1000.0;
  printf("gptpu: %12.3f(us) | ref(cblas): %12.3f(us)\n", the_us, ref_us);  

// print rank
  print_rank(in_rank1, size);
  print_rank(in_rank2, size);

  scale = get_mean(in_rank1, size);

//  eval_clustering(in_rank1, in_rank2, size);  
  get_error(in_rank1, out_rank_tpu, size);
  return compute_norm(in_rank1, /*in_rank2*/out_rank_tpu, size, degree);

}

int main(int argc, char* argv[]){
  if(argc != 4){
    std::cout << "argc = " << argc << std::endl;
    std::cout << "Usage: ./pageRank [adj. matrix size] [iter] [num]" << std::endl;
    return 1;
  }
  int idx = 1;
  int size = atoi(argv[idx++]); // matrix size
  int iter = atoi(argv[idx++]); // # of iteration of power method
  int num  = atoi(argv[idx++]); // run [num] times the same setting to get average error measurement
  int norm_degree = 2; //error measurement
  double norm_error = 0;
  set_block(size, size);
  for(int i = 0 ; i < num ; i++){
    norm_error += PageRank(size, iter, norm_degree);
  }
//  std::cout << "Average L" << norm_degree << " Norm: " << ((double)norm_degree/num) << " over " << num << " time(s)." << std::endl;
  return 0;}
