#include <string.h>
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
  std::cout << "rank: " << std::endl;
  T max_rank  = 0, min_rank = 256;
  for(int i = 0 ; i < size ; i ++){
    if(i < threshold){ std::cout << rank[i] <<  " " ; }
    if(rank[i] > max_rank){
      max_rank = rank[i];
    }
    if(rank[i] < min_rank){
      min_rank = rank[i];
    }
    
  }
  std::cout << std::endl;
  std::cout << "max rank: " << max_rank << ", min rank: " << min_rank << ", diff: " << max_rank - min_rank << std::endl;
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
    irank[i] = 1;
  }
}

template <class T>
void generate_random_adj_matrix(double* adj_matrix, T* iadj_matrix, int size, unsigned int seed){
  // assume in row-major
  // randomly mark 1, indicating exist edge
  srand(seed);
  for(int i = 0 ; i < size*size ; i++){ 
    adj_matrix[i] = iadj_matrix[i] = rand()%2; 
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
  std::cout << __func__ << ": max: " << max << ", min: " << min << ", diff: " << (max-min) << ", scale: " << scale << std::endl;
// ===== scaling ===== 
  scale = 0; //scale / 255;
  std::cout << __func__ << ": max: " << max << ", min: " << min << ", diff: " << (max-min) << ", scale: " << scale << std::endl;
}

double diff(double* v1, double* v2, int size){
  double result = 0;
  for(int i = 0 ; i < size ; i++){
    result += fabs(v1[i] - v2[i]);
  }
  std::cout << __func__ << ": result: " << result << std::endl;
  return result;
}

double PageRank(int size){
  double* in_rank1 = (double*) malloc(size*sizeof(double));
  double* out_rank1= (double*) malloc(size*sizeof(double));
  double* temp1;
  double* w1       = (double*) malloc(size*size*sizeof(double));
 
  TYPE* in_rank2   = (TYPE*) malloc(size*sizeof(TYPE)); 
  TYPE* w2         = (TYPE*) malloc(size*size*sizeof(TYPE));

  initial_pagerank<TYPE>(in_rank1, in_rank2, size);
  generate_random_adj_matrix<TYPE>(w1, w2, size, seed); 
//  print_matrix<double>(w1, size);
// ===== iterative pagerank calculation (power method) =====
  timing ref_s = clk::now();
  int iter = 0;
  while(true){
    iter += 1;
    cblas_dgemv(CblasRowMajor, CblasNoTrans, size, size, 1, w1, size, in_rank1, 1, 0, out_rank1, 1);
    temp1 = in_rank1; in_rank1 = out_rank1; out_rank1 = temp1;
    if(diff(in_rank1, out_rank1, size) < 1E-7){
      break;
    }
  }
  timing ref_e = clk::now();
// ===== end reference =====================================
  double ref_us = std::chrono::duration_cast<std::chrono::nanoseconds>(ref_e - ref_s).count()/1000.0;
  printf("ref(cblas) time: %12.3f(us), get iter = %d\n", ref_us, iter);  

// print rank
  print_rank(in_rank1, size);
  print_rank(in_rank2, size);
}

int main(int argc, char* argv[]){
  if(argc != 2){
    std::cout << "argc = " << argc << std::endl;
    std::cout << "Usage: ./get_iteration [adj. matrix size]" << std::endl;
    return 1;
  }
  int idx = 1;
  int size = atoi(argv[idx++]); // matrix size
  PageRank(size);
  return 0;
}
