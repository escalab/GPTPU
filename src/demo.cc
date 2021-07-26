#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <stdint.h>
#include <time.h>
#include <chrono>
#include <stdlib.h>
#include <sstream>  //for stringstream
#include <bitset>
#include <iostream>
#include <iomanip>
#include <cblas.h>
#include <gptpu.h>
#include <math.h> 

int PRINT_THRESHOLD = 64;
int SEED = 9487; 

typedef std::chrono::time_point<std::chrono::high_resolution_clock> timing;
typedef std::chrono::high_resolution_clock clk;

//void randFill(aligned_vector<int>& vec, int low, int high){
//  std::default_random_engine eng;
//  std::uniform_int_distribution<int> dis(low, high);
//  vstd::generate(vec.begin(), vec.end(), [&] { return dis(eng) } ); 
//}

void set_threshold(int t){
  PRINT_THRESHOLD = t;
}

template <class T>
void print_vector(T* a, int size){
  if(PRINT_THRESHOLD <= 0){ std::cout << "PRINT WARNING: threshold is 0 or smaller."; }
  std::cout << std::setprecision(7);
  for(int i = 0 ; i < size & i < PRINT_THRESHOLD ; i++){
    std::cout << a[i] << "\t";
  }
  std::cout << std::endl;
}

template <class T>
void print_matrix(T *a, int r, int c, bool colMajor){
  if(PRINT_THRESHOLD <= 0){ std::cout << "PRINT WARNING: threshold is 0 or smaller."; }
  std::cout << std::setprecision(7) << std::dec;
  int cnt = 0;
  int offset;
  for(int i = 0 ; i < MIN(10, r); i++){
    for(int j = 0 ; j < MIN(10, c) ; j++){
      if(cnt >= PRINT_THRESHOLD){
        std::cout << std::endl;
        return;
      }
      offset = (colMajor == false)?(i*c+j):(j*r+i);
      std::cout << a[offset] << "\t";
      cnt++;
    }
    std::cout << std::endl;
  }
}

int cpu_mac(int* a, int* b, int&c, int size){
  int sum = 0;
  for(int i = 0 ; i < size ; i++){
    sum += a[i] * b[i];
  }
  c = sum;
  return sum;
}

int cpu_mv(int* b, int* a, int *c, int A, int B){
  int sum;
  for(int i = 0 ; i < A ; i++){
    sum = 0;
    for(int j = 0 ; j < B ; j++){
      sum += (int)a[j]*b[i*B+j];
    }
    c[i] = sum;
  }
  return 0;
}

int cpu_smv(int* b, int* a, int *c, int A, int B, float scale){
  int sum;
  for(int i = 0 ; i < A ; i++){
    sum = 0;
    for(int j = 0 ; j < B ; j++){
      sum += (int)a[j]*b[i*B+j];
    }
    c[i] = (int)sum * scale;
  }
  return 0;
}

int cpu_vs(int* a, int* c, int A, float scale){
  for(int i = 0 ; i < A ; i++){
    c[i] = scale * a[i];
  }
  return 0;
}

int cpu_add(int* a, int* b, int* c, int A, int B){
  for(int i = 0 ; i < A*B ; i++){
    c[i] = a[i] + b[i];
  }
  return 0;
}

int cpu_sub(int* a, int* b, int* c, int A, int B){
  for(int i = 0 ; i < A*B ; i++){
    c[i] = a[i] - b[i];
  }
  return 0;
}
int cpu_mul(int* a, int* b, int* c, int A, int B){
  for(int i = 0 ; i < A*B ; i++){
    c[i] = a[i] * b[i];
    if(c[i] < 0){ std::cout << "c: " << c[i] << " = " << a[i] << " * " << b[i] << "( i: " << i << " )" << std::endl; }
  }
  return 0;
}

int cpu_maxpool(int* a, int* c, int A, int B){
  c[0] = INT_MIN;
  for(int i = 0 ; i < A*B ; i++){
     if(a[i] > c[0]){
       c[0] = a[i];
     }
  }
  return 0;
}

int cpu_mean(int* a, int* c, int A, int B){
  float avg = 0;
  long long int sum = 0;
  int cnt = 0;
  for(int i = 0 ; i < A*B ; i++){
    sum += a[i];
//     avg = (avg * cnt + a[i]) / (cnt + 1);
//     cnt++;
//     std::cout << "avg: " << avg << ", cnt: " << cnt << ", " << A << "x" << B << std::endl;
  }
  c[0] = sum / (A*B);
  std::cout << __func__ << ": mean: " << (double)sum/(double)(A*B) << ", c: " << c[0] << std::endl;
//  c[0] = avg;
  return 0;
}

int cpu_crop(int*a ,int* c, int A, int B, int blk_row, int blk_col, int start_i, int start_j){
  for(int i = 0; i < blk_row ; i++){
    for(int j = 0 ; j < blk_col ; j++){
      c[i*blk_col+j] = a[(i+start_i)*B+(j+start_j)]; 
    }
  }
  return 0;
}

int cpu_conv(int* a, int* f, int*c, int A, int B, int M, int N){
  int* a_pad = (int*) malloc((A+2)*(B+2)*sizeof(int)); 
  int A_pad = A+2;
  int B_pad = B+2;
  int pad_h = (A_pad - A)/2;
  int pad_w = (B_pad - B)/2;
//  std::cout << "a: " << std::endl;
//  print_matrix<int>(a, A, B, 0);
  // shifting inner data matrix
  for(int i = B-1 ; i >= 0 ; i--){
   memcpy(a_pad+((i+pad_w)*A_pad)+pad_h, a+(i*A), A*sizeof(int));
  }
  // actual same adding
  for(int i = 0 ; i < A_pad ; i++){
   for(int j = 0 ; j < B_pad ; j++){
     if(i < pad_h){                  // upper stride
       a_pad[i*B_pad+j] = a_pad[pad_h*B_pad+j];
     }else if(i >= (A_pad - pad_h)){ // lower stride
       a_pad[i*B_pad+j] = a_pad[(A_pad - pad_h - 1)*B_pad+j];
     }else if(j < pad_w){            // left  stride
       a_pad[i*B_pad+j] = a_pad[i*B_pad+pad_w];
     }else if(j >= (B_pad - pad_w)){ // right stride
       a_pad[i*B_pad+j] = a_pad[i*B_pad+(B_pad - pad_w - 1)];
     }else{ /* do nothing*/ }
   }
  }
//  std::cout << "a_pad: " << std::endl;
//  print_matrix<int>(a_pad, A_pad, B_pad, 0);
// start conv2D
  for(int i = 0 ; i < A; i++){
    for(int j = 0 ; j < B ; j++){
      int ai = i+1;
      int aj = j+1;
      c[i*B+j] = a_pad[ai*B_pad+aj]*f[4]+ // c
                 a_pad[(ai-1)*B_pad+aj]*f[1]+      // n
                 a_pad[ai*B_pad+(aj+1)]*f[5]+      // e
                 a_pad[ai*B_pad+(aj-1)]*f[3]+      // w
                 a_pad[(ai+1)*B_pad+aj]*f[7];      // s
    }
  }
   
}

int a_m = 0, a_range = 128, b_m = 0, b_range = 128;

void calculate_scale(int B, int a_range, int a_m, int b_range, int b_m, float& scale){
 //a[i] was rand()%a_range+a_m
 //b[i] was rand()%b_range+b_m
 // B is the length of summation vector
//TODO: select a proper scale value 
  int max_value = (b_range-1)+b_m;
  scale = 1;// (256/1024)/max_value ;//(1024/256)/(1); //255 * 0.1; // if you want the result to be enlarged 2 times, use 2* 255 = 510
  std::cout << "scale decided as: " << scale << std::endl;
}



void run(std::string& op, int tf, int gptpu, int cpu, int blas, int input_size, int output_size, bool b_transpose, int blk_row, int blk_col, int start_i, int start_j, int inner_size){
  int A = input_size;
  int B = output_size;  
  int C = 0;
  int D = 0;
  if(op == "mm"){
    B = inner_size;
    C = output_size;
  }else if(op == "crop"){
    C = blk_row;
    D = blk_col;
  }else if(op == "ext"){
    A = blk_row; 
    B = blk_col;
    C = input_size;
    D = output_size;
  }else if(op == "mpmm"){
    A = B = input_size;
  }
  int *a, *tpu_a, *cpu_a;
  int *b;
  float scale;
  int *cpu_c;
  int *tpu_c;
  int *tf_c;
  double *blas_a;
  double *blas_b;
  double *blas_c;

  timing tf_start;
  timing tf_end;
  timing tpu_start;
  timing tpu_end;
  timing cpu_start;
  timing cpu_end;
  timing blas_start;
  timing blas_end;
  
  std::cout << "start demo..." << std::endl;  
  srand(SEED);
  int complexity = 7;

  if(op == "mv"|| op == "bmv" || op == "imv"){
    blas_a = (double*) malloc(A*B*sizeof(double));
    blas_b = (double*) malloc(B*sizeof(double));
    blas_c = (double*) malloc(A*sizeof(double));

    a = (int*) malloc(A*B*sizeof(int));
    b = (int*) malloc(B*sizeof(int));
    cpu_c = (int*) malloc(A*sizeof(int));
    tpu_c = (int*) malloc(A*sizeof(int));
    tf_c  = (int*) malloc(A*sizeof(int));
    for(int i = 0 ; i < A*B ; i++){
      a[i] =  (rand() % 2)  ;//7 - (int)log10(rand()%100000000); //(rand() % 256);
//      if(rand()%10!=0)a[i] =0;
//      if(rand()%10 !=0){ a[i] = 0; }
//      if(i >= 1024*1023 && i < 1024*1024){
//        a[i] = 2;
//      }else{ 
//        a[i] = 0;
//      }
      blas_a[i] = (double)a[i];
    }
    for(int i = 0 ; i < B ; i++){
      b[i] =  (rand() % 2) ;//7 - (int)log10(rand()%100000000);//(rand() % 256); //(float)(pow(2, rand() % complexity - complexity));
//      if(rand()%10 !=0){ b[i] = 0; }
      blas_b[i] = (double)b[i];
    }
  }else if(op == "vs"){
    a = (int*) malloc(A*sizeof(int));
    scale = 2;
    cpu_c = (int*) malloc(A*sizeof(int));
    tpu_c = (int*) malloc(A*sizeof(int));
    tf_c  = (int*) malloc(A*sizeof(int));
    for(int i = 0 ; i < A ; i++){
      a[i] = 1; //(rand() % 10) * 0.1;
    }
  }else if(op == "mpmm"){
    a = (int*) malloc(A*B*sizeof(int));
    b = (int*) malloc(A*B*sizeof(int));
    tpu_c = (int*) malloc(A*B*sizeof(int));
  }else if(op == "add" || op == "sub" || op == "mul" || op == "tanh" || op == "relu"){
    a = (int*) malloc(A*B*sizeof(int));
    tpu_a = (int*) malloc(A*B*sizeof(int));
    cpu_a = (int*) malloc(A*B*sizeof(int));
    b = (int*) malloc(A*B*sizeof(int));
    cpu_c = (int*) malloc(A*B*sizeof(int));
    tpu_c = (int*) malloc(A*B*sizeof(int));
    tf_c  = (int*) malloc(A*B*sizeof(int));
    for(int i = 0 ; i < A*B ; i++){
      a[i] = rand()%32768; //rand() % 100+50;
      b[i] = rand()%32768;//50;
    }
  }else if(op == "maxpool" || op == "mean" || op == "max"){
    a = (int*) malloc(A*B*sizeof(int));
    cpu_c = (int*) malloc(sizeof(int));
    tpu_c = (int*) malloc(sizeof(int));
    tf_c  = (int*) malloc(sizeof(int));
    for(int i = 0 ; i < A*B ; i++){
      a[i] = rand()%8;
      if(i%2 == 1)
        a[i] = 97;
    }
    

  }else if(op == "mm"){
    blas_a = (double*) malloc(A*B*sizeof(double));
    blas_b = (double*) malloc(B*C*sizeof(double));
    blas_c = (double*) malloc(A*C*sizeof(double));

    a = (int*) malloc(A*B*sizeof(int));
    b = (int*) malloc(B*C*sizeof(int));
    cpu_c = (int*) malloc(A*C*sizeof(int));
    tpu_c = (int*) malloc(A*C*sizeof(int));
    tf_c  = (int*) malloc(A*C*sizeof(int));
  
    for(int i =  0 ; i < A*B ; i++){
      int idx_r = i/B;
      int idx_c = i%B; 
      a[i] = rand()%a_range+a_m ;// rand()%2;
//      if(rand()%10 != 0) a[i] = 0;
      
//      if(i == 0){ 
//        a[i] = rand()%8;
//      }else if(i == 257){
//        a[i] = rand()%1024;
//      }else if(i == 256){
//        a[i] = rand()%1024;
//      }else if(i == 1){
//        a[i] = rand()%8;
//      }else{
//        a[i] = 0;
//      }

//      if(idx_r < 1 && idx_c < 1){
//        a[i] = rand()%256;
//      }else{
//        a[i] = 0;
//      }
      blas_a[i] = (double)a[i];      
    }
    // assume here b is column-major
    for(int i =  0 ; i < B*C ; i++){
      int idx_r = i%C;
      int idx_c = i/C; 
      b[i] = rand()%b_range+b_m; //rand()%2; // + rand()%2 << 4;
//      if(i >= 0 && i < 256 ){b[i] = i;}
//      if(i == 0){ 
//        b[i] = rand()%8;
//      }else if(i == 257){
//        b[i] = rand()%1024;
//      }else if(i == 256){
//        b[i] = rand()%1024;
//      }else if(i == 1){
//        b[i] = rand()%8;
//      }else{
//        b[i] = 0;
//      }
//      b[i] = 1;
//      if(idx_r < 1 && idx_c < 1/*&& idx_c >= 0 && idx_c < 3*/){
//        b[i] = rand()%256;
//      }else{
//        b[i] = 0;
//      }
      blas_b[i] = (double)b[i];      
    }
  }else if(op == "crop" || op == "ext"){
    a = (int*) malloc(A*B*sizeof(int));
    cpu_c = (int*) malloc(C*D*sizeof(int));
    tpu_c = (int*) malloc(C*D*sizeof(int));
    tf_c  = (int*) malloc(C*D*sizeof(int));
    for(int i = 0 ; i < A*B ; i++){
      a[i] = rand() % 256;
    }
  }else if(op == "mac"){
    a = (int*) malloc(B*sizeof(int));
    b = (int*) malloc(B*sizeof(int)); 
    cpu_c = (int*) malloc(1*sizeof(int));
    tpu_c = (int*) malloc(1*sizeof(int));
    for(int i = 0 ; i < B ; i++){
      a[i] = rand() % 2;
      b[i] = rand() % 2;
      if(rand()%20!=0)b[i]=0;
    }
  }else if(op == "conv"){
    a = (int*) malloc(A*B*sizeof(int));
    b = (int*) malloc(3*3*sizeof(int));
    cpu_c = (int*) malloc(A*B*sizeof(int));
    tpu_c = (int*) malloc(A*B*sizeof(int));
    for(int i = 0 ; i < A*B ; i++){
      a[i] = rand()%5; //rand() % 2;
    }
    for(int i = 0 ; i < 9 ; i++){
      b[i] = (i == 0 || i == 2 || i == 6 || i == 8)?0:1;
    }
    b[1] = b[3] = b[5] = b[7] = 1/*76*/;
    b[4] = 1/*255*/;
  }else if(op == "black"){
  }else{
    std::cout << "undefined operation name: " << op << std::endl;
    exit(0);
  }

  if(tf == 1){
    tf_start = clk::now();
    if(op == "mv"){
      std::cout << "tf    result: " << gptpu_tf_mv(a, b, tf_c ,A, B) << std::endl;
    }else if(op == "imv"){
      std::cout << "tf    result: " << gptpu_tf_mv(a, b, tf_c ,A, B) << std::endl;
    }else if(op == "bmv"){
      std::cout << "tf    result: " << gptpu_tf_bmv(a, b, tf_c ,A, B) << std::endl;
    }else if(op == "vs"){
      std::cout << "tf    result: " << gptpu_tf_vs(a, tf_c ,A, scale) << std::endl;
    }else if(op == "add"){
      std::cout << "tf    result: " << gptpu_tf_add(a, b, tf_c ,A, B) << std::endl;
    }else if(op == "sub"){
      std::cout << "tf    result: " << gptpu_tf_sub(a, b, tf_c ,A, B) << std::endl;
    }else if(op == "mul"){
      std::cout << "tf    result: " << gptpu_tf_mul(a, b, tf_c ,A, B) << std::endl;
    }else if(op == "maxpool"){
      std::cout << "tf    result: " << gptpu_tf_maxpool(a, tf_c ,A, B) << std::endl;
    }else if(op == "mean"){
      std::cout << "tf    result: " << gptpu_tf_mean(a, tf_c ,A, B) << std::endl;
    }else if(op == "mm"){ // a x b(col-major) = c
      std::cout << "tf    result: " << gptpu_tf_mm(a, b, tf_c, A, B, C, b_transpose) << std::endl;
    }else if(op == "crop"){
      std::cout << "crop  result: " << gptpu_tf_crop(a, tf_c, A, B, blk_row, blk_col, start_i, start_j, false) << std::endl;
    }
    tf_end  = clk::now();
  }
  if(gptpu == 1){
    tpu_start = clk::now();
    if(op == "mv"){
      std::cout << "tpu   result: " << gptpu_mv(a, b, tpu_c ,A, B) << std::endl;
    }else if(op == "black"){
      std::cout << "tpu   result: " << gptpu_black(a, b, tpu_c, A, B) << std::endl;
    }else if(op == "imv"){
      std::cout << "tpu   result: " << gptpu_imv(a, b, tpu_c ,A, B) << std::endl;
    }else if(op == "vs"){
      std::cout << "tpu   result: " << gptpu_vs(a, tpu_c ,A, scale) << std::endl;
    }else if(op == "mm"){  // A x B = C
      float scale = 1;
      calculate_scale(B, a_range, a_m, b_range, b_m, scale);
      set_scale(scale);
      std::cout << "tpu   result: " << gptpu_mm(a, b, tpu_c, A, B, C, b_transpose) << std::endl;
    }else if(op == "mpmm"){
      std::cout << "mpmm  result: " << gptpu_mpmm("~/GPTPU/src/del.tflite", a, b, tpu_c, A, B, 1);
      exit(0);
    }else if(op == "add"){
      std::cout << "tf    result: " << gptpu_add(a, b, tpu_c, A, B) << std::endl;
    }else if(op == "sub"){
      std::cout << "tf    result: " << gptpu_sub(a, b, tpu_c, A, B) << std::endl;
    }else if(op == "mul"){
      std::cout << "tf    result: " << gptpu_mul(a, b, tpu_c, A, B) << std::endl;
    }else if(op == "tanh"){
      std::cout << "tf    result: " << gptpu_tanh(a, tpu_c, A, B) << std::endl;
    }else if(op == "relu"){
      std::cout << "tf    result: " << gptpu_relu(a, tpu_c, A, B) << std::endl;
    }else if(op == "mean"){
      std::cout << "tf    result: " << gptpu_mean(a, tpu_c, A, B) << std::endl;
    }else if(op == "max"){
      std::cout << "tf    result: " << gptpu_max(a, tpu_c, A, B) << std::endl;
    }else if(op == "crop"){
      std::cout << "crop  result: " << gptpu_crop(a, tpu_c, A, B, blk_row, blk_col, start_i, start_j, false) << std::endl;
    }else if(op == "ext"){
      std::cout << "crop  result: " << gptpu_ext(a, tpu_c, A, B, C, D, start_i, start_j, false) << std::endl;
    }else if(op == "mac"){
      std::cout << "mac   result: " << gptpu_mac(a, b, tpu_c[0], B) << std::endl;
      std::cout << "tpu_c[0]: " << tpu_c[0] << std::endl;
    }else if(op == "conv"){
      std::cout << "conv  result: " << gptpu_conv2D(a, b/*f*/, tpu_c, A, B, 3, 3, /*padding*/"replication") << std::endl;
    }
    tpu_end   = clk::now();
  }
  if(cpu == 1){
    cpu_start = clk::now();
    if(op == "mv" || op == "bmv"){
      std::cout << "cpu   result: " << cpu_mv(a, b, cpu_c ,A, B) << std::endl;
    }else if(op == "vs"){
      std::cout << "cpu   result: " << cpu_vs(a, cpu_c ,A, scale) << std::endl;
    }else if(op == "add"){
      std::cout << "cpu   result: " << cpu_add(a, b, cpu_c ,A, B) << std::endl;
    }else if(op == "sub"){
      std::cout << "cpu   result: " << cpu_sub(a, b, cpu_c ,A, B) << std::endl;
    }else if(op == "mul"){
      std::cout << "cpu   result: " << cpu_mul(a, b, cpu_c ,A, B) << std::endl;
    }else if(op == "maxpool"){
      std::cout << "cpu   result: " << cpu_maxpool(a, cpu_c, A, B) << std::endl;
    }else if(op == "mean"){
      std::cout << "cpu   result: " << cpu_mean(a, cpu_c, A, B) << std::endl;
    }else if(op == "crop"){
      std::cout << "crop result: " << cpu_crop(a, cpu_c, A, B, blk_row, blk_col, start_i, start_j) << std::endl;
    }else if(op == "mac"){
      std::cout << "mac   resut: " << cpu_mac(a, b, cpu_c[0], B) << std::endl;
    }else if(op == "conv"){
      std::cout << "conv  resut: " << cpu_conv(a, b/*f*/, cpu_c, A, B, 3, 3) << std::endl;
    }
    cpu_end   = clk::now();
  }
  if(blas == 1){
    blas_start = clk::now();
    if(op == "mv" || op == "bmv"){    //matrix: blas_b[A][B], vector: blas_a[B], vector: blas_c[A]
      cblas_dgemv(CblasRowMajor, CblasNoTrans, A, B, 1, blas_a, B, blas_b, 1, 0, blas_c, 1 );
    }
    if(op == "mm"){ // A x Bt = C
      timing gemms = clk::now();
      cblas_dgemm(CblasRowMajor, CblasNoTrans, (b_transpose != 0)?CblasTrans:CblasNoTrans, A, C, B, 1, blas_a, B, blas_b, B, 1, blas_c, C);
      timing gemme = clk::now();
      double gemmus = std::chrono::duration_cast<std::chrono::nanoseconds>(gemme-gemms).count()/1000.0;
      printf("the cblas_demm time is : %12.3f (us) , size: %d, %d, %d\n", gemmus, A, B, C);  
    }
    blas_end   = clk::now();
  }
  // Verify output vector 
  std::cout << "verifying..." << std::endl;
  int errors = 0;
  int N = 0;
  if(op == "mv" || op == "bmv"){
    N = A;
  }else if(op == "vs"){
    N = A;
  }else if(op == "add" || op == "sub" || op == "mul" || op == "tanh" || op == "relu"){
    N = A*B;
  }else if(op == "maxpool" || op == "mean"){
    N = 1;
  }else if(op == "mm"){
    N = A*C;
  }else if(op == "crop"){
    N = C*D;
  }else if(op == "mac"){
    N = 1;
  }else if(op == "conv"){
    N = A*B;
  }else{
    std::cout << "undefined operation name." << std::endl;
  }
  int v1, v2, MAX=INT_MIN, MIN=INT_MAX;
  if(op == "mac"){
    printf("cpu_c: %d, tpu_c:%d\n", cpu_c[0], tpu_c[0]);
  }

  if(op == "conv"){
    print_matrix<int>(a, A, B, 0);
    print_matrix<int>(b,3,3,0);
    print_matrix<int>(tpu_c, A, B, 0);
  }

  double square_sum_avg = 0;
  for(int i = 0 ; i < N ; i++){
    v1 = (op == "mm" || op == "mv")?(int)blas_c[i]:(int)cpu_c[i];//(int)blas_c[i]; //cpu_c[i];
//    if(op == "mm" || op == "mv"){
//      v1 = (int)blas_c[i];
//    }
    v2 = tpu_c[i]; //(gptpu == 1)?tpu_c[i]:((tf == 1)?tf_c[i]:cpu_c[i]);
    //if((op == "add" || op == "sub" || op == "mul" || op == "crop") && (gptpu == 0)){
    //  v2 = tf_c[i];
    //}

    square_sum_avg = (square_sum_avg * i + pow((v1 - v2), 2)) / (i + 1);
    if(v1 != v2){
//      printf("c[%d,%d] is wrong, ans:%d , tpu_c:%d\n", i/A, i%A, v1, tpu_c[i]);
      errors++;
    }else{
//      std::cout << "v1: " << v1 << ", v2: " << v2 << ", tf_c: " << tf_c[i] << std::endl;
    }
    if(v1 > MAX){ MAX = v1; }
    if(v1 < MIN){ MIN = v1; }
  }
  std::cout << "the valid scale is: " << scale/255 << std::endl;
  if(errors > 0){
    std::cout << "verify fails, errors = " << errors << "(" << (N) << ")" << std::endl;
  }else{
    std::cout << "verify pass!" << " (MAX: " << MAX << ", MIN: " << MIN << ")" << std::endl;
  }

  if(1){
    std::cout << "===== data =====" << std::endl;
    if(op == "mv"){
      std::cout << "matrix: (size =  " << A << "x" << B << ")" << std::endl;
      print_matrix<int>(a, A, B, 0/*1 for column major*/);
      std::cout << "input vector: (size = " << B << ")" << std::endl;  
      print_vector<int>(b, B);
    }else if(op == "add" || op == "mul" || op == "sub"){ 
      std::cout << "input vector: (size = " << A << "x" << B << ")" << std::endl;  
      print_vector<int>(a, N);
      std::cout << "input vector: (size = " << A << "x" << B << ")" << std::endl;  
      print_vector<int>(b, N);
      std::cout << "tpu_c: " << std::endl;
      print_vector<int>(tpu_c, N);
      std::cout << "cpu_c: " << std::endl;
      print_vector<int>(cpu_c, N);
    }else if(op == "tanh" || op == "relu"){
      std::cout << "input vector: (size = " << A << "x" << B << ")" << std::endl;  
      print_vector<int>(a, N);
    }else if(op == "maxpool" || op == "mean"){
      std::cout << "input vector: (size = " << A << "x" << B << ")" << std::endl;  
      print_vector<int>(a, A*B);
    }else if(op == "mm"){
      std::cout << "input matrix A: (size = " << A << "x" << B << ")" << std::endl;
      print_matrix<int>(a, A, B, 0);
      std::cout << "input matrix B: (size = " << B << "x" << C << ")" << std::endl;
      print_matrix<int>(b, B, C, b_transpose);
    }else if(op == "crop"){
      std::cout << "input matrix A: (size = " << A << "x" << B << ")" << std::endl;
      print_matrix<int>(a, A, B, 0);
    }else if(op == "mac"){
      std::cout << "input vector A: (size = " << B << ")" << std::endl;
      print_vector<int>(a, B);
      std::cout << "input vector B: (size = " << B << ")" << std::endl;
      print_vector<int>(b, B);
    }else if(op == "conv"){
      std::cout << "input  matrix A: (size = " << A << "x" << B << ")" << std::endl;
      print_matrix<int>(a, A, B, b_transpose);
      std::cout << "filter matrix B: (size = " << 3 << "x" << 3 << ")" << std::endl;
      print_matrix<int>(b, 3, 3, b_transpose);
    }else{
      std::cout << "input vector: (size = " << A << "x" << B << ")" << std::endl;  
      print_vector<int>(a, N);
    }
    if(op == "mm"){
      std::cout << "cpu_c: " << std::endl;
      print_matrix<int>(cpu_c, A, C, 0);
      std::cout << "tpu_c: " << std::endl;
      print_matrix<int>(tpu_c, A, C, 0);
//      std::cout << "tf_c: " << std::endl;
//      print_matrix<int>(tf_c, A, C, 0);
      std::cout << "blas_c: " << std::endl;
      print_matrix<double>(blas_c, A, C, 0);
    }else if(op == "mv"|| op == "sub"){
      std::cout << "cpu_c: " << std::endl;
      print_vector<int>(cpu_c, N);
      std::cout << "tpu_c: " << std::endl;
      print_vector<int>(tpu_c, N);
      std::cout << "tf_c: " << std::endl;
      print_vector<int>(tf_c, N);
      if(op == "mv"){
        std::cout << "blas_c: " << std::endl;
        print_vector<double>(blas_c, N);
      }
    }else if(op == "crop"){    
      std::cout << "tf_c: " << std::endl;
      print_matrix<int>(tf_c, C, D, 0);
      std::cout << "cpu_c: " << std::endl;
      print_matrix<int>(cpu_c, C, D, 0);
    }else if(op == "maxpool" || op == "mean"){
      std::cout << "cpu_c: " << cpu_c[0] << std::endl; 
      std::cout << "tpu_c: " << tpu_c[0] << std::endl;
      std::cout << "tf_c: "  << tf_c[0]  << std::endl;
    }else if(op == "mac"){
      std::cout << "cpu_c: " << cpu_c[0] << std::endl;
      std::cout << "tpu_c: " << tpu_c[0] << std::endl;
    }else if(op == "conv"){
      std::cout << "tpu_c: " << std::endl;
      print_matrix<int>(tpu_c, A, B, 0);
      std::cout << "cpu_c: " << std::endl;
      print_matrix<int>(cpu_c, A, B, 0);
    }else if(op == "tanh" || op == "relu"){
      std::cout << "tpu_c: " << std::endl;
      print_matrix<int>(tpu_c, A, B, 0);
    }
  }
// === calculate blas_C acerage ===
  double avg = 0;
  double rate = 0;
  for(int i = 0 ; i < N ; i++){ // monitor rolling average
    avg = (avg*(i) + blas_c[i]) / (i+1);
    rate = (rate * i + (fabs(tpu_c[i] - blas_c[i]))) / (i+1);
  }
  double RMSE = sqrt(square_sum_avg);
  std::cout << "blas_c: max: " << MAX << ", MIN: " << MIN << std::endl;
  std::cout << "RMSE: " << RMSE << ", blas_c avg: " << avg << ", RMSE pecentage: " << (RMSE/avg)*100 << "%" << ", error rate: " << (rate/avg)*100 << "%" << std::endl;
  std::cout << "bla_c range percentage over avg: " << ((double)(MAX - MIN)/avg)*100 << "%" << std::endl;

  if(errors > 0){
    std::cout << "verify fails, errors = " << errors << "(" << (N) << ")" << std::endl;
  }else{
    std::cout << "verify pass!" << " (MAX: " << MAX << ", MIN: " << MIN << ")" << std::endl;
  }

  std::cout << "===== time =====" << std::endl;
  if(tf == 1){
    printf("tf elapsed time   : %12.3f (us).\n", std::chrono::duration_cast<std::chrono::nanoseconds>(tf_end - tf_start).count()/1000.0);
  }
  if(gptpu == 1){
    printf("gptpu elapsed time: %12.3f (us).\n", std::chrono::duration_cast<std::chrono::nanoseconds>(tpu_end - tpu_start).count()/1000.0);
  }
  if(cpu == 1){
    printf("cpu elapsed time  : %12.3f (us).\n", std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_end - cpu_start).count()/1000.0);
  }
  if(blas == 1){
    printf("blas elapsed time : %12.3f (us).\n", std::chrono::duration_cast<std::chrono::nanoseconds>(blas_end - blas_start).count()/1000.0);
  }
  std::cout << "free up..." << std::endl;
  free(a);
  if(op != "maxpool" && op != "mean" && op != "crop"){
    free(b);
  }
  if(op == "mm" || op == "mv" || op == "imv" || op == "smv"){
    free(blas_a);
    free(blas_b);
    free(blas_c);
  }
  free(tf_c);
  free(tpu_c);
  free(cpu_c);
}

union ker{
  unsigned int i;
  float f;
};

int main(int argc, char* argv[]){
  if(argc != 30){
    std::cout << "argc = " << argc << std::endl;
    std::cout << "Usage: ./demo [operation name] b<tf> b<gptpu> b<cpu> b<blas> [input size] [inner_size] [output size] b<b_tranpose> [crop_row_len] [crop_col_len] [start_i] [start_j] [tpu_id] [iteration] [dev_cnt] [MM_BLK_ROW] [MM_BLK_COL] [verbose] [breakdown] [data_type] [scale] [zero_point] [start_chunk] [chunk_size] [chunk_num] [exact_mode] [threshold] [ramdisk]" << std::endl;
    std::cout << "mm     : requires [input size] [inner size] [output size] and b<b_tranpose>, return matrix" << std::endl;
    std::cout << "mv     : requires [input size] [output size], return vector" << std::endl;
    std::cout << "mac    : requires [input size]              , return scalar" << std::endl;
    std::cout << "vs     : requires [input size]              , return vector" << std::endl;
    std::cout << "add    : requires [input size] [output size], return matrix" << std::endl;
    std::cout << "sub    : requires [input size] [output size], return matrix" << std::endl;
    std::cout << "mul    : requires [input size] [output size], return matrix" << std::endl;
    std::cout << "mean   : requires [input size] [output size], return scale" << std::endl;
    std::cout << "maxpool: requires [input size] [output size], return scale" << std::endl;
    std::cout << "crop   : requires [input size] [output size] [crop_row_len] [crop_col_len] [start_i] [start_j], return matrix" << std::endl;
    std::cout << "please give dummy size for unused parameter(s)" << std::endl;
    return 1;
  }
  int idx = 1;
  std::string op   = argv[idx++]; // operation name
  int tf           = atoi(argv[idx++]); // enable tf-generating version
  int gptpu        = atoi(argv[idx++]); // enable gptpu version
  int cpu          = atoi(argv[idx++]); // enable cpu version
  int blas         = atoi(argv[idx++]); // enable openblas version in double type
  int input_size   = atoi(argv[idx++]);
  int inner_size   = atoi(argv[idx++]);
  int output_size  = atoi(argv[idx++]);
  bool b_transpose = atoi(argv[idx++]);
  int blk_row_len  = atoi(argv[idx++]);
  int blk_col_len  = atoi(argv[idx++]);
  int start_i      = atoi(argv[idx++]);
  int start_j      = atoi(argv[idx++]);
  int tpuid        = atoi(argv[idx++]);
  int iter         = atoi(argv[idx++]);
  int dev_cnt      = atoi(argv[idx++]);
  int MM_BLK_ROW   = atoi(argv[idx++]);
  int MM_BLK_COL   = atoi(argv[idx++]);
  int verbose      = atoi(argv[idx++]);
  int breakdown    = atoi(argv[idx++]);
  std::string data_type = argv[idx++];
  float s          = atof(argv[idx++]); 
  int zp           = atoi(argv[idx++]);
  int start_chunk  = atoi(argv[idx++]);
  int chunk_size   = atoi(argv[idx++]);
  int chunk_num    = atoi(argv[idx++]);
  int exact_mode   = atoi(argv[idx++]);
  int threshold    = atoi(argv[idx++]);
  int ramdisk      = atoi(argv[idx++]);
  
  set_start_chunk(start_chunk);
  set_chunk_size(chunk_size);
  set_chunk_num(chunk_num);
  set_exact_mode(exact_mode);
  set_data_type(data_type);
  std::cout << __func__ << ", set SCALE:  " << s << std::endl;
  set_scale(s);
  set_zp(zp);
  set_tpu_id(tpuid);
  set_iteration(iter);
  set_verbose(verbose);
  set_breakdown(breakdown);
  set_dev_cnt(dev_cnt);
  set_block(MM_BLK_ROW, MM_BLK_COL);
  set_threshold(threshold);
  set_ramdisk(ramdisk);

  run(op, tf, gptpu, cpu, blas, input_size, output_size, b_transpose, blk_row_len, blk_col_len, start_i, start_j, inner_size);
  std::cout << "done ./demo microbenchmark" << std::endl;
  return 0;
}
