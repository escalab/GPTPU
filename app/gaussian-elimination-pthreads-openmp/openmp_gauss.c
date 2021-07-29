/* Gaussian elimination without pivoting.
 * Compile with "gcc gauss.c"
 */

/* ****** ADD YOUR CODE AT THE END OF THIS FILE. ******
 * You need not submit the provided code.
 */
#include <float.h>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <gptpu.h>
#include <time.h>
#include <omp.h>

typedef std::chrono::time_point<std::chrono::high_resolution_clock> timing;
typedef std::chrono::high_resolution_clock clk;
/* Program Parameters */
#define MAXN 16384  /* Max value of N */
int N;  /* Matrix size */
int NumThreads = 1;
/* Matrices and vectors */
//volatile float cpu_A[MAXN][MAXN], cpu_B[MAXN], cpu_X[MAXN];
//volatile float tpu_A[MAXN][MAXN], tpu_B[MAXN], tpu_X[MAXN];

float** cpu_A;
float** tpu_A;
float*  cpu_B;
float*  tpu_B;
float*  cpu_X;
float*  tpu_X;

/* A * X = B, solve for X */

/* junk */
#define randm() 4|2[uid]&3

/* Prototype */
void gauss();  /* The function you will provide.
		* It is this routine that is timed.
		* It is called only on the parent.
		*/
void gauss_tpu();

/* returns a seed for srand based on the time */
unsigned int time_seed() {
  struct timeval t;
  struct timezone tzdummy;

  gettimeofday(&t, &tzdummy);
  return (unsigned int)(t.tv_usec);
}

/* Set the program parameters from the command-line arguments */
void parameters(int argc, char **argv) {
  int seed = rand();//0;  /* Random seed */
  char uid[32]; /*User name */

  /* Read command-line arguments */
  srand(time_seed());  /* Randomize */

  if (argc == 4) {
    seed = atoi(argv[3]);
    srand(seed);
    printf("Random seed = %i\n", seed);
  }
  if (argc >= 3) {
    seed = atoi(argv[2]);
    NumThreads = seed;
    printf("# of open threads = %i\n", seed);
  }
  if (argc >= 2) {
    N = atoi(argv[1]);
    if (N < 1 || N > MAXN) {
      printf("N = %i is out of range.\n", N);
      exit(0);
    }
  }
  else {
    printf("Usage: %s <matrix_dimension> <# of openmp threads> [random seed]\n",
           argv[0]);
    exit(0);
  }

  /* Print parameters */
  printf("\nMatrix dimension N = %i.\n", N);
}


/* Print input matrices */
void print_inputs(float** A, float* B) {
  int row, col;
 

    printf("\nA =\n\t");
    for (row = 0; row < MIN(N, 10); row++) {
      for (col = 0; col < MIN(N, 10); col++) {
	printf("%5.2f%s", A[row][col], (col < N-1) ? ", " : ";\n\t");
      }
    }
    printf("\nB = [");
    for (col = 0; col < MIN(N, 10); col++) {
      printf("%5.2f%s", B[col], (col < N-1) ? "; " : "]\n");
    }
}

void print_X(float* X) {
  int row;

    printf("\nX = [");
    for (row = 0; row < MIN(N, 10); row++) {
      printf("%5.2f%s", X[row], (row < N-1) ? "; " : "]\n");
    }
}
/* Initialize A and B (and X to 0.0s) */
void initialize_inputs() {
  int row, col;
  printf("\nInitializing...\n");
  for (row = 0; row < N; row++) {
    for (col = 0; col < N; col++) {
      if(row == 0 || col == 0){
        cpu_A[row][col] = tpu_A[row][col] = 1;
      }else{
        cpu_A[row][col] = tpu_A[row][col] = cpu_A[row-1][col-1] + cpu_A[row-1][col];
      }
    }
    cpu_B[row] = tpu_B[row] = cpu_A[row][N-1];
    cpu_X[row] = tpu_X[row] = 0.0;
  }
//  exit(0);
}

void compare_X(float* cpu_X, float* tpu_X){
  int cnt = 0;
  for(int i = 0 ; i < N ; i++){
    if(abs(cpu_X[i] - tpu_X[i]) > 1e-5){
      cnt += 1;
      if(cnt < 10){
        printf("wrong: cpu_X[%d]: %f | tpu_X[%d]: %f\n", i, cpu_X[i], i , tpu_X[i]);
      }
    }else{
      if(i < 10){
        printf("cpu_X[%d]: %f | tpu_X[%d]: %f\n", i, cpu_X[i], i , tpu_X[i]);
      }
    }
  }
//  if(cnt == 0){
//    printf("Verify pass!\n");
//  }else{
//    printf("Verify fail, (%d/%d)\n", cnt, N);
//  }

  double avg = 0;
  double rate = 0;
  double square_sum_avg = 0;
  for(int i = 0 ; i < N ; i++){
    avg = (avg * i + cpu_X[i] ) / (i+1);
    rate =( rate * i + (fabs(tpu_X[i] - cpu_X[i]))) / (i+1);
    square_sum_avg = (square_sum_avg * i + pow((cpu_X[i] - tpu_X[i]), 2)) / (i+1);
  }
  double RMSE = sqrt(square_sum_avg);
  std::cout << "RMSE: " << RMSE << ", blas_c avg: " << avg << ", RMSE pecentage: " << (RMSE/avg)*100 << "%" << ", error rate: " << (rate/avg)*100 << "%" << std::endl;

}
void mul(int* rows_int, int* mf_int, int* rows_int2, int A, int B){
  for(int i = 0 ; i < A ; i++){
    for(int j= 0 ; j < B ; j++){
      rows_int[i*B+j] = mf_int[i*B+j] * rows_int2[i*B+j];
    }
  }
}

int main(int argc, char **argv) {
  /* Timing variables */
  struct timeval etstart, etstop;  /* Elapsed times using gettimeofday() */
  struct timezone tzdummy;
  clock_t etstart2, etstop2;  /* Elapsed times using times() */
  unsigned long long usecstart, usecstop;
  struct tms cputstart, cputstop;  /* CPU times for my processes */

  cpu_A = (float**) malloc(MAXN*sizeof(float));
  tpu_A = (float**) malloc(MAXN*sizeof(float));
  cpu_B = (float*)  malloc(MAXN*sizeof(float));
  tpu_B = (float*)  malloc(MAXN*sizeof(float));
  cpu_X = (float*)  malloc(MAXN*sizeof(float));
  tpu_X = (float*)  malloc(MAXN*sizeof(float));
  for(int i = 0 ; i < MAXN ; i++){
    cpu_A[i] = (float*) malloc(MAXN*sizeof(float));
    tpu_A[i] = (float*) malloc(MAXN*sizeof(float));
  }

  /* Process program parameters */
  parameters(argc, argv);

  /* Initialize A and B */
  initialize_inputs();

  /* Print input matrices */
  print_inputs(cpu_A, cpu_B);

  /* Start Clock */
  printf("\nStarting clock.\n");
  gettimeofday(&etstart, &tzdummy);
  etstart2 = times(&cputstart);

  /* Gaussian Elimination */
  gauss();
  print_X(cpu_X);
  gauss_tpu();
  /* Stop Clock */
  gettimeofday(&etstop, &tzdummy);
  etstop2 = times(&cputstop);
  printf("Stopped clock.\n");
  usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
  usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;

  /* Compare output */
  compare_X(cpu_X, tpu_X);
  //compare_X(cpu_B, tpu_B);

  /* Display timing results */
  printf("\nElapsed time = %g ms.\n",
	 (float)(usecstop - usecstart)/(float)1000);

  printf("(CPU times are accurate to the nearest %g ms)\n",
	 1.0/(float)CLOCKS_PER_SEC * 1000.0);
  printf("My total CPU time for parent = %g ms.\n",
	 (float)( (cputstop.tms_utime + cputstop.tms_stime) -
		  (cputstart.tms_utime + cputstart.tms_stime) ) /
	 (float)CLOCKS_PER_SEC * 1000);
  printf("My system CPU time for parent = %g ms.\n",
	 (float)(cputstop.tms_stime - cputstart.tms_stime) /
	 (float)CLOCKS_PER_SEC * 1000);
  printf("My total CPU time for child processes = %g ms.\n",
	 (float)( (cputstop.tms_cutime + cputstop.tms_cstime) -
		  (cputstart.tms_cutime + cputstart.tms_cstime) ) /
	 (float)CLOCKS_PER_SEC * 1000);
      /* Contrary to the man pages, this appears not to include the parent */
  printf("--------------------------------------------\n");

  exit(0);
}

/* ------------------ Above Was Provided --------------------- */

/****** You will replace this routine with your own parallel version *******/
/* Provided global variables are MAXN, N, A[][], B[], and X[],
 * defined in the beginning of this code.  X[] is initialized to zeros.
 */
void gauss() {
  int norm, row, col;  /* Normalization row, and zeroing
			* element row and col */
  float multiplier;
  
  printf("Computing Serially.\n");

  omp_set_num_threads(NumThreads);
  /* Gaussian elimination */
  timing f_s = clk::now();
  for (norm = 0; norm < N - 1; norm++) {
    #pragma omp parallel for shared(cpu_A, cpu_B) private(multiplier,row,col)
    for (row = norm + 1; row < N; row++) {
      multiplier = cpu_A[row][norm] / cpu_A[norm][norm];
      for (col = norm; col < N; col++) {
                 //printf("row: %d, col: %d, norm: %d\n", row, col, norm);
//                 printf("\tA[%d][%d] -= A[%d][%d] * m\n", row, col, norm, col);
	         cpu_A[row][col] -= cpu_A[norm][col] * multiplier;
      }
//      printf("B[%d] -= B[%d] * m (norm=%d)\n", row, norm, norm);
      cpu_B[row] -= cpu_B[norm] * multiplier;
    }
//    for(int i = 0 ; i < N ; i++){
//      printf("B[%d]= %f (norm=%d)\n", i, cpu_B[i], norm);
//    }
  }
  timing f_e = clk::now();
  /* (Diagonal elements are not normalized to 1.  This is treated in back
   * substitution.)
   */
  print_inputs(cpu_A, cpu_B);
//  printf("print A and B before back subsitution\n");
//  print_inputs(cpu_A, cpu_B);

  /* Back substitution */
  timing b_s = clk::now();
  for (row = N - 1; row >= 0; row--) {
    cpu_X[row] = cpu_B[row];
    for (col = N-1; col > row; col--) {
      cpu_X[row] -= cpu_A[row][col] * cpu_X[col];
    }
    cpu_X[row] /= cpu_A[row][row];
  }
  timing b_e = clk::now();
  double f_us = std::chrono::duration_cast<std::chrono::nanoseconds>(f_e-f_s).count()/1000.0;
  double b_us = std::chrono::duration_cast<std::chrono::nanoseconds>(b_e-b_s).count()/1000.0;
//  print_X(cpu_X);
  printf("forward  time: %12.3f (us)\n", f_us);
  printf("backward time: %12.3f (us)\n", b_us);
  printf("total    time: %12.3f (us)\n", f_us+b_us);
  
}


void gauss_tpu() {
  int norm, row, col;  /* Normalization row, and zeroing
			* element row and col */
  float multiplier;
  printf("======================================\n");
  printf("Computing Serially. Start gauss_tpu...\n");
//  print_inputs(tpu_A, tpu_B);

//  omp_set_num_threads(NumThreads);
  /* Gaussian elimination */
  float* rows = (float*) malloc((N-1)*(N+1)*sizeof(float));
  float* mf   = (float*) malloc((N-1)*(N+1)*sizeof(float));

  timing mfs, mfe, rs, re, ks, ke, ws, we;
  double mfus = 0, rus = 0, kus = 0, wus = 0;
  long long int cnt = 0;
  timing f_s = clk::now();
  float MMAX = FLT_MIN, MMIN = FLT_MAX;
  float RMAX = FLT_MIN, RMIN = FLT_MAX;
  int* rows_int = (int*) malloc((N+1)*(N+1)*sizeof(int));
  int* mf_int   = (int*) malloc((N+1)*(N+1)*sizeof(int));
//  for (norm = 0; norm < N - 1; norm++) {
//    for (row = norm + 1; row < N; row++) {
//      multiplier = cpu_A[row][norm] / cpu_A[norm][norm];
//      for (col = norm; col < N; col++) {
//	         cpu_A[row][col] -= cpu_A[norm][col] * multiplier;
//      }
//      cpu_B[row] -= cpu_B[norm] * multiplier;
//    }
//  }


  for (norm = 0; norm < N - 1; norm++) {
    for(int i = 0 ; i < (N-norm+1); i++){
      multiplier = tpu_A[i+norm+1][norm] / tpu_A[norm][norm];
      for(int j = 0 ; j < (N-norm+1) ; j++){
        mf[i*(N-norm+1)+j] = multiplier;// same for each row
      }
    }
    for(int i = 0 ; i < (N-norm-1) ; i++){
      for(int j = 0 ; j < (N-norm) ; j++){
        rows[i*(N-norm+1)+j] = tpu_A[norm][j+norm];
      }
      rows[i*(N-norm+1)+N-norm] = tpu_B[norm];
    }
// ========== the gptpu_mul kernel ==========
    //gptpu_mul(rows_int, mf_int, rows_int, (N-norm-1), (N-norm+1));
//    mul(rows_int, mf_int, rows_int, (N-norm-1), (N-norm+1));
// ====== quantize to uint8 =============================
    MMAX = FLT_MIN, MMIN = FLT_MAX;
    RMAX = FLT_MIN, RMIN = FLT_MAX;
    for(int i = 0 ; i < (N-norm-1)*(N-norm+1) ; i++){
      if(mf[i] > MMAX){ MMAX = mf[i]; }
      if(mf[i] < MMIN){ MMIN = mf[i]; }
      if(rows[i] > RMAX){ RMAX = rows[i]; }
      if(rows[i] < RMIN){ RMIN = rows[i]; }
    }
    for(int i = 0 ; i < (N-norm-1)*(N-norm+1) ; i++){
      rows_int[i] = (int)((rows[i] / RMAX) * 255);
      mf_int[i]   = (int)((mf[i]   / MMAX) * 255);
    }
    for(int i = 0 ; i < (N-norm-1)*(N-norm+1) ; i++){
      rows[i] = rows_int[i] * mf_int[i];
    }
    for(int i = 0 ; i < (N-norm-1)*(N-norm+1) ; i++){
      rows[i] = rows[i] * (RMAX / 255) * (MMAX / 255);
    }
//    mul(rows_int, mf_int, rows_int, (N-norm-1), (N-norm+1));
    for(int i = 0 ; i < (N-norm-1); i++){
      for(int j = 0 ; j < (N-norm+1); j++){
        rows[i*(N-norm+1)+j] = mf[i*(N-norm+1)+j] * rows[i*(N-norm+1)+j];
      }
    }
// ====== end quantize ==================================
    cnt += (N-norm-1)*(N+norm+1);
// ========== write back ==========
    for(int i = 0 ; i < (N-norm-1) ; i++){
      for(int j = 0 ; j < (N-norm) ; j++){
        tpu_A[i+norm+1][j+norm] -= rows[i*(N-norm+1)+j];
      }
      tpu_B[i+norm+1] -= rows[i*(N-norm+1)+N-norm]; 
    }   
  }
//  printf("print A and B before back subsitution\n");
//  print_inputs(tpu_A, tpu_B);
  /* Back substitution */
  for (row = N - 1; row >= 0; row--) {
    tpu_X[row] = tpu_B[row];
    for (col = N-1; col > row; col--) {
      tpu_X[row] -= tpu_A[row][col] * tpu_X[col];
    }
    tpu_X[row] /= tpu_A[row][row];
  }
  //double f_us = std::chrono::duration_cast<std::chrono::nanoseconds>(f_e-f_s).count()/1000.0;
  //double b_us = std::chrono::duration_cast<std::chrono::nanoseconds>(b_e-b_s).count()/1000.0;
//  print_X(tpu_X);
//  printf("cnt = %lld\n", cnt);
//  printf("forward  time: %12.3f (us)\n", f_us);
//  printf("->mf     time: %12.3f (us)\n", mfus);
//  printf("->rows   time: %12.3f (us)\n", rus);
//  printf("->kernel time: %12.3f (us)\n", kus);
//  printf("->WB     time: %12.3f (us)\n", wus);
//  printf("backward time: %12.3f (us)\n", b_us);
//  printf("total    time: %12.3f (us)\n", f_us+b_us);
}
    





