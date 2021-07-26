#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "common.h"

void stopwatch_start(stopwatch *sw){
    if (sw == NULL)
        return;

    bzero(&sw->begin, sizeof(struct timeval));
    bzero(&sw->end  , sizeof(struct timeval));

    gettimeofday(&sw->begin, NULL);
}

void stopwatch_stop(stopwatch *sw){
    if (sw == NULL)
        return;

    gettimeofday(&sw->end, NULL);
}

double 
get_interval_by_sec(stopwatch *sw){
    if (sw == NULL)
        return 0;
    return ((double)(sw->end.tv_sec-sw->begin.tv_sec)+(double)(sw->end.tv_usec-sw->begin.tv_usec)/1000000);
}

int 
get_interval_by_usec(stopwatch *sw){
    if (sw == NULL)
        return 0;
    return ((sw->end.tv_sec-sw->begin.tv_sec)*1000000+(sw->end.tv_usec-sw->begin.tv_usec));
}

func_ret_t 
create_matrix_from_file(int **mp, const char* filename, int *size_p){
  int i, j, size;
  int *m;
  FILE *fp = NULL;

  fp = fopen(filename, "rb");
  if ( fp == NULL) {
      return RET_FAILURE;
  }

  fscanf(fp, "%d\n", &size);

  m = (int*) malloc(sizeof(int)*size*size);
  if ( m == NULL) {
      fclose(fp);
      return RET_FAILURE;
  }

  for (i=0; i < size; i++) {
      for (j=0; j < size; j++) {
          fscanf(fp, "%d ", m+i*size+j);
      }
  }

  fclose(fp);

  *size_p = size;
  *mp = m;

  return RET_SUCCESS;
}


func_ret_t
create_matrix_from_random(float **mp, int size){
  float *l, *u, *m;
  int i,j,k;

  srand(time(NULL));

  l = (float*)malloc(size*size*sizeof(float));
  if ( l == NULL)
    return RET_FAILURE;

  u = (float*)malloc(size*size*sizeof(float));
  if ( u == NULL) {
      free(l);
      return RET_FAILURE;
  }

  for (i = 0; i < size; i++) {
      for (j=0; j < size; j++) {
          if (i>j) {
              l[i*size+j] = GET_RAND_FP;
          } else if (i == j) {
              l[i*size+j] = 1;
          } else {
              l[i*size+j] = 0;
          }
      }
  }

  for (j=0; j < size; j++) {
      for (i=0; i < size; i++) {
          if (i>j) {
              u[j*size+i] = 0;
          }else {
              u[j*size+i] = GET_RAND_FP; 
          }
      }
  }

  for (i=0; i < size; i++) {
      for (j=0; j < size; j++) {
          for (k=0; k <= MIN(i,j); k++)
            m[i*size+j] = l[i*size+k] * u[j*size+k];
      }
  }

  free(l);
  free(u);

  *mp = m;

  return RET_SUCCESS;
}

void
matrix_multiply(float *inputa, float *inputb, float *output, int size){
  int i, j, k;

  for (i=0; i < size; i++)
    for (k=0; k < size; k++)
      for (j=0; j < size; j++)
        output[i*size+j] = inputa[i*size+k] * inputb[k*size+j];

}

func_ret_t
lud_verify(int *m, int *lu, int matrix_dim, int verbose){
  int i,j,k;
  int *tmp = (int*)malloc(matrix_dim*matrix_dim*sizeof(int));
  int *L   = (int*)malloc(matrix_dim*matrix_dim*sizeof(int));
  int *U   = (int*)malloc(matrix_dim*matrix_dim*sizeof(int));

  for (i=0; i < matrix_dim; i ++)
    for (j=0; j< matrix_dim; j++) {
        int sum = 0;
        int l,u;
        for (k=0; k <= MIN(i,j); k++){
            if ( i==k)
              l=1;
            else
              l=lu[i*matrix_dim+k];
            u=lu[k*matrix_dim+j];
            L[i*matrix_dim+k] = l;
            U[k*matrix_dim+j] = u;
            sum+=l*u;
        }
        tmp[i*matrix_dim+j] = sum;
    }
  /* printf(">>>>>LU<<<<<<<\n"); */
  /* for (i=0; i<matrix_dim; i++){ */
  /*   for (j=0; j<matrix_dim;j++){ */
  /*       printf("%f ", lu[i*matrix_dim+j]); */
  /*   } */
  /*   printf("\n"); */
  /* } */
  if(verbose != 0){
    printf("myprint L:\n");
    for(i = 0 ; i < matrix_dim ; i++){
      for(j = 0 ; j < matrix_dim ; j++){
        if(i >= j){
          printf("%3d ", L[i*matrix_dim+j]);
        }else{
          printf("--- ");
        }
      }
    }
    printf("\nmyprint U:\n");
    for(i = 0 ; i < matrix_dim ; i++){
      for(j = 0 ; j < matrix_dim ; j++){
        if(i >= j){
          printf("%3d ", U[i*matrix_dim+j]);
        }else{
          printf("--- ");
        }
      }
    }
    printf("\n>>>>>result<<<<<<<\n"); 
    for (i=0; i<matrix_dim; i++){ 
      for (j=0; j<matrix_dim;j++){ 
          printf("%d ", tmp[i*matrix_dim+j]); 
      } 
      printf("\n"); 
    } 
    printf("\n>>>>>input<<<<<<<\n");
    for (i=0; i<matrix_dim; i++){ 
      for (j=0; j<matrix_dim;j++){ 
        printf("%d ", m[i*matrix_dim+j]); 
      } 
      printf("\n"); 
    }
  } // end verbose 
  int error_cnt = 0;
  int b = 0;
  double avg = 0;
  double rate = 0;
  double square_sum_avg = 0;
  int idx = 0;
  for (i=0; i<matrix_dim; i++){
      for (j=0; j<matrix_dim; j++){
          if ( m[i*matrix_dim+j] != tmp[i*matrix_dim+j]){
            if(b < 10)printf("dismatch at (%d, %d): (o)%d (n)%d\n", i, j, m[i*matrix_dim+j], tmp[i*matrix_dim+j]);
            error_cnt++;
            b++;
          }
          idx = i*matrix_dim+j;
          avg = (avg * idx + m[i*matrix_dim+j]) / (idx+ 1);
          rate = ( rate * idx + (fabs(m[i*matrix_dim+j] - tmp[i*matrix_dim+j]))) / (idx + 1);
          square_sum_avg = (square_sum_avg * idx + pow((m[i*matrix_dim+j] - tmp[i*matrix_dim+j]), 2)) / (idx+1); 
      }
  }
  if(error_cnt == 0){
    printf("verify pass!\n");
  }else{
    printf("verify fail, error count: %d (%d)\n", error_cnt, matrix_dim*matrix_dim);
  }
  double RMSE = sqrt(square_sum_avg);
  std::cout << "RMSE: " << RMSE << ", blas_c avg: " << avg << ", RMSE pecentage: " << (RMSE/avg)*100 << "%" << ", error rate: " << (rate/avg)*100     << "%" << std::endl;
  free(tmp);
}

void
matrix_duplicate(int *src, int **dst, int matrix_dim) {
    int s = matrix_dim*matrix_dim*sizeof(int);
   int *p = (int *) malloc (s);
   memcpy(p, src, s);
   *dst = p;
}

void
print_matrix(float *m, int matrix_dim) {
    int i, j;
    for (i=0; i<matrix_dim;i++) {
      for (j=0; j<matrix_dim;j++)
        printf("%f ", m[i*matrix_dim+j]);
      printf("\n");
    }
}


// Generate well-conditioned matrix internally  by Ke Wang 2013/08/07 22:20:06

func_ret_t
create_matrix(int **mp, int size){
  int *m;
  int i,j;
  float lamda = -0.001;
  float coe[2*size-1];
  float coe_i =0.0;

  for (i=0; i < size; i++)
    {
      coe_i = 10*exp(lamda*i); 
      j=size-1+i;     
      coe[j]=coe_i;
      j=size-1-i;     
      coe[j]=coe_i;
    }

  m = (int*) malloc(sizeof(int)*size*size);
  int* L = (int*) malloc(sizeof(int)*size*size);
  int* U = (int*) malloc(sizeof(int)*size*size);
  if ( m == NULL) {
      return RET_FAILURE;
  }

  for (i=0; i < size; i++) {
      for (j=0; j < size; j++) {
        if(i > j){
          L[i*size+j] = (rand() % 127)+1;
        }else if(i == j){
          L[i*size+j] = 1;
        }else{
          L[i*size+j] = 0;
        }
      }
  }
  for (i=0; i < size; i++) {
      for (j=0; j < size; j++) {
        if(i <= j){
          U[i*size+j] = (rand() % 127)+1;
        }else{
          U[i*size+j] = 0;
        }
      }
  }


  for (i=0; i < size; i++) {
      for (j=0; j < size; j++) {
	m[i*size+j]=coe[size-1-i+j];
        m[i*size+j] = (i>=j)?(j+1):(i+1);
        
      }
  }
  for (i=0; i < size; i++) {
      for (j=0; j < size; j++) {
         for(int k = 0 ; k < size; k++){
//         m[i*size+k] = L[i*size+j] * U[j*size+k];
//printf("m[%d,%d]: %d, L: %d, U: %d\n", i, k, m[i*size+k], L[i*size+j], U[j*size+k]);
         } 
      }
  }

  *mp = m;

  return RET_SUCCESS;
}
