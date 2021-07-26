/*
 * =====================================================================================
 *
 *       Filename:  suite.c
 *
 *    Description:  The main wrapper for the suite
 *
 *        Version:  1.0
 *        Created:  10/22/2009 08:40:34 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Liang Wang (lw2aw), lw2aw@virginia.edu
 *        Company:  CS@UVa
 *
 * =====================================================================================
 */

#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <assert.h>
#include <chrono>
#include "common.h"

static int do_verify = 1;
int omp_num_threads = 1;

static struct option long_options[] = {
  /* name, has_arg, flag, val */
  {"input", 1, NULL, 'i'},
  {"size", 1, NULL, 's'},
  {"verify", 0, NULL, 'v'},
  {"BS", 1, NULL, 'a'},
  {0,0,0,0}
};

extern void lud_omp(  int *m, int matrix_dim, int BS);
extern void lud_gptpu(int *m, int matrix_dim, int BS);
extern void lud_omp_int(int *m, int matrix_dim, int BS);

typedef std::chrono::time_point<std::chrono::high_resolution_clock> timing;
typedef std::chrono::high_resolution_clock clk;
int
main ( int argc, char *argv[] )
{
  int matrix_dim = 1024; /* default size */
  int BS         = 256;
  int opt, option_index=0;
  func_ret_t ret;
  const char *input_file = NULL;
  int *m, *mm, *m_gptpu;
  int verbose = 0;
	
  while ((opt = getopt_long(argc, argv, "::vs:n:i:a", 
                            long_options, &option_index)) != -1 ) {
    switch(opt){
    case 'i':
      input_file = optarg;
      break;
    case 'v':
      verbose = 1;
      break;
    case 'n':
      omp_num_threads = atoi(optarg);
      break;
    case 'a':
      BS = atoi(optarg);
      break;
    case 's':
      matrix_dim = atoi(optarg);
//      BS = matrix_dim/2;
      printf("Generate input matrix internally, size =%d, BS = %d\n", matrix_dim, BS);
      // fprintf(stderr, "Currently not supported, use -i instead\n");
      // fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
      // exit(EXIT_FAILURE);
      break;
    case '?':
      fprintf(stderr, "invalid option\n");
      break;
    case ':':
      fprintf(stderr, "missing argument\n");
      break;
    default:
      fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n",
	      argv[0]);
      exit(EXIT_FAILURE);
    }
  }
  
  if ( (optind < argc) || (optind == 1)) {
    fprintf(stderr, "Usage: %s [-v] [-n no. of threads] [-s matrix_size|-i input_file]\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  if (input_file) {
    printf("Reading matrix from file %s\n", input_file);
    ret = create_matrix_from_file(&m, input_file, &matrix_dim);
    if (ret != RET_SUCCESS) {
      m = NULL;
      fprintf(stderr, "error create matrix from file %s\n", input_file);
      exit(EXIT_FAILURE);
    }
  }
  else if (matrix_dim) {
    printf("Creating matrix internally size=%d\n", matrix_dim);
    ret = create_matrix(&m, matrix_dim);
    ret = create_matrix(&m_gptpu, matrix_dim);
    if (ret != RET_SUCCESS) {
      m = NULL;
      m_gptpu = NULL;
      fprintf(stderr, "error create matrix internally size=%d\n", matrix_dim);
      exit(EXIT_FAILURE);
    }
  }
 
  else {
    printf("No input file specified!\n");
    exit(EXIT_FAILURE);
  } 

  if (do_verify){
    printf("Before LUD\n");
    /* print_matrix(m, matrix_dim); */
    matrix_duplicate(m,       &mm, matrix_dim);
    matrix_duplicate(m_gptpu, &mm, matrix_dim);
  }

  timing b_s = clk::now();
//  lud_omp(  m,       matrix_dim, BS);
  timing b_e = clk::now();
  timing g_s = clk::now();
  lud_omp_int(m_gptpu, matrix_dim, BS);
  timing g_e = clk::now();

  if (do_verify){
    printf("After LUD\n");
    /* print_matrix(m, matrix_dim); */
//    printf("Verifying baseline... \n");
//    lud_verify(mm, m, matrix_dim, verbose); 
    printf("Verifying gptpu version... \n");
    lud_verify(mm, m_gptpu, matrix_dim, verbose); 
    free(mm);
  }

  double baseline_us = std::chrono::duration_cast<std::chrono::nanoseconds>(b_e - b_s).count()/1000.0;
  double gptpu_us = std::chrono::duration_cast<std::chrono::nanoseconds>(g_e - g_s).count()/1000.0;
  
  printf("CPU time: %12.3f (us)\n", baseline_us);
  printf("TPU time: %12.3f (us)\n", gptpu_us);

  free(m);
  free(m_gptpu);
  return EXIT_SUCCESS;
}				/* ----------  end of function main  ---------- */
