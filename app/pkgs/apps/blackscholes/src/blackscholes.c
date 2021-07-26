// Copyright (c) 2007 Intel Corp.

// Black-Scholes
// Analytical method for calculating European Options
//
// 
// Reference Source: Options, Futures, and Other Derivatives, 3rd Edition, Prentice 
// Hall, John C. Hull,

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <math.h>
#include <iomanip>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <assert.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <gptpu.h>
//#define ERR_CHK
#include <float.h>

typedef std::chrono::time_point<std::chrono::high_resolution_clock> timing;
typedef std::chrono::high_resolution_clock clk;

#ifdef ENABLE_PARSEC_HOOKS
#include <hooks.h>
#endif

// Multi-threaded pthreads header
#ifdef ENABLE_THREADS
// Add the following line so that icc 9.0 is compatible with pthread lib.
#define __thread __threadp
MAIN_ENV
#undef __thread
#endif

// Multi-threaded OpenMP header
#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

#ifdef ENABLE_TBB
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/tick_count.h"

using namespace std;
using namespace tbb;
#endif //ENABLE_TBB

// Multi-threaded header for Windows
#ifdef WIN32
#pragma warning(disable : 4305)
#pragma warning(disable : 4244)
#include <windows.h>
#endif

//Precision to use for calculations
#define fptype float

#define NUM_RUNS 1

typedef struct OptionData_ {
        fptype s;          // spot price
        fptype strike;     // strike price
        fptype r;          // risk-free interest rate
        fptype divq;       // dividend rate
        fptype v;          // volatility
        fptype t;          // time to maturity or option expiration in years 
                           //     (1yr = 1.0, 6mos = 0.5, 3mos = 0.25, ..., etc)  
        char OptionType;   // Option type.  "P"=PUT, "C"=CALL
        fptype divs;       // dividend vals (not used in this test)
        fptype DGrefval;   // DerivaGem Reference Value
} OptionData;

OptionData *data;
fptype *prices;
fptype *prices_ref;
int numOptions;

int    * otype;
fptype * sptprice;
fptype * strike;
fptype * rate;
fptype * volatility;
fptype * otime;
int numError = 0;
int nThreads;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Cumulative Normal Distribution Function
// See Hull, Section 11.8, P.243-244
#define inv_sqrt_2xPI 0.39894228040143270286

fptype CNDF ( fptype InputX ) 
{
    int sign;

    fptype OutputX;
    fptype xInput;
    fptype xNPrimeofX;
    fptype expValues;
    fptype xK2;
    fptype xK2_2, xK2_3;
    fptype xK2_4, xK2_5;
    fptype xLocal, xLocal_1;
    fptype xLocal_2, xLocal_3;

    // Check for negative value of InputX
    if (InputX < 0.0) {
        InputX = -InputX;
        sign = 1;
    } else 
        sign = 0;

    xInput = InputX;
 
    // Compute NPrimeX term common to both four & six decimal accuracy calcs
    expValues = exp(-0.5f * InputX * InputX);
    xNPrimeofX = expValues;
    xNPrimeofX = xNPrimeofX * inv_sqrt_2xPI;

    xK2 = 0.2316419 * xInput;
    xK2 = 1.0 + xK2;
    xK2 = 1.0 / xK2;
    xK2_2 = xK2 * xK2;
    xK2_3 = xK2_2 * xK2;
    xK2_4 = xK2_3 * xK2;
    xK2_5 = xK2_4 * xK2;
    
    xLocal_1 = xK2 * 0.319381530;
    xLocal_2 = xK2_2 * (-0.356563782);
    xLocal_3 = xK2_3 * 1.781477937;
    xLocal_2 = xLocal_2 + xLocal_3;
    xLocal_3 = xK2_4 * (-1.821255978);
    xLocal_2 = xLocal_2 + xLocal_3;
    xLocal_3 = xK2_5 * 1.330274429;
    xLocal_2 = xLocal_2 + xLocal_3;

    xLocal_1 = xLocal_2 + xLocal_1;
    xLocal   = xLocal_1 * xNPrimeofX;
    xLocal   = 1.0 - xLocal;

    OutputX  = xLocal;
    
    if (sign) {
        OutputX = 1.0 - OutputX;
    }
    
    return OutputX;
} 

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
fptype BlkSchlsEqEuroNoDiv( fptype sptprice,
                            fptype strike, fptype rate, fptype volatility,
                            fptype time, int otype, float timet )
{
    fptype OptionPrice;

    // local private working variables for the calculation
    fptype xStockPrice;
    fptype xStrikePrice;
    fptype xRiskFreeRate;
    fptype xVolatility;
    fptype xTime;
    fptype xSqrtTime;

    fptype logValues;
    fptype xLogTerm;
    fptype xD1; 
    fptype xD2;
    fptype xPowerTerm;
    fptype xDen;
    fptype d1;
    fptype d2;
    fptype FutureValueX;
    fptype NofXd1;
    fptype NofXd2;
    fptype NegNofXd1;
    fptype NegNofXd2;    
    
    xStockPrice = sptprice;
    xStrikePrice = strike;
    xRiskFreeRate = rate;
    xVolatility = volatility;

    xTime = time;
    xSqrtTime = sqrt(xTime);

    logValues = log( sptprice / strike );
        
    xLogTerm = logValues;
        
    
    xPowerTerm = xVolatility * xVolatility;
    xPowerTerm = xPowerTerm * 0.5;
        
    xD1 = xRiskFreeRate + xPowerTerm;
    xD1 = xD1 * xTime;
    xD1 = xD1 + xLogTerm;

    xDen = xVolatility * xSqrtTime;
    xD1 = xD1 / xDen;
    xD2 = xD1 -  xDen;

    d1 = xD1;
    d2 = xD2;
    
    NofXd1 = CNDF( d1 );
    NofXd2 = CNDF( d2 );

    FutureValueX = strike * ( exp( -(rate)*(time) ) );        
    if (otype == 0) {            
        OptionPrice = (sptprice * NofXd1) - (FutureValueX * NofXd2);
    } else { 
        NegNofXd1 = (1.0 - NofXd1);
        NegNofXd2 = (1.0 - NofXd2);
        OptionPrice = (FutureValueX * NegNofXd2) - (sptprice * NegNofXd1);
    }
    
    return OptionPrice;
}

#ifdef ENABLE_TBB
struct mainWork {
  mainWork() {}
  mainWork(mainWork &w, tbb::split) {}

  void operator()(const tbb::blocked_range<int> &range) const {
    fptype price;
    int begin = range.begin();
    int end = range.end();

    for (int i=begin; i!=end; i++) {
      /* Calling main function to calculate option value based on 
       * Black & Scholes's equation.
       */

      price = BlkSchlsEqEuroNoDiv( sptprice[i], strike[i],
                                   rate[i], volatility[i], otime[i], 
                                   otype[i], 0);
      prices[i] = price;

#ifdef ERR_CHK 
      fptype priceDelta = data[i].DGrefval - price;
      if( fabs(priceDelta) >= 1e-5 ){
        fprintf(stderr,"Error on %d. Computed=%.5f, Ref=%.5f, Delta=%.5f\n",
               i, price, data[i].DGrefval, priceDelta);
        numError ++;
      }
#endif
    }
  }
};

#endif // ENABLE_TBB

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_TBB
int bs_thread(void *tid_ptr) {
    int j;
    tbb::affinity_partitioner a;

    mainWork doall;
    for (j=0; j<NUM_RUNS; j++) {
      tbb::parallel_for(tbb::blocked_range<int>(0, numOptions), doall, a);
    }

    return 0;
}
#else // !ENABLE_TBB

void fp2fixed(float* in_array, int* out_fixed, int size, float& scale){
  float min = FLT_MAX, max = FLT_MIN;
  for(int i = 0 ; i < size; i++){
    if(in_array[i] > max){ max = in_array[i]; continue; }
    if(in_array[i] < min){ min = in_array[i]; continue; }
  }
//  printf("max: %f, min: %f\n", max, min);
  float range = max - min;
  for(int i = 0 ; i < size; i++){
    out_fixed[i] = (int)((float)((float)(in_array[i] - min)/(float)(max - min))*255);
  }
  scale = (float)(255)/(float)max;
}

#ifdef WIN32
DWORD WINAPI bs_thread(LPVOID tid_ptr){
#else
int bs_thread(void *tid_ptr) {
#endif
    int i, j;
    fptype price;
    fptype priceDelta;
    int tid = *(int *)tid_ptr;
    int start = tid * (numOptions / nThreads);
    int end = start + (numOptions / nThreads);

    timing ss, ee, tpu_s, tpu_e, cndf_s, cndf_e;
    double us = 0, tpu_us = 0, cndf_us = 0;
 
    for (j=0; j<NUM_RUNS; j++) {
#ifdef ENABLE_OPENMP
#pragma omp parallel for private(i, price, priceDelta)
        for (i=0; i<numOptions; i++) {
#else  //ENABLE_OPENMP
        for (i=start; i<end; i++) {
#endif //ENABLE_OPENMP
            /* Calling main function to calculate option value based on 
             * Black &/ Scholes's equation.
             */
            ss = clk::now();
            price = BlkSchlsEqEuroNoDiv( sptprice[i], strike[i],
                                         rate[i], volatility[i], otime[i], 
                                         otype[i], 0);
            prices_ref[i] = price;
            ee = clk::now();
            us += std::chrono::duration_cast<std::chrono::nanoseconds>(ee-ss).count()/1000.0;
#ifdef ERR_CHK
//            priceDelta = data[i].DGrefval - price;
//            if( fabs(priceDelta) >= 1e-4 && numError < 20){
//                printf("Error on %d. Computed=%.5f, Ref=%.5f, Delta=%.5f\n",
//                       i, price, data[i].DGrefval, priceDelta);
//                numError ++;
//            }
#endif
        }
    }
//    printf("openmp time: %12.3f (us) \n", us);
   
// ===================================my modify part starts =============================
    fptype* xSqrtTime    = (fptype*) malloc(numOptions*sizeof(fptype));
    fptype* logValues    = (fptype*) malloc(numOptions*sizeof(fptype));
    fptype* xPowerTerm   = (fptype*) malloc(numOptions*sizeof(fptype));
    fptype* xD1          = (fptype*) malloc(numOptions*sizeof(fptype));
    fptype* xD2          = (fptype*) malloc(numOptions*sizeof(fptype));
    fptype* xDen         = (fptype*) malloc(numOptions*sizeof(fptype));
    fptype* NofXd1       = (fptype*) malloc(numOptions*sizeof(fptype));
    fptype* NofXd2       = (fptype*) malloc(numOptions*sizeof(fptype));
    fptype* FutureValueX = (fptype*) malloc(numOptions*sizeof(fptype));
    fptype* OptionPrice  = (fptype*) malloc(numOptions*sizeof(fptype));
    
    ss = clk::now();
    for (j=0; j<NUM_RUNS; j++) {
//#pragma omp parallel for private(i, price, priceDelta)
        for (i=0; i<numOptions; i++) {
//            price = BlkSchlsEqEuroNoDiv( sptprice[i], strike[i],
//                                         rate[i], volatility[i], otime[i], 
//                                         otype[i], 0);
          xSqrtTime[i] = sqrt(otime[i]);
        }
//printf("numOptions: %d\n", numOptions);
        for (i=0; i<numOptions; i++) {
          logValues[i] = log(sptprice[i] / strike[i]);
//          volatility[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
//          rate[i]       = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
//          otime[i]      = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
//          logValues[i]  = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
        
//        for(i=0; i<10 ;i++){
//          printf("v[]: %f, r[]: %f, o[]: %f, log[]: %f (spt[]: %f, strike[]: %f)\n", volatility[i], rate[i], otime[i], logValues[i], sptprice[i], strike[i]);
//        }
        tpu_s = clk::now();

/* baseline start*/
        for (i=0; i<numOptions; i++) {
          xPowerTerm[i] = 0.5* (volatility[i] * volatility[i]);
//          if(i == 0){printf("xPowerTerm: %f = 0.5 * %f * %f\n", xPowerTerm[i], volatility[i], volatility[i]);}
        }
        for (i=0; i<numOptions; i++) {
          xD1[i] = ((rate[i] + xPowerTerm[i]) * otime[i]) + logValues[i];
//          if(i == 0){printf("xD1: %f = ((%f + %f) * %f) + %f\n", xD1[i], rate[i], xPowerTerm[i], otime[i], logValues[i]);}
        }
        tpu_e   = clk::now();
/* baseline end*/
/* fixed point start*/
        int* fixed_v = (int*) malloc(numOptions*sizeof(int));
        int* fixed_r = (int*) malloc(numOptions*sizeof(int));
        int* fixed_o = (int*) malloc(numOptions*sizeof(int));
        int* fixed_l = (int*) malloc(numOptions*sizeof(int));
        float scale_v, scale_r, scale_o, scale_l;
        fp2fixed(volatility, fixed_v, numOptions, scale_v);
        fp2fixed(rate,       fixed_r, numOptions, scale_r);
        fp2fixed(otime,      fixed_o, numOptions, scale_o);
        fp2fixed(logValues,  fixed_l, numOptions, scale_l);

//        for(i=0; i< 10; i++){
//          printf("v: %d, scale: %f\n", fixed_v[i], scale_v);
//        }
//        for(i=0; i< 10; i++){
//          printf("r: %d, scale: %f\n", fixed_r[i], scale_r);
//        }
//        for(i=0; i< 10; i++){
//          printf("o: %d, scale: %f\n", fixed_o[i], scale_o);
//        }
//        for(i=0; i< 10; i++){
//          printf("l: %d, scale: %f\n", fixed_l[i], scale_l);
//        }
              
        for (i=0; i<numOptions; i++) {
         // xPowerTerm[i] = 0.5* (volatility[i] * volatility[i]);
          xPowerTerm[i] = 0.5 * (float)(fixed_v[i] * fixed_v[i]) * (1.0/scale_v * 1.0/scale_v);
 //         if(i <= 10){printf("xPowerTerm: %f = 0.5 * (%d * %d) * (%f * %f)\n", xPowerTerm[i], fixed_v[i], fixed_v[i], 1.0/scale_v, 1.0/scale_v);}
        }
        for (i=0; i<numOptions; i++) {
          //xD1[i] = ((rate[i] + xPowerTerm[i]) * otime[i]) + logValues[i];
          xD1[i] = ((((float)fixed_r[i] * 1.0/scale_r) + xPowerTerm[i]) * (float)fixed_o[i] * 1.0/scale_o) + logValues[i];
 //         if(i <= 10){printf("xD1: %f = ((%d * %f + %f) * %d * %f) + %f\n", xD1[i], fixed_r[i], 1.0/scale_r, xPowerTerm[i], fixed_o[i], 1.0/scale_o, logValues[i]);}
        }
/* fixed point end*/
//        for(i=0; i<10;i++){
//          printf("xP[]: %f, xD[]: %f\n", xPowerTerm[i], xD1[i]);
//        }
        tpu_us += std::chrono::duration_cast<std::chrono::nanoseconds>(tpu_e-tpu_s).count()/1000.0;
        for (i=0; i<numOptions; i++) {
          xDen[i] = volatility[i] * xSqrtTime[i];
        }
        for (i=0; i<numOptions; i++) {
          xD1[i] = xD1[i] / xDen[i];
        }
        for (i=0; i<numOptions; i++) {
          xD2[i] = xD1[i] - xDen[i];
        }
        cndf_s = clk::now();

//        for(float i = 0 ; i < 4.9 ; i+=0.01){
//          printf("CNDF(%f) = %f\n", i, CNDF(i));
//        }
        for (i=0; i<numOptions; i++) {
          NofXd1[i] = CNDF(xD1[i]);
        }
        for (i=0; i<numOptions; i++) {
          NofXd2[i] = CNDF(xD2[i]);
        }
        cndf_e = clk::now();
        cndf_us = std::chrono::duration_cast<std::chrono::nanoseconds>(cndf_e-cndf_s).count()/1000.0;
        for (i=0; i<numOptions; i++) {
          FutureValueX[i] = strike[i] * (exp( -(rate[i]) * (otime[i])));
        }
        for (i=0; i<numOptions; i++) {
          if(otype[i] == 0){
            OptionPrice[i] = (sptprice[i] * NofXd1[i]) - (FutureValueX[i] * NofXd2[i]);
          }else{
            OptionPrice[i] = (FutureValueX[i] * (1.0 - NofXd2[i])) - (sptprice[i] * (1.0 - NofXd1[i]));
          }
          prices[i] = OptionPrice[i];
        }


           //prices[i] = price;


    }
    ee = clk::now();
    us = std::chrono::duration_cast<std::chrono::nanoseconds>(ee-ss).count()/1000.0;
    double MSE = 0;
    double rate = 0;
    double mean = 0;
    for (i=0; i<numOptions; i++) {
#ifdef ERR_CHK
            priceDelta =/* data[i].DGrefval*/prices_ref[i] - prices[i];
            if( fabs(priceDelta) >= 1e-4  && numError < 20){
                printf("Error on %d. Computed=%.5f, Ref=%.5f, Delta=%.5f\n",
                       i, prices[i], /*data[i].DGrefval*/prices_ref[i], priceDelta);
                numError ++;
            }
#endif
      //if(i < 100) printf("prices_ref: %f, pirces: %f\n", prices_ref[i], prices[i]);
      MSE = (MSE * i + pow(/*data[i].DGrefval*/prices_ref[i] - prices[i], 2)) / (i+1);
      mean = (mean * i + /*data[i].DGrefval*/prices_ref[i]) / (i+1);
      rate = (rate * i + fabs(/*data[i].DGrefval*/prices_ref[i] - prices[i])) / (i + 1);
//      printf("prices_ref[%d]: %f, prices: %f\n", i, prices_ref[i], prices[i]);
    }
    printf("RMSE: %f, mean: %f, RMSE%%: %f %%, error rate: %f %%\n", sqrt(MSE), mean, (sqrt(MSE)/mean)*100, (rate/mean)*100);
// ===================================my modify part ends ===============================
//    printf("gptpu  time: %12.3f (us) \n", us);
//    printf("kernel time: %12.3f (us) \n", tpu_us);
//    printf("cndf   time: %12.3f (us) \n", cndf_us);
//    printf("other  time: %12.3f (us) \n", us - tpu_us - cndf_us);
    return 0;
}
#endif //ENABLE_TBB

int main (int argc, char **argv)
{
    FILE *file;
    int i;
    int loopnum;
    fptype * buffer;
    int * buffer2;
    int rv;

#ifdef PARSEC_VERSION
#define __PARSEC_STRING(x) #x
#define __PARSEC_XSTRING(x) __PARSEC_STRING(x)
        printf("PARSEC Benchmark Suite Version "__PARSEC_XSTRING(PARSEC_VERSION)"\n");
	fflush(NULL);
#else
        printf("PARSEC Benchmark Suite\n");
	fflush(NULL);
#endif //PARSEC_VERSION
#ifdef ENABLE_PARSEC_HOOKS
   __parsec_bench_begin(__parsec_blackscholes);
#endif

   if (argc != 4)
        {
                printf("Usage:\n\t%s <nthreads> <inputFile> <outputFile>\n", argv[0]);
                exit(1);
        }
    nThreads = atoi(argv[1]);
    char *inputFile = argv[2];
    char *outputFile = argv[3];
    //Read input data from file
    file = fopen(inputFile, "r");
    if(file == NULL) {
      printf("ERROR: Unable to open file `%s'.\n", inputFile);
      exit(1);
    }
    rv = fscanf(file, "%i", &numOptions);
    if(rv != 1) {
      printf("ERROR: Unable to read from file `%s'.\n", inputFile);
      fclose(file);
      exit(1);
    }
    if(nThreads > numOptions) {
      printf("WARNING: Not enough work, reducing number of threads to match number of options.\n");
      nThreads = numOptions;
    }

//    int fd = open(outputFile, O_RDWR | O_CREAT | O_TRUNC, 0777);
//    size_t out_size = 9*numOptions*sizeof(fptype);
//    printf("out_size: %d\n", out_size);
//    char* dst;
//    dst = static_cast<char*>(mmap(NULL, out_size, PROT_WRITE, MAP_PRIVATE, fd, 0));
//    assert(dst != MAP_FAILED);
//    if(ftruncate(fd, out_size) != 0){ printf("fruncate fail\n"); exit(0); }

#if !defined(ENABLE_THREADS) && !defined(ENABLE_OPENMP) && !defined(ENABLE_TBB)
    if(nThreads != 1) {
        printf("Error: <nthreads> must be 1 (serial version), nThreads = %d\n", nThreads);
        exit(1);
    }
#endif

    // alloc spaces for the option data
    data = (OptionData*)malloc(numOptions*sizeof(OptionData));
    prices = (fptype*)malloc(numOptions*sizeof(fptype));
    prices_ref = (fptype*)malloc(numOptions*sizeof(fptype));
    for ( loopnum = 0; loopnum < numOptions; ++ loopnum )
    {
        rv = fscanf(file, "%f %f %f %f %f %f %c %f %f", &data[loopnum].s, &data[loopnum].strike, &data[loopnum].r, &data[loopnum].divq, &data[loopnum].v, &data[loopnum].t, &data[loopnum].OptionType, &data[loopnum].divs, &data[loopnum].DGrefval);
        if(rv != 9) {
          printf("ERROR: Unable to read from file `%s'.\n", inputFile);
          fclose(file);
          exit(1);
        }
// TODO : save data back in binary format with new file name
       // char* buf = reinterpret_cast<char*>(&data[loopnum].s, sizeof(fptype));
//        memcpy(dst+loopnum*9*sizeof(fptype)+0*sizeof(fptype), reinterpret_cast<char*>(&data[loopnum].s), sizeof(fptype));
//        memcpy(dst+loopnum*9*sizeof(fptype)+1*sizeof(fptype), reinterpret_cast<char*>(&data[loopnum].strike), sizeof(fptype));
//        memcpy(dst+loopnum*9*sizeof(fptype)+2*sizeof(fptype), reinterpret_cast<char*>(&data[loopnum].r), sizeof(fptype));
//        memcpy(dst+loopnum*9*sizeof(fptype)+3*sizeof(fptype), reinterpret_cast<char*>(&data[loopnum].divq), sizeof(fptype));
//        memcpy(dst+loopnum*9*sizeof(fptype)+4*sizeof(fptype), reinterpret_cast<char*>(&data[loopnum].v), sizeof(fptype));
//        memcpy(dst+loopnum*9*sizeof(fptype)+5*sizeof(fptype), reinterpret_cast<char*>(&data[loopnum].t), sizeof(fptype));
//        memcpy(dst+loopnum*9*sizeof(fptype)+6*sizeof(fptype), reinterpret_cast<char*>(&data[loopnum].OptionType), sizeof(fptype));
//        memcpy(dst+loopnum*9*sizeof(fptype)+7*sizeof(fptype), reinterpret_cast<char*>(&data[loopnum].divs), sizeof(fptype));
//        memcpy(dst+loopnum*9*sizeof(fptype)+8*sizeof(fptype), reinterpret_cast<char*>(&data[loopnum].DGrefval), sizeof(fptype));
    }
//    munmap(dst, out_size);
//    close(fd);
    rv = fclose(file);
    if(rv != 0) {
      printf("ERROR: Unable to close file `%s'.\n", inputFile);
      exit(1);
    }

#ifdef ENABLE_THREADS
    MAIN_INITENV(,8000000,nThreads);
#endif
//    printf("Num of Options: %d\n", numOptions);
//    printf("Num of Runs: %d\n", NUM_RUNS);

#define PAD 256
#define LINESIZE 64

    buffer = (fptype *) malloc(5 * numOptions * sizeof(fptype) + PAD);
    sptprice = (fptype *) (((unsigned long long)buffer + PAD) & ~(LINESIZE - 1));
    strike = sptprice + numOptions;
    rate = strike + numOptions;
    volatility = rate + numOptions;
    otime = volatility + numOptions;

    buffer2 = (int *) malloc(numOptions * sizeof(fptype) + PAD);
    otype = (int *) (((unsigned long long)buffer2 + PAD) & ~(LINESIZE - 1));

    for (i=0; i<numOptions; i++) {
        otype[i]      = (data[i].OptionType == 'P') ? 1 : 0;
        sptprice[i]   = data[i].s;
        strike[i]     = data[i].strike;
        rate[i]       = data[i].r;
        volatility[i] = data[i].v;    
        otime[i]      = data[i].t;
    }

//    printf("Size of data: %ld\n", numOptions * (sizeof(OptionData) + sizeof(int)));

#ifdef ENABLE_PARSEC_HOOKS
    __parsec_roi_begin();
#endif

#ifdef ENABLE_THREADS
#ifdef WIN32
    HANDLE *threads;
    int *nums;
    threads = (HANDLE *) malloc (nThreads * sizeof(HANDLE));
    nums = (int *) malloc (nThreads * sizeof(int));

    for(i=0; i<nThreads; i++) {
        nums[i] = i;
        threads[i] = CreateThread(0, 0, bs_thread, &nums[i], 0, 0);
    }
    WaitForMultipleObjects(nThreads, threads, TRUE, INFINITE);
    free(threads);
    free(nums);
#else
    int *tids;
    tids = (int *) malloc (nThreads * sizeof(int));

    for(i=0; i<nThreads; i++) {
        tids[i]=i;
        CREATE_WITH_ARG(bs_thread, &tids[i]);
    }
    WAIT_FOR_END(nThreads);
    free(tids);
#endif //WIN32
#else //ENABLE_THREADS
#ifdef ENABLE_OPENMP
    {
        int tid=0;
//         printf("for openmp: # of threads = %d\n", nThreads);
        omp_set_num_threads(nThreads);
        bs_thread(&tid);
    }
#else //ENABLE_OPENMP
#ifdef ENABLE_TBB
    tbb::task_scheduler_init init(nThreads);

    int tid=0;
    bs_thread(&tid);
#else //ENABLE_TBB
    //serial version
    int tid=0;
    bs_thread(&tid);
#endif //ENABLE_TBB
#endif //ENABLE_OPENMP
#endif //ENABLE_THREADS

#ifdef ENABLE_PARSEC_HOOKS
    __parsec_roi_end();
#endif

    //Write prices to output file
//    file = fopen(outputFile, "w");
//    if(file == NULL) {
//      printf("ERROR: Unable to open file `%s'.\n", outputFile);
//      exit(1);
//    }
//    rv = fprintf(file, "%i\n", numOptions);
//    if(rv < 0) {
//      printf("ERROR: Unable to write to file `%s'.\n", outputFile);
//      fclose(file);
//      exit(1);
//    }
//    for(i=0; i<numOptions; i++) {
//      rv = fprintf(file, "%.18f\n", prices[i]);
//      if(rv < 0) {
//        printf("ERROR: Unable to write to file `%s'.\n", outputFile);
//        fclose(file);
//        exit(1);
//      }
//    }
//    rv = fclose(file);
//    if(rv != 0) {
//      printf("ERROR: Unable to close file `%s'.\n", outputFile);
//      exit(1);
//    }

#ifdef ERR_CHK
    printf("Num Errors: %d\n", numError);
#endif
    free(data);
    free(prices);
    free(prices_ref);

#ifdef ENABLE_PARSEC_HOOKS
    __parsec_bench_end();
#endif

    return 0;
}

