//Copyright (c) 2009 Princeton University
//Written by Christian Bienia
//Generate input files for blackscholes benchmark

#include <stdio.h>
#include <stdlib.h>



//Precision to use
#define fptype double

typedef struct OptionData_ {
        fptype s;          // spot price
        fptype strike;     // strike price
        fptype r;          // risk-free interest rate
        fptype divq;       // dividend rate
        fptype v;          // volatility
        fptype t;          // time to maturity or option expiration in years 
                           //     (1yr = 1.0, 6mos = 0.5, 3mos = 0.25, ..., etc)  
        const char *OptionType;  // Option type.  "P"=PUT, "C"=CALL
        fptype divs;       // dividend vals (not used in this test)
        fptype DGrefval;   // DerivaGem Reference Value
} OptionData;

//Total number of options in optionData.txt
#define MAX_OPTIONS 1000

OptionData data_init[] = {
    #include "optionData.txt"
};



int main (int argc, char **argv) {
  int numOptions;
  char *fileName;
  int rv;
  int i;

  if (argc != 3) {
    printf("Usage:\n\t%s <numOptions> <fileName>\n", argv[0]);
    exit(1);
  }
  numOptions = atoi(argv[1]);
  fileName = argv[2];
  if(numOptions < 1) {
    printf("ERROR: Number of options must at least be 1.\n");
    exit(1);
  }

  FILE *file;
  file = fopen(fileName, "w");
  if(file == NULL) {
    printf("ERROR: Unable to open file `%s'.\n", fileName);
    exit(1);
  }

  //write number of options
  rv = fprintf(file, "%i\n", numOptions);
  if(rv < 0) {
    printf("ERROR: Unable to write to file `%s'.\n", fileName);
    fclose(file);
    exit(1);
  }

#define RAND(a) (float)rand()/((float)RAND_MAX/a)
#define IN RAND(65536)//(214783648)
  //write values for options
  srand(time(NULL));
  int t = 0;
  int w = 0;
  for(i=0; i<numOptions; i++) {
    t = rand();
    w = (rand() % MAX_OPTIONS) + 1;
    //NOTE: DG RefValues specified exceed double precision, output will deviate
    //rv = fprintf(file, "%.2f %.2f %.4f %.2f %.2f %.2f %c %.2f %.18f\n", data_init[i % MAX_OPTIONS].s, data_init[i % MAX_OPTIONS].strike, data_init[i % MAX_OPTIONS].r, data_init[i % MAX_OPTIONS].divq, data_init[i % MAX_OPTIONS].v, data_init[i % MAX_OPTIONS].t, data_init[i % MAX_OPTIONS].OptionType[0], data_init[i % MAX_OPTIONS].divs, data_init[i % MAX_OPTIONS].DGrefval);
    rv = fprintf(file, "%.2f %.2f %.4f %.2f %.2f %.2f %c %.2f %.18f\n", IN, IN, data_init[i % MAX_OPTIONS].r, data_init[i % MAX_OPTIONS].divq, data_init[i % MAX_OPTIONS].v, data_init[i % MAX_OPTIONS].t, data_init[i % MAX_OPTIONS].OptionType[0], data_init[i % MAX_OPTIONS].divs, data_init[i % MAX_OPTIONS].DGrefval);




    //rv = fprintf(file, "%.2f %.2f %.4f %.2f %.2f %.2f %c %.2f %.18f\n", data_init[t % MAX_OPTIONS].s, data_init[t % MAX_OPTIONS].strike, data_init[t % MAX_OPTIONS].r, data_init[t % MAX_OPTIONS].divq, data_init[t % MAX_OPTIONS].v, data_init[t % MAX_OPTIONS].t, data_init[t % MAX_OPTIONS].OptionType[0], data_init[t % MAX_OPTIONS].divs, data_init[t % MAX_OPTIONS].DGrefval);
    //rv = fprintf(file, "%.2f %.2f %.4f %.2f %.2f %.2f %c %.2f %.18f\n", data_init[t % w].s, data_init[t % w].strike, data_init[t % w].r, data_init[t % w].divq, data_init[t % w].v, data_init[t % w].t, data_init[t % w].OptionType[0], data_init[t % w].divs, data_init[t % w].DGrefval);
    if(rv < 0) {
      printf("ERROR: Unable to write to file `%s'.\n", fileName);
      fclose(file);
      exit(1);
    }
  }

  rv = fclose(file);
  if(rv != 0) {
    printf("ERROR: Unable to close file `%s'.\n", fileName);
    exit(1);
  }

  return 0;
}
