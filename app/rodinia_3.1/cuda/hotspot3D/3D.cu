#include <iostream>
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <stdlib.h> 
#include <gptpu.h>
#include <math.h> 
#include <sys/time.h>
#include <chrono>
#include <float.h>
#define BLOCK_SIZE 16
#define STR_SIZE 256

#define block_x_ 128 
#define block_y_ 2
#define block_z_ 1
#define MAX_PD	(3.0e6)
/* required precision in degrees	*/
#define PRECISION	0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5
#include "opt1.cu"

typedef std::chrono::time_point<std::chrono::high_resolution_clock> timing;
typedef std::chrono::high_resolution_clock clk;
/* chip parameters	*/
float t_chip = 0.0005;
float chip_height = 0.016; float chip_width = 0.016; /* ambient temperature, assuming no package at all	*/
float amb_temp = 80.0;

double get_RMSE(float* a, float* b, int size){
  double square_sum_avg = 0;
  for(int i = 0 ; i < size ; i++){
    square_sum_avg = (square_sum_avg * i + pow((a[i] - b[i]), 2)) / (i+1);
  }
  return sqrt(square_sum_avg);
}

double get_error_rate(float* a, float* b, int size){
  double rate = 0;
  for(int i = 0 ; i < size ; i++){
    rate = (rate * (double)i + fabs((double)a[i] - (double)b[i])) / (i+1);
  }
  return rate;
}

double get_mean(float* a, int size){
  double sum = 0;
  for(int i = 0 ; i < size ; i++){
    sum += a[i];
  }
  return sum/(double)size;
}

        //run_us[z] += gptpu_conv2D(tIn_int[z], filter, tOut_int[z], nx, ny, 3, 3, "replication"); // TODO: an scale factor for fixed point scaling

void conv2D_cpu(int* in, int* filter, int* out, int A, int B){

  int up, down, left, right;
  for(int i = 0 ; i < A ; i++){
    for(int j = 0 ; j < B ; j++){
      up   = (i > 0  )?in[(i-1)*B+j]:in[i*B+j];
      down = (i < A-1)?in[(i+1)*B+j]:in[i*B+j];
      left = (j > 0  )?in[i*B+j-1]:in[i*B+j];
      left = (j < B-1)?in[i*B+j+1]:in[i*B+j];

      out[i*B+j] = 
                   up        * filter[1] +
                 
                   left      * filter[3] +
                   in[i*B+j] * filter[4] +
                   right     * filter[5] +
                 
                   down      * filter[7];
                 
    }
  }
}



void fatal(const char *s)
{
    fprintf(stderr, "Error: %s\n", s);
}

void readinput(float *vect, int grid_rows, int grid_cols, int layers, char *file) {
    int i,j,k;
    FILE *fp;
    char str[STR_SIZE];
    float val;

    if( (fp  = fopen(file, "r" )) ==0 )
      fatal( "The file was not opened" );


    for (i=0; i <= grid_rows-1; i++) 
      for (j=0; j <= grid_cols-1; j++)
        for (k=0; k <= layers-1; k++)
          {
            if (fgets(str, STR_SIZE, fp) == NULL) fatal("Error reading file\n");
            if (feof(fp))
              fatal("not enough lines in file");
            if ((sscanf(str, "%f", &val) != 1))
              fatal("invalid file format");
            vect[i*grid_cols+j+k*grid_rows*grid_cols] = val;
          }

    fclose(fp);	

}


void writeoutput(float *vect, int grid_rows, int grid_cols, int layers, char *file) {

    int i,j,k, index=0;
    FILE *fp;
    char str[STR_SIZE];

    if( (fp = fopen(file, "w" )) == 0 )
      printf( "The file was not opened\n" );

    for (i=0; i < grid_rows; i++) 
      for (j=0; j < grid_cols; j++)
        for (k=0; k < layers; k++)
          {
            sprintf(str, "%d\t%g\n", index, vect[i*grid_cols+j+k*grid_rows*grid_cols]);
            fputs(str,fp);
            index++;
          }

    fclose(fp);	
}

void computeTempTPU(float *pIn/*powerIn*/, float* tIn/*tempCpoy*/, float *tOut/*answer*/, 
        int nx/*numCols*/, int ny/*numRows*/, int nz/*layers*/, float Cap, 
        float Rx, float Ry, float Rz, 
        float dt, int numiter) 
{   float ce, cw, cn, cs, ct, cb, cc;
    float stepDivCap = dt / Cap;
    ce = cw =stepDivCap/ Rx;
    cn = cs =stepDivCap/ Ry;
    ct = cb =stepDivCap/ Rz;

    cc = 1.0 - (2.0*ce + 2.0*cn + 3.0*ct);
    printf("ce: %f, cw: %f, cn: %f, cs: %f, ct: %f, cb: %f, cc: %f, dt/Cap: %f\n", ce, cw, cn, cs, ct, cb, cc, stepDivCap);
    int /*c, unused*/w,e,n,s,b,t;
    int x,y,z;
    int i = 0, idx;
    int size = nx*ny*nz;
    double allocate_us = 0, copy_us = 0, core_us = 0, shift_us = 0;
    timing a_s, a_e, m_s, m_e, c_s, c_e, s_s, s_e;
    a_s = clk::now();
    float* coef   = (float*)malloc(7*sizeof(float));
    float* layout = (float*)malloc(size*7*sizeof(float));
    a_e = clk::now();
    allocate_us += std::chrono::duration_cast<std::chrono::nanoseconds>(a_e-a_s).count()/1000.0;
    m_s = clk::now();
      coef[0] = cc;
      coef[1] = cn;  coef[2] = cs;  coef[3] = ce;
      coef[4] = cw;  coef[5] = ct;  coef[6] = cb;
    m_e = clk::now();
    copy_us += std::chrono::duration_cast<std::chrono::nanoseconds>(m_e-m_s).count()/1000.0;
    do{
      m_s = clk::now();
      for(z = 0 ; z < nz; z++){
        for(y = 0 ; y < ny ; y++){
          for(x = 0 ;x < nx ; x++){
            idx = x + y * nx + z * nx * ny;
            w = (x == 0)?      idx : idx-1;
            e = (x == nx - 1)? idx : idx+1;
            n = (y == 0)?      idx : idx - nx;
            s = (y == ny -1)?  idx : idx + nx;
            b = (z == 0)?      idx : idx - nx * ny;
            t = (z == nz - 1)? idx : idx + nx * ny;
            layout[idx*7+0] = tIn[idx];
            layout[idx*7+1] = tIn[n];
            layout[idx*7+2] = tIn[s];
            layout[idx*7+3] = tIn[e];
            layout[idx*7+4] = tIn[w];
            layout[idx*7+5] = tIn[t];
            layout[idx*7+6] = tIn[b];
          }
        }
      }
      m_e = clk::now();
      copy_us += std::chrono::duration_cast<std::chrono::nanoseconds>(m_e-m_s).count()/1000.0;
      c_s = clk::now();
      for(idx = 0 ; idx < size ; idx++){
        tOut[idx] = layout[idx*7+0]*coef[0]
                  + layout[idx*7+1]*coef[1]
                  + layout[idx*7+2]*coef[2]
                  + layout[idx*7+3]*coef[3]
                  + layout[idx*7+4]*coef[4]
                  + layout[idx*7+5]*coef[5]
                  + layout[idx*7+6]*coef[6];
      }
//      gptpu_mv(layout, coef, tOut, size, 7);
      c_e = clk::now();
      core_us += std::chrono::duration_cast<std::chrono::nanoseconds>(c_e-c_s).count()/1000.0;
      s_s = clk::now();
      for(idx = 0 ; idx < size ; idx++){
        tOut[idx] += (dt/Cap) * pIn[idx] + ct*amb_temp;
      }
      s_e = clk::now();
      shift_us += std::chrono::duration_cast<std::chrono::nanoseconds>(s_e-s_s).count()/1000.0;
      float *temp = tIn;
      tIn = tOut;
      tOut = temp;
      i++; 
    }while(i < numiter);
  a_s = clk::now();
  free(coef);
  free(layout);
  a_e = clk::now();
  allocate_us += std::chrono::duration_cast<std::chrono::nanoseconds>(a_e-a_s).count()/1000.0;
  printf("allocate time: %12.3f (us)\n", allocate_us);
  printf("layout   time: %12.3f (us)\n", copy_us);
  printf("core MV  time: %12.3f (us) can be replaced with GPTPU_mv\n", core_us);
  printf("ADD      time: %12.3f (us) can be replaced with GPTPU_add\n", shift_us);
}

void computeTempTPU_conv(float *pIn/*powerIn*/, float* tIn/*tempCpoy*/, float *tOut/*answer*/, 
        int nx/*numCols*/, int ny/*numRows*/, int nz/*layers*/, float Cap, 
        float Rx, float Ry, float Rz, 
        float dt, int numiter) 
{   float ce, cw, cn, cs, ct, cb, cc;
    float stepDivCap = dt / Cap;
    ce = cw =stepDivCap/ Rx;
    cn = cs =stepDivCap/ Ry;
    ct = cb =stepDivCap/ Rz;

    cc = 1.0 - (2.0*ce + 2.0*cn + 3.0*ct) ;//+ /*only when z == 1*/(ct + cb);

    ce = cw = cn = cs = 0.136533;
    cc = 0.453667;
    cb = ct = 0.000067;
    stepDivCap = 1.365333;

    int c,w,e,n,s,b,t;
    int x,y,z;
    int i = 0, j = 0;

    int** pIn_int   = (int**)  malloc(nz*sizeof(int*));
    int** tIn_int   = (int**)  malloc(nz*sizeof(int*));
    int** tOut_int  = (int**)  malloc(nz*sizeof(int*));
    float* pIn_max  = (float*) malloc(nz*sizeof(float));
    float* tIn_max  = (float*) malloc(nz*sizeof(float));
    float* pIn_min  = (float*) malloc(nz*sizeof(float));
    float* tIn_min  = (float*) malloc(nz*sizeof(float));
    float* in_scale = (float*) malloc(nz*sizeof(float));
    float* coef_a   = (float*) malloc(nz*sizeof(float));
    float* coef_b   = (float*) malloc(nz*sizeof(float));
    for(int j = 0 ; j < nz ; j++){
      pIn_int[j]  = (int*) malloc(nx*ny*sizeof(int));
      tIn_int[j]  = (int*) malloc(nx*ny*sizeof(int));
      tOut_int[j] = (int*) malloc(nx*ny*sizeof(int));
      pIn_max[j]  = FLT_MIN, tIn_max[j] = FLT_MIN;
      pIn_min[j]  = FLT_MAX, tIn_min[j] = FLT_MAX;
      for(i = 0 ; i < nx*ny ; i++){
        pIn_int[j][i]  = pIn[j*nx*ny+i];
        tIn_int[j][i]  = tIn[j*nx*ny+i];
        if(pIn[j*nx*ny+i] > pIn_max[j]){pIn_max[j] = pIn[j*nx*ny+i];}
        if(tIn[j*nx*ny+i] > tIn_max[j]){tIn_max[j] = tIn[j*nx*ny+i];}
        if(pIn[j*nx*ny+i] < pIn_min[j]){pIn_min[j] = pIn[j*nx*ny+i];}
        if(tIn[j*nx*ny+i] < tIn_min[j]){tIn_min[j] = tIn[j*nx*ny+i];}
      }
      in_scale[j] = (float)(UCHAR_MAX)/(float)tIn_max[j];
      coef_a[j]   = (float)(UCHAR_MAX)/(float)(tIn_max[j] - tIn_min[j]);
      coef_b[j]   = (float)(UCHAR_MAX * tIn_min[j])/(float)(tIn_max[j] - tIn_min[j]);
    } 
    for(int j = 0 ; j < nz ; j++){
      for(int i = 0 ; i < nx*ny ; i++){
        tIn_int[j][i] = (int)((float)((float)(tIn[j*nx*ny+i] - tIn_min[j])/(float)(tIn_max[j] - tIn_min[j]))*UCHAR_MAX); 
      }
    }
//    printf("pIn_max: %f, pIn_min: %f, tIn_max: %f, tIn_min: %f\n", pIn_max, pIn_min, tIn_max, tIn_min);
// gptpu_conv2D(int* in, int* out, int* filter, int IN_W, int IN_H);
// Assume F_W == F_H == 3 with 'SAME' padding width == 1.
// The 3x3 filter has 4 zeros on corners, 5 elements elsewhere.

// If z == 1, then gptpu_add() two more arrays (tIn[t] and tIn[b]), which si not incoulded in gptpu_conv2D()
// So let's taget at hotspot2D first, where z == 1, such that gptpu_conv2D() can be directly used.

// Notes that:
// in and out dims are the same
    float max_coef  = MAX(MAX(MAX(MAX(ce, cw), cn), cs), cc);
    float min_coef  = MIN(MIN(MIN(MIN(ce, cw), cn), cs), cc);
    float range_coef = max_coef - min_coef;
    float f_scale = (float)UCHAR_MAX/(float)max_coef; 
    printf("%d, %d, f_scale = %f\n",   (int)((float)((float)ce/(float)max_coef)*UCHAR_MAX), UCHAR_MAX, f_scale); 
    int* filter = (int*) malloc(9*sizeof(int));
    filter[0] = filter[2] = filter[6] = filter[8] = 0;
    filter[1] = (int)((float)((float)cn/(float)max_coef)*UCHAR_MAX);// cn
    filter[3] = (int)((float)((float)cw/(float)max_coef)*UCHAR_MAX);// cw
    filter[4] = (int)((float)((float)cc/(float)max_coef)*UCHAR_MAX);// cc
    filter[5] = (int)((float)((float)ce/(float)max_coef)*UCHAR_MAX);// ce
    filter[7] = (int)((float)((float)cs/(float)max_coef)*UCHAR_MAX);// cs
    int f_sum  = 0;
    for(int i = 0 ; i < 9 ; i++){f_sum += filter[i];} // for re-construct tOut[i]
    printf("ce: %f, cw: %f, cn: %f, cs: %f, cc: %f | cb: %f, ct: %f\n", ce, cw, cn, cs, cc, cb, ct);
    printf("ce: %f, cw: %f, cn: %f, cs: %f, cc: %f | cb: %f, ct: %f, dt/Cap: %f\n", ce, cw, cn, cs, cc, cb, ct, stepDivCap);

    //set_breakdown(1);
    float* the_scale = (float*) malloc(nz*sizeof(float));
    for(int i = 0 ; i < nz ; i++){
      the_scale[i] = ((float)1.0/(float(in_scale[i]*f_scale)))*0.7; 
    } 
    float* run_us = (float*) malloc(nz*sizeof(float));
    for(int i = 0 ; i < nz ; i++){
      run_us[i] = 0;
    }
    double gptpu_us = 0, reverse_us = 0, rest_us = 0;
    timing gptpu_s, gptpu_e, reverse_s, reverse_e, rest_s, rest_e;
    i = 0 ;
    do{
      for(z = 0 ; z < nz; z++){
        //set_scale(the_scale[z]); // need to set a scale to avoid overflow and rescaling back to float 
        //scale = tIn_scale * f_scale * (avoid flow controlling scale)
        gptpu_s = clk::now();
        //run_us[z] += gptpu_conv2D(tIn_int[z], filter, tOut_int[z], nx, ny, 3, 3, "replication"); // TODO: an scale factor for fixed point scaling
        //conv2D_cpu(tIn_int[z], filter, tOut_int[z], nx, ny); // TODO: an scale factor for fixed point scaling
        gptpu_e = clk::now();
        gptpu_us   += std::chrono::duration_cast<std::chrono::nanoseconds>(gptpu_e - gptpu_s).count()/1000.0;  
      }
      printf("ce: %f, cw: %f, cn: %f, cs: %f, cc: %f\n", ce, cw, cn, cs, cc);
      for(z = 0 ; z < nz; z++){
// ============ result ================================
// tOut[i] = (1/(a * f_scale)) * tOut_int[i] + ((b * f_int_sum)/(a * f_scale))
// ====================================================
        reverse_s = clk::now();
        for(j = 0 ; j < nx*ny ; j++ ){
          tOut_int[z][j] /= the_scale[z]; // valid iff use the re-constructed tOut , percentage error comes from here
          tOut[z*nx*ny+j]  = ((float)1.0/(float)(coef_a[z] * f_scale)) * tOut_int[z][j] + ((float)coef_b[z]/(float)(coef_a[z] * f_scale)) * f_sum;
          //if(j < 20) printf("tOut[%2d] = %f, tOut_int[%2d]: %d\n", j , tOut[j], j, tOut_int[z][j]);
        }
        reverse_e = clk::now();
        rest_s = clk::now();
        for(y = 0 ; y < ny ; y++){
          for(x = 0 ;x < nx ; x++){
            c = x + y * nx + z * nx * ny;
//            w = (x == 0)?      c : c-1;
//            e = (x == nx - 1)? c : c+1;
//            n = (y == 0)?      c : c - nx;
//            s = (y == ny - 1)? c : c + nx;
            b = (z == 0)?      c : c - nx * ny;
            t = (z == nz - 1)? c : c + nx * ny;
            tOut[c] += /*tIn[c]*cc + tIn[n]*cn + tIn[s]*cs + tIn[e]*ce + tIn[w]*cw +*//*when z == 1*/tIn[t]*ct + tIn[b]*cb + /*(dt/Cap)*/stepDivCap * pIn[c] + ct*amb_temp;
          }
        }
        rest_e = clk::now();
        reverse_us += std::chrono::duration_cast<std::chrono::nanoseconds>(reverse_e - reverse_s).count()/1000.0;  
        rest_us    += std::chrono::duration_cast<std::chrono::nanoseconds>(rest_e - rest_s).count()/1000.0;  
      } // end of z
      int *temp;// = tIn_int;
      //tIn_int = tOut_int;
      //tOut_int = temp;
      for(j = 0 ; j < nz ; j++){
        temp = tIn_int[j];
        tIn_int[j] = tOut_int[j];
        tOut_int[j] = temp;
      }
      i++; 
    }while(i < numiter);
    for(int j = 0 ; j < nz ; j++){
      std::cout << "run_us[" << j << "]: " << run_us[j] << std::endl;
    }
    printf("gptpu   time: %12.3f (us)\n", gptpu_us);
    printf("reverse time: %12.3f (us)\n", reverse_us);
    printf("rest    time: %12.3f (us)\n", rest_us);
   
}
void computeTemp_multi_layer_TPU(float *pIn/*powerIn*/, float* tIn/*tempCpoy*/, float *tOut/*answer*/, 
        int nx/*numCols*/, int ny/*numRows*/, int nz/*layers*/, float Cap, 
        float Rx, float Ry, float Rz, 
        float dt, int numiter) 
{   float ce, cw, cn, cs, ct, cb, cc;
    float stepDivCap = dt / Cap;
    ce = cw =stepDivCap/ Rx;
    cn = cs =stepDivCap/ Ry;
    ct = cb =stepDivCap/ Rz;

    cc = 1.0 - (2.0*ce + 2.0*cn + 3.0*ct);
    ce = cw = cn = cs = 0.136533;
    cc = 0.453667;
    cb = ct = 0.000067;
    stepDivCap = 1.365333;
    int c,w,e,n,s,b,t;
    int x,y,z;
    int i = 0;
    do{
      //for(z = 0 ; z < nz; z++){
        for(y = 0 ; y < ny ; y++){
          for(x = 0 ;x < nx ; x++){
            c = x + y * nx;// + z * nx * ny;
            w = (x == 0)?      c : c-1;
            e = (x == nx - 1)? c : c+1;
            n = (y == 0)?      c : c - nx;
            s = (y == ny - 1)? c : c + nx;
            //b = (z == 0)?      c : c - nx * ny; // for nz == 1, b == c
            //t = (z == nz - 1)? c : c + nx * ny; // for nz == 1, t == c
            //tOut[c] = tIn[c]*cc + tIn[n]*cn + tIn[s]*cs + tIn[e]*ce + tIn[w]*cw + tIn[t/*t*/]*ct + tIn[b/*b*/]*cb + /*(dt/Cap)*/ stepDivCap * pIn[c] + ct*amb_temp;
            tOut[c] = tIn[c]*(cc+ct+cb) + tIn[n]*cn + tIn[s]*cs + tIn[e]*ce + tIn[w]*cw;//  + /*(dt/Cap)*/  stepDivCap * pIn[c] + 0.000067*80.0/*ct*amb_temp*/;
          }
        }
      //}
      float *temp = tIn;
      tIn = tOut;
      tOut = temp;
      i++; 
    }while(i < numiter);
    float *temp = tIn;
    tIn = tOut;
    tOut = temp;
    for(int cnt = 1 ; cnt < numiter ; cnt++){
      i=0;
      do{
        printf("CHECK\n");
        for(y = 0 ; y < ny ; y++){
          for(x = 0 ; x < nx ; x++){
            c = x + y * nx;// + z * nx * ny;
            w = (x == 0)?      c : c-1;
            e = (x == nx - 1)? c : c+1;
            n = (y == 0)?      c : c - nx;
            s = (y == ny - 1)? c : c + nx;
            tOut[c] += (stepDivCap*pIn[c]+0.000067*80.0)*(cc+ct+cb) +
                       (stepDivCap*pIn[n]+0.000067*80.0)*cn +
                       (stepDivCap*pIn[s]+0.000067*80.0)*cs +
                       (stepDivCap*pIn[e]+0.000067*80.0)*ce +
                       (stepDivCap*pIn[w]+0.000067*80.0)*cw;
          }
        }
        i++;
      }while(i<cnt);
    }
    for(c = 0 ; c < (nx*ny) ; c++){
      tOut[c] += stepDivCap*pIn[c] + 0.000067*80.0; 
    }
    temp = tIn;
    tIn = tOut;
    tOut = temp;
}
void computeTempCPU(float *pIn/*powerIn*/, float* tIn/*tempCpoy*/, float *tOut/*answer*/, 
        int nx/*numCols*/, int ny/*numRows*/, int nz/*layers*/, float Cap, 
        float Rx, float Ry, float Rz, 
        float dt, int numiter) 
{   float ce, cw, cn, cs, ct, cb, cc;
    float stepDivCap = dt / Cap;
    ce = cw =stepDivCap/ Rx;
    cn = cs =stepDivCap/ Ry;
    ct = cb =stepDivCap/ Rz;

    cc = 1.0 - (2.0*ce + 2.0*cn + 3.0*ct);
    ce = cw = cn = cs = 0.136533;
    cc = 0.453667;
    cb = ct = 0.000067;
    stepDivCap = 1.365333;
    int c,w,e,n,s,b,t;
    int x,y,z;
    int i = 0;
    do{
      for(z = 0 ; z < nz; z++){
        for(y = 0 ; y < ny ; y++){
          for(x = 0 ;x < nx ; x++){
            c = x + y * nx + z * nx * ny;
            w = (x == 0)?      c : c-1;
            e = (x == nx - 1)? c : c+1;
            n = (y == 0)?      c : c - nx;
            s = (y == ny - 1)? c : c + nx;
            b = (z == 0)?      c : c - nx * ny; // for nz == 1, b == c
            t = (z == nz - 1)? c : c + nx * ny; // for nz == 1, t == c
            tOut[c] = tIn[c]*cc + tIn[n]*cn + tIn[s]*cs + tIn[e]*ce + tIn[w]*cw + tIn[t/*c*/]*ct + tIn[b/*c*/]*cb + /*(dt/Cap)*/ stepDivCap * pIn[c] + ct*amb_temp;
          }
        }
      }
      float *temp = tIn;
      tIn = tOut;
      tOut = temp;
      i++; 
    }while(i < numiter);
}
float accuracy(float *arr1, float *arr2, int len)
{
    float err = 0.0; 
    int i;
    for(i = 0; i < len; i++)
    {
        err += (arr1[i]-arr2[i]) * (arr1[i]-arr2[i]);
    }

    return (float)sqrt(err/len);
}
 

void usage(int argc, char **argv)
{
    fprintf(stderr, "Usage: %s <rows/cols> <layers> <iterations> <powerFile> <tempFile> <outputFile>\n", argv[0]);
    fprintf(stderr, "\t<rows/cols>  - number of rows/cols in the grid (positive integer)\n");
    fprintf(stderr, "\t<layers>  - number of layers in the grid (positive integer)\n");

    fprintf(stderr, "\t<iteration> - number of iterations\n");
    fprintf(stderr, "\t<powerFile>  - name of the file containing the initial power values of each cell\n");
    fprintf(stderr, "\t<tempFile>  - name of the file containing the initial temperature values of each cell\n");
    fprintf(stderr, "\t<outputFile - output file\n");
    exit(1);
}

int main(int argc, char** argv)
{
    if (argc != 7)
    {
        usage(argc,argv);
    }

    char *pfile, *tfile, *ofile;
    int iterations = atoi(argv[3]);

    pfile = argv[4];
    tfile = argv[5];
    ofile = argv[6];
    int numCols = atoi(argv[1]);
    int numRows = atoi(argv[1]);
    int layers = atoi(argv[2]);
//    printf("size: %d, layers: %d, iter: %d\n", numCols, layers, iterations);
    /* calculating parameters*/

    float dx = chip_height/numRows;
    float dy = chip_width/numCols;
    float dz = t_chip/layers;

    float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * dx * dy;
    float Rx = dy / (2.0 * K_SI * t_chip * dx);
    float Ry = dx / (2.0 * K_SI * t_chip * dy);
    float Rz = dz / (K_SI * dx * dy);

    float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    float dt = PRECISION / max_slope;


    float *powerIn, *tempOut, *tempIn, *tempCopy, *TPUCopy, *multiCopy;
    int size = numCols * numRows * layers; // 512 * 512 *8 ~= 2M
//    std::cout << "size: " << size << ",numCols: " << numCols << ", numRows: " << numRows << ", layers: " << layers << std::endl;
    powerIn = (float*)calloc(size, sizeof(float));
    tempCopy = (float*)malloc(size * sizeof(float));
    TPUCopy = (float*)malloc(size * sizeof(float));
    multiCopy = (float*)malloc(size * sizeof(float));
    tempIn = (float*)calloc(size,sizeof(float));
    tempOut = (float*)calloc(size, sizeof(float));
    float* answer = (float*)calloc(size, sizeof(float));
    int* TPUanswer = (int*)calloc(size, sizeof(int));
    float* TPUanswer_float = (float*)calloc(size, sizeof(float));

    readinput(powerIn,numRows, numCols, layers,pfile);
    readinput(tempIn, numRows, numCols, layers, tfile);

    memcpy(tempCopy,tempIn, size * sizeof(float));
    memcpy(TPUCopy,tempIn, size * sizeof(float));
    memcpy(multiCopy,tempIn, size * sizeof(float));

    timing GPU_s = clk::now();
    hotspot_opt1(powerIn, tempIn, tempOut, numCols, numRows, layers, Cap, Rx, Ry, Rz, dt,iterations);
    timing GPU_e = clk::now();
    timing CPU_s = clk::now();
    computeTempCPU(powerIn, tempCopy,  answer, numCols, numRows, layers, Cap, Rx, Ry, Rz, dt,iterations);
    timing CPU_e = clk::now();

    char *model_name;
    model_name = (char*) malloc(100*sizeof(char));

    strcpy(model_name, "./hotspot3D_ex_model.tflite"); 

    std::cout << "start run on tpu ..." << std::endl;
    run_a_hotspot(model_name, 1, numCols/*=numRows*/, powerIn, TPUCopy, TPUanswer);


// ===== get some stats =====
    float ref_max = FLT_MIN;
    float ref_min = FLT_MAX;
    float ref_sum = 0;
    float ref_mean = 0;
    float ref_range = 0;
    int tpu_max = INT_MIN;
    int tpu_min = INT_MAX;
    int tpu_sum = 0;
    int tpu_mean = 0;
    int tpu_range = 0;
    float ftpu_max = FLT_MIN;
    float ftpu_min = FLT_MAX;
    float ftpu_sum = 0;
    float ftpu_mean = 0;
    float ftpu_range = 0;

//    std::cout << "start to collect stats..." << std::endl;   
    for(int i = 0 ; i < numCols*numRows; i++){
      if(answer[i] > ref_max){ ref_max = answer[i]; }
      if(answer[i] < ref_min){ ref_min = answer[i]; }
      if(TPUanswer[i] > tpu_max){ tpu_max = TPUanswer[i]; }
      if(TPUanswer[i] < tpu_min){ tpu_min = TPUanswer[i]; }
      ref_sum += answer[i];
      tpu_sum += TPUanswer[i];
    }
    ref_range = ref_max - ref_min;
    ref_mean = ref_sum / (numCols*numRows);
    tpu_range = tpu_max - tpu_min;
    tpu_mean = (int)(tpu_sum / (numCols*numRows));
//    std::cout << "ref max: " << ref_max << ", mean: " << ref_mean << ", min: " << ref_min << ", range: " << ref_range << std::endl;
//    std::cout << "TPU max: " << tpu_max << ", mean: " << tpu_mean << ", min: " << tpu_min << ", range: " << tpu_range << std::endl;

// scaling back to expected range ==============
//    std::cout << "start to scaling back to float" << std::endl;
    for(int i = 0 ; i < numCols*numRows ; i++){
      TPUanswer_float[i] = TPUanswer[i] - (float)tpu_mean;
      TPUanswer_float[i] = ( TPUanswer_float[i] / tpu_range ) * ref_range;
      TPUanswer_float[i] += ref_mean;
    }
    for(int i = 0 ; i < numCols*numRows ; i++){
      if(TPUanswer_float[i] > ftpu_max){ ftpu_max = TPUanswer_float[i];  }
      if(TPUanswer_float[i] < ftpu_min){ ftpu_min = TPUanswer_float[i];  }
      ftpu_sum += TPUanswer_float[i];
    }
    ftpu_range = ftpu_max - ftpu_min;
    ftpu_mean = ftpu_sum / (numCols * numRows);
//    std::cout << "fTPU max: " << ftpu_max << ", mean: " << ftpu_mean << ", min: " << ftpu_min << ", range: " << ftpu_range << std::endl;
//    std::cout << "scaled back float output from tpu: " << std::endl;
//    for(int i = 0 ; i < 10 ; i++){
//      for(int j = 0 ;  j < 10 ; j++){
//        std::cout << TPUanswer_float[i*numCols+j] << " ";
//      }
//      std::cout << std::endl;
//    }

    timing multi_s = clk::now();
//    computeTemp_multi_layer_TPU(powerIn, multiCopy,  multiTPUanswer, numCols, numRows, layers, Cap, Rx, Ry, Rz, dt,iterations);
    timing multi_e = clk::now();
    timing TPU_s = clk::now();
//    computeTempTPU_conv(powerIn, TPUCopy,TPUanswer, numCols, numRows, layers, Cap, Rx, Ry, Rz, dt,iterations);
    timing TPU_e = clk::now();
    double MSE = 0;
    double rate = 0;
    double rank1_mean = 0;
    int multi_error_cnt = 0;
    for(int i = 0 ;  i < numCols*numRows*layers ; i++){
      MSE = (MSE * i + pow(answer[i] - TPUanswer_float[i], 2)) / (i+1);
      rank1_mean = (rank1_mean * i + answer[i]) / (i+1);
      rate = (rate * i + fabs(answer[i] - TPUanswer_float[i])) / (i+1);
//      if(fabs(answer[i] - multiTPUanswer[i]) > 1E-3){
//        multi_error_cnt++;
//        if(multi_error_cnt < 10)
//          printf("multiTPUanswer[%d]: %f != answer[%d]: %f\n", i, multiTPUanswer[i], i, answer[i]);
//      }
//      if(i < 1024*1024){
//        printf("answer[%d]: %f, TPUanswer[%d]: %f, diff: %f(rate: %f), diff^2: %f\n", i, answer[i], i, TPUanswer[i], fabs(answer[i] - TPUanswer[i]), rate, pow(answer[i] - TPUanswer[i], 2));
//      }
    }
    printf("RMSE: %f, mean: %f, RMSE%%: %f %%, error rate: %f(rate: %f, mean: %f) %%\n", sqrt(MSE), rank1_mean, (sqrt(MSE)/rank1_mean)*100, (rate/rank1_mean)*100, rate, rank1_mean);
//    double RMSE       = get_RMSE(      answer, TPUanswer, numCols*numRows*layers);
//    double error_rate = get_error_rate(answer, TPUanswer, numCols*numRows*layers);
//    double CPU_avg    = get_mean(      answer,            numCols*numRows*layers); 
//    printf("CPU               mean: %f\n", CPU_avg);
//    printf("CPU and TPU       RMSE: %f over %d elements (RMSE%% = %f%%)\n", RMSE, numCols*numRows*layers, (RMSE/CPU_avg)*100);
//    printf("CPU and TPU error rate: %f %%\n", (error_rate/CPU_avg)*100);
//    double GPU_us = std::chrono::duration_cast<std::chrono::nanoseconds>(GPU_e - GPU_s).count()/1000.0;  
//    double CPU_us = std::chrono::duration_cast<std::chrono::nanoseconds>(CPU_e - CPU_s).count()/1000.0;  
//    double multi_us = std::chrono::duration_cast<std::chrono::nanoseconds>(multi_e - multi_s).count()/1000.0;  
//    double TPU_us = std::chrono::duration_cast<std::chrono::nanoseconds>(TPU_e - TPU_s).count()/1000.0;  
//    float acc = accuracy(tempOut,answer,numRows*numCols*layers);
//    printf("GPU   time: %12.3f (us)\n", GPU_us);
//    printf("multi time: %12.3f (us), multi_error_cnt = %d\n", multi_us, multi_error_cnt);
//    printf("CPU   time: %12.3f (us), GPU and CPU Accuracy: %e = %12.8f\n", CPU_us, acc, acc);
//    float acc2= accuracy(tempOut,TPUanswer_float,numRows*numCols*layers);
//    printf("TPU   time: %12.3f (us), GPU and TPU Accuracy: %e = %12.8f\n", TPU_us, acc2, acc2);
//    float ep = 1e-4;
//    bool ans = (fabs(acc - acc2) < ep)?true:false;
//    printf("TPU accuracy compare: %d with epsilon = %f\n", ans, ep);
  
//    for(int i = 0 ; i < 10 ; i++){
//      printf("powerIn: %f, tempIn: %f| CPU: %f, TPU: %f\n", powerIn[i], tempIn[i], answer[i], TPUanswer_float[i]);
//    }
//    writeoutput(tempOut,numRows, numCols, layers, ofile);

    free(tempIn);
    free(tempOut); free(powerIn);
    free(tempCopy); free(TPUCopy);  free(multiCopy);
    free(answer); free(TPUanswer);  free(TPUanswer_float); free(model_name);
    return 0;
}	


